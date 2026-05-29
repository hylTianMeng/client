"""Evaluate a trained model against the hand-crafted aggressive policy.

This runs games between:
- Model side: PUCT (PersistentMCTS) guided by the loaded model
- Opponent: StrategyFactory.get_aggressive_action_strategy()

It plays both sides (model as team1 and team2) and reports W/L/D.

Example
-------
python model_vs_aggressive.py --model training_data_bootstrap/run_20260529_114024/bootstrap_model.pt --games-per-side 10 --simulations 200 --device cuda

If you want to match the bootstrap distribution, keep --random-init on (default).
"""

from __future__ import annotations

import argparse
import random
from typing import Tuple, List, Optional

import torch
from tqdm import tqdm

import numpy as np

from env import Environment, PieceArg, Point, ActionSet, AttackContext, SpellContext, Area
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory
from strategy_utils import step_with_action

from strategy_utils import fork_environment, get_legal_moves, get_attackable_targets


def _build_init_message(env: Environment, player_id: int):
    init_message = type("InitGameMessage", (), {})()
    init_message.piece_cnt = env.player1.PIECE_CNT if player_id == 1 else env.player2.PIECE_CNT
    init_message.id = player_id
    init_message.board = env.board
    return init_message


def _wrap_piece_args(piece_args):
    policy = type("InitPolicyMessage", (), {})()
    policy.piece_args = piece_args
    return policy


def _random_aggressive_init_positions(board, player_id: int, piece_cnt: int, rng: random.Random) -> List[Point]:
    bdr = board.boarder
    candidates = []
    for y in range(board.height):
        for x in range(board.width):
            if not board.is_within_bounds(Point(x, y)):
                continue
            if board.grid[x][y].state != 1:
                continue
            if player_id == 1 and not (y < bdr):
                continue
            if player_id == 2 and not (y > bdr):
                continue
            candidates.append((x, y))

    if len(candidates) < piece_cnt:
        raise RuntimeError(f"Not enough placement cells: have {len(candidates)} need {piece_cnt}")

    rng.shuffle(candidates)
    picked = candidates[:piece_cnt]
    return [Point(x, y) for x, y in picked]


def random_aggressive_init_strategy(seed: int) -> callable:
    def strategy(init_message) -> List[PieceArg]:
        rng = random.Random(seed)
        # mix in per-call randomness so two players don't mirror perfectly
        rng.seed(seed + init_message.id * 1000003 + random.randint(0, 2**31 - 1))

        board = init_message.board
        pid = init_message.id
        positions = _random_aggressive_init_positions(board, pid, init_message.piece_cnt, rng)

        piece_args: List[PieceArg] = []
        for pos in positions:
            arg = PieceArg()
            # same stats as aggressive init
            arg.strength = 20
            arg.dexterity = 8
            arg.intelligence = 2
            arg.equip = Point(2, 3)
            arg.pos = pos
            piece_args.append(arg)
        return piece_args

    return strategy


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = float(np.sum(ex))
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x, dtype=np.float64) / x.size
    return ex / s


def _idx_of(pos: Point) -> int:
    return int(pos.x) * 20 + int(pos.y)


def _find_piece_by_id(env: Environment, piece_id: int):
    for p in env.action_queue:
        if getattr(p, "id", None) == piece_id:
            return p
    return None


def make_greedy_model_strategy(
    model: TacticalPolicyNet,
    processor: StateProcessor,
    device: torch.device,
) -> callable:
    """Fast model policy: greedily decide move->attack->spell using staged inference.

    This matches the staged data format used in training and is much faster than MCTS.
    """

    model.eval()

    def infer(env: Environment, stage_value: float) -> dict:
        state = processor.build_input(env, stage_value)
        with torch.no_grad():
            x = torch.tensor(state[None, ...], dtype=torch.float32, device=device)
            out = model(x)
        switch = _softmax_np(out["switch_logits"].detach().cpu().numpy()[0])
        move = _softmax_np(out["move_logits"].detach().cpu().numpy()[0])
        attack = _softmax_np(out["attack_logits"].detach().cpu().numpy()[0])
        raw_spell = out["spell_logits"].detach().cpu().numpy()[0]
        spell = _softmax_np(raw_spell.reshape(-1)).reshape(raw_spell.shape)
        return {"switch": switch, "move": move, "attack": attack, "spell": spell}

    def strategy(env: Environment) -> ActionSet:
        action = ActionSet()
        action.move = False
        action.attack = False
        action.spell = False

        cp = env.current_piece
        if cp is None or not getattr(cp, "is_alive", True):
            return action

        # ---- stage 0: move ----
        out0 = infer(env, 0.3)
        do_move = bool(out0["switch"][1] > out0["switch"][0])
        if do_move and cp.get_action_points() > 0:
            legal = get_legal_moves(env)
            # allow stay
            candidates: List[Point] = []
            if cp.position is not None:
                candidates.append(cp.position)
            candidates.extend(legal)
            # pick best by move prob
            best: Optional[Point] = None
            best_p = -1.0
            for pos in candidates:
                p = float(out0["move"][_idx_of(pos)])
                if p > best_p:
                    best_p = p
                    best = pos
            if best is not None:
                action.move = True
                action.move_target = best

        # apply move to a fork for next stage conditioning
        env_move = fork_environment(env)
        if getattr(action, "move", False):
            partial = ActionSet()
            partial.move = True
            partial.move_target = action.move_target
            partial.attack = False
            partial.spell = False
            env_move.execute_player_action(partial)
            if env_move.current_piece is None:
                env_move.begin_turn_host()

        # ---- stage 1: attack ----
        out1 = infer(env_move, 0.6)
        do_attack = bool(out1["switch"][1] > out1["switch"][0])
        if do_attack and env_move.current_piece is not None and env_move.current_piece.get_action_points() > 0:
            targets = get_attackable_targets(env_move)
            best_t = None
            best_p = -1.0
            for t in targets:
                p = float(out1["attack"][_idx_of(t.position)])
                if p > best_p:
                    best_p = p
                    best_t = t
            if best_t is not None:
                # rebind target to real env
                real_t = _find_piece_by_id(env, getattr(best_t, "id", -999999))
                if real_t is not None and getattr(real_t, "is_alive", True):
                    action.attack = True
                    ctx = AttackContext()
                    ctx.attacker = env.current_piece
                    ctx.target = real_t
                    if hasattr(ctx, "attackPosition") and env.current_piece is not None:
                        ctx.attackPosition = env.current_piece.position
                    action.attack_context = ctx

        # apply attack to a fork for spell conditioning
        env_attack = fork_environment(env_move)
        if getattr(action, "attack", False) and getattr(action, "attack_context", None) is not None:
            partial = ActionSet()
            partial.move = False
            partial.attack = True
            # build ctx for fork env
            fork_t = _find_piece_by_id(env_attack, getattr(action.attack_context.target, "id", -999999))
            if fork_t is not None:
                ctx = AttackContext()
                ctx.attacker = env_attack.current_piece
                ctx.target = fork_t
                if hasattr(ctx, "attackPosition") and env_attack.current_piece is not None:
                    ctx.attackPosition = env_attack.current_piece.position
                partial.attack_context = ctx
                partial.spell = False
                env_attack.execute_player_action(partial)
                if env_attack.current_piece is None:
                    env_attack.begin_turn_host()

        # ---- stage 2: spell ----
        out2 = infer(env_attack, 1.0)
        do_spell = bool(out2["switch"][1] > out2["switch"][0])
        if do_spell and env_attack.current_piece is not None and env_attack.current_piece.get_action_points() > 0 and env_attack.current_piece.spell_slots > 0:
            piece = env_attack.current_piece
            spells = env_attack.get_available_spells(piece)
            best = None
            best_p = -1.0
            best_target = None
            for sp in spells:
                s_idx = max(0, min(int(sp.id) - 1, out2["spell"].shape[0] - 1))
                if getattr(sp, "is_area_effect", False):
                    # use current pos as target center
                    pos = piece.position
                    p = float(out2["spell"][s_idx, _idx_of(pos)])
                    if p > best_p:
                        best_p = p
                        best = sp
                        best_target = None
                else:
                    for t in env_attack.get_spell_targets(sp, piece):
                        p = float(out2["spell"][s_idx, _idx_of(t.position)])
                        if p > best_p:
                            best_p = p
                            best = sp
                            best_target = t

            if best is not None:
                action.spell = True
                sc = SpellContext()
                sc.caster = env.current_piece
                sc.spell = best
                if best_target is not None:
                    real_t = _find_piece_by_id(env, getattr(best_target, "id", -999999))
                    sc.target = real_t
                    sc.target_area = Area(real_t.position.x, real_t.position.y, 0) if real_t is not None else None
                else:
                    sc.target = None
                    sc.target_area = Area(env.current_piece.position.x, env.current_piece.position.y, getattr(best, "area_radius", 0))
                action.spell_context = sc

        return action

    return strategy


def play_one_game(
    model_strategy,
    opp_strategy,
    processor: StateProcessor,
    init_fn_p1,
    init_fn_p2,
    max_steps: int,
) -> int:
    """Return winner: 1/2 or 0 draw."""

    env = Environment(local_mode=True, if_log=0)
    env.init_board_only()

    init1_args = init_fn_p1(_build_init_message(env, 1))
    init2_args = init_fn_p2(_build_init_message(env, 2))
    env.apply_init_policy(1, _wrap_piece_args(init1_args))
    env.apply_init_policy(2, _wrap_piece_args(init2_args))
    env.setup_battle_host()
    env.begin_turn_host()

    # reset persistent MCTS per new game (whichever side uses it)
    if hasattr(model_strategy, "_persistent_mcts"):
        model_strategy._persistent_mcts.reset()
    if hasattr(opp_strategy, "_persistent_mcts"):
        opp_strategy._persistent_mcts.reset()

    step = 0
    while not env.is_game_over and step < max_steps:
        if env.current_piece is None:
            env.begin_turn_host()
            if env.current_piece is None:
                break

        if env.current_piece.team == 1:
            action = model_strategy(env)
        else:
            action = opp_strategy(env)

        step_with_action(env, action)
        step += 1

    p1_alive = any(p.is_alive for p in env.player1.pieces)
    p2_alive = any(p.is_alive for p in env.player2.pieces)

    if p1_alive and not p2_alive:
        return 1
    if p2_alive and not p1_alive:
        return 2
    return 0


def evaluate(
    model_path: str,
    device: torch.device,
    games_per_side: int,
    simulations: int,
    mode: str,
    init_strategy: str,
    random_init: bool,
    seed: int,
    max_steps: int,
) -> Tuple[int, int, int]:
    model = TacticalPolicyNet(in_channels=19)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    processor = StateProcessor()

    # model uses either greedy policy or PUCT; opponent uses hand-crafted aggressive
    if mode == "puct":
        model_strategy = StrategyFactory.get_puct_action_strategy(
            model=model,
            processor=processor,
            device=device,
            simulations=simulations,
            sample=False,
        )
    elif mode == "policy":
        model_strategy = make_greedy_model_strategy(model, processor, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    opp_strategy = StrategyFactory.get_aggressive_action_strategy()

    # init
    if random_init:
        init_fn_p1 = random_aggressive_init_strategy(seed)
        init_fn_p2 = random_aggressive_init_strategy(seed + 1)
    else:
        init_fn_p1 = StrategyFactory.get_init_strategy_by_name(init_strategy)
        init_fn_p2 = StrategyFactory.get_init_strategy_by_name(init_strategy)

    wins = 0
    losses = 0
    draws = 0

    total_games = games_per_side * 2
    bar = tqdm(range(total_games), desc="Model vs Aggressive")

    for g in bar:
        # swap sides
        if g < games_per_side:
            # model is P1(team1), opp is P2(team2)
            winner = play_one_game(
                model_strategy=model_strategy,
                opp_strategy=opp_strategy,
                processor=processor,
                init_fn_p1=init_fn_p1,
                init_fn_p2=init_fn_p2,
                max_steps=max_steps,
            )
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:
            # model is P2(team2), opp is P1(team1)
            # We reuse the same strategies, but swap which side calls them by swapping teams via init.
            # Easiest: play game and interpret winner inverted.
            winner = play_one_game(
                model_strategy=opp_strategy,
                opp_strategy=model_strategy,
                processor=processor,
                init_fn_p1=init_fn_p1,
                init_fn_p2=init_fn_p2,
                max_steps=max_steps,
            )
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        done = wins + losses + draws
        bar.set_postfix({
            "W": wins,
            "L": losses,
            "D": draws,
            "W%": f"{(wins/done):.1%}" if done else "0.0%",
        })

    bar.close()
    return wins, losses, draws


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model vs aggressive")
    p.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--device", default="cuda", help="cuda/cpu")
    p.add_argument("--games-per-side", type=int, default=10)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--mode", choices=["policy", "puct"], default="policy", help="policy=fast greedy, puct=slower but stronger")

    p.add_argument("--init-strategy", default="archer29", help="Used when --random-init is off")
    p.add_argument("--random-init", action="store_true", default=True)
    p.add_argument("--no-random-init", action="store_true", help="Disable random init")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.no_random_init:
        args.random_init = False
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    wins, losses, draws = evaluate(
        model_path=args.model,
        device=device,
        games_per_side=args.games_per_side,
        simulations=args.simulations,
        mode=args.mode,
        init_strategy=args.init_strategy,
        random_init=args.random_init,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    total = wins + losses + draws
    print("\n" + "=" * 60)
    print("Model vs Aggressive")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Games: {total} (per side={args.games_per_side})")
    print(f"Mode: {args.mode}")
    if args.mode == "puct":
        print(f"Simulations: {args.simulations}")
    print(f"Init: {'random_aggressive_positions' if args.random_init else args.init_strategy}")
    print(f"Result: {wins}W / {losses}L / {draws}D | win_rate={wins/total:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

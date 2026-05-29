"""Bootstrap pretraining from Aggressive-vs-Aggressive games.

What it does
------------
1) Generate many games with two hand-crafted aggressive policies.
2) Randomize ONLY the initialization positions to increase state diversity.
3) Convert each decision into 3 staged training samples (move/attack/spell) using
   the existing data format (same as self_play.py).
4) Train TacticalPolicyNet on the generated dataset and save a checkpoint.

This is meant as a fast "warm start" before self-play training.

Example
-------
python bootstrap_aggressive_pretrain.py --games 200 --simulations 0 --max-steps 120 --epochs 20 --device cuda

Notes
-----
- Aggressive policy in strategy_factory.py never uses spells by default.
  The spell head will mostly see switch=0 samples in this bootstrap phase.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from env import Environment, PieceArg, Point
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory
from strategy_utils import step_with_action

from dataset_utils import create_dataloader, build_model
from self_play import load_examples_from_env, save_npz


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    """Pick random distinct walkable cells on the player's side."""
    bdr = board.boarder
    candidates: List[Tuple[int, int]] = []

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


def random_aggressive_init_strategy(seed: int | None = None):
    """Aggressive init stats, but randomize positions."""

    def strategy(init_message) -> List[PieceArg]:
        rng = random.Random(seed)
        # Add extra entropy per game by mixing in board hash-ish and pid.
        # (We don't rely on Python's hash randomization.)
        mix = (init_message.id * 1000003) ^ (init_message.piece_cnt * 9176)
        rng.seed((seed if seed is not None else 0) + mix + random.randint(0, 2**31 - 1))

        board = init_message.board
        pid = init_message.id
        positions = _random_aggressive_init_positions(board, pid, init_message.piece_cnt, rng)

        piece_args: List[PieceArg] = []
        for pos in positions:
            arg = PieceArg()
            # Same as StrategyFactory.get_aggressive_init_strategy()
            arg.strength = 20
            arg.dexterity = 8
            arg.intelligence = 2
            arg.equip = Point(2, 3)
            arg.pos = pos
            piece_args.append(arg)
        return piece_args

    return strategy


@dataclass
class GenerateConfig:
    games: int
    max_steps: int
    seed: int


def generate_dataset(config: GenerateConfig, processor: StateProcessor) -> List[dict]:
    """Generate (state, targets...) samples from aggressive vs aggressive games."""

    examples: List[dict] = []

    # Fixed policies (no MCTS) for speed and stability
    p1_strategy = StrategyFactory.get_aggressive_action_strategy()
    p2_strategy = StrategyFactory.get_aggressive_action_strategy()

    # Randomized init per game
    init1_fn = random_aggressive_init_strategy(seed=config.seed)
    init2_fn = random_aggressive_init_strategy(seed=config.seed + 1)

    game_bar = tqdm(range(config.games), desc="Bootstrap games")

    rng = random.Random(config.seed)

    for game_idx in game_bar:
        env = Environment(local_mode=True, if_log=0)
        env.init_board_only()

        # Per-game seed changes placement deterministically but diversely
        init1_args = init1_fn(_build_init_message(env, 1))
        init2_args = init2_fn(_build_init_message(env, 2))
        env.apply_init_policy(1, _wrap_piece_args(init1_args))
        env.apply_init_policy(2, _wrap_piece_args(init2_args))
        env.setup_battle_host()
        env.begin_turn_host()

        game_examples: List[dict] = []
        step = 0

        while not env.is_game_over and step < config.max_steps:
            if env.current_piece is None:
                env.begin_turn_host()
                if env.current_piece is None:
                    break

            current_team = env.current_piece.team
            action = p1_strategy(env) if current_team == 1 else p2_strategy(env)

            # staged training samples (move/attack/spell)
            game_examples.extend(load_examples_from_env(env, action, processor, current_team))
            step_with_action(env, action)
            step += 1

        # Determine winner
        winner = 0
        if any(p.is_alive for p in env.player1.pieces) and not any(p.is_alive for p in env.player2.pieces):
            winner = 1
        elif any(p.is_alive for p in env.player2.pieces) and not any(p.is_alive for p in env.player1.pieces):
            winner = 2
        else:
            winner = 0

        # Assign final value to all samples from the acting player's perspective
        for ex in game_examples:
            if winner == 0:
                ex["value"] = 0.0
            else:
                ex["value"] = 1.0 if ex.get("player_team") == winner else -1.0
            ex.pop("player_team", None)

        examples.extend(game_examples)

        # small live stats
        if (game_idx + 1) % 10 == 0:
            values = [e["value"] for e in examples[-min(len(examples), 3000):]]
            if values:
                game_bar.set_postfix({
                    "recent_v_mean": f"{float(np.mean(values)):.2f}",
                    "samples": len(examples),
                })

        # advance RNG (kept for future extensions)
        rng.random()

    return examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap pretrain from aggressive-vs-aggressive games")

    # generation
    p.add_argument("--save-dir", default="training_data_bootstrap", help="Output folder")
    p.add_argument("--games", type=int, default=200, help="Number of games to generate")
    p.add_argument("--max-steps", type=int, default=120, help="Max steps per game")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    _ensure_dir(run_dir)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")

    processor = StateProcessor()

    # 1) Generate dataset
    gen_cfg = GenerateConfig(games=args.games, max_steps=args.max_steps, seed=args.seed)
    examples = generate_dataset(gen_cfg, processor)

    data_path = os.path.join(run_dir, "bootstrap_data.npz")
    save_npz(data_path, examples)
    print(f"Saved dataset: {data_path} (samples={len(examples)})")

    # 2) Train model on dataset
    model = TacticalPolicyNet(in_channels=19)
    model.to(device)

    train_loader = create_dataloader(data_path, batch_size=args.batch_size, shuffle=True)

    ckpt_path = os.path.join(run_dir, "bootstrap_model.pt")
    build_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_path=ckpt_path,
    )

    print(f"Saved checkpoint: {ckpt_path}")
    print("Done.")


if __name__ == "__main__":
    main()

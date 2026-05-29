import os
import gc
from typing import List
from tqdm import tqdm

import numpy as np
import torch

from env import Environment, Point, ActionSet, SpellContext, AttackContext
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory
from strategy_utils import fork_environment, step_with_action


def save_npz(path: str, examples: List[dict]):
    """
    Save self-play examples to a .npz file.
    我们会按照 iteration 来保存 examples，每个 iteration 会保存一个 .npz 文件。
    
    Args:
        path: Path to save the .npz file
        examples: List of example dictionaries containing state, switch, move, attack,
                  spell, value, stage, and optionally move_probs, attack_probs, spell_probs.
    """
    states = np.stack([example["state"] for example in examples])
    switches = np.array([example["switch"] for example in examples], dtype=np.int64)
    moves = np.array([example["move"] for example in examples], dtype=np.int64)
    attacks = np.array([example["attack"] for example in examples], dtype=np.int64)
    spells = np.array([example["spell"] for example in examples], dtype=np.int64)
    values = np.array([example["value"] for example in examples], dtype=np.float32)
    stages = np.array([example["stage"] for example in examples], dtype=np.int64)

    save_dict = {
        "states": states,
        "switch": switches,
        "move": moves,
        "attack": attacks,
        "spell": spells,
        "value": values,
        "stage": stages,
    }

    # ★ 保存 MCTS 访问分布（用于软标签训练），尺寸从 state 推断
    has_probs = any("move_probs" in ex for ex in examples)
    if has_probs:
        # 从 states 形状推断棋盘大小: (N, C, H, W) → num_pos = H*W
        NP = states.shape[2] * states.shape[3]
        move_probs = np.stack([example.get("move_probs", np.zeros(NP, dtype=np.float32)) for example in examples])
        attack_probs = np.stack([example.get("attack_probs", np.zeros(NP, dtype=np.float32)) for example in examples])
        spell_probs = np.stack([example.get("spell_probs", np.zeros(4 * NP, dtype=np.float32)) for example in examples])
        save_dict["move_probs"] = move_probs
        save_dict["attack_probs"] = attack_probs
        save_dict["spell_probs"] = spell_probs

    np.savez_compressed(path, **save_dict)


def load_npz(path: str) -> List[dict]:
    """从 .npz 文件读取自对弈样本，还原为 examples 列表。

    Args:
        path: .npz 文件路径（由 save_npz 生成）

    Returns:
        List[dict]: 还原后的样本字典列表，每个字典包含 state, switch, move,
                    attack, spell, value, stage 键。
    """
    data = np.load(path, allow_pickle=True)
    states = data["states"]
    switches = data["switch"]
    moves = data["move"]
    attacks = data["attack"]
    spells = data["spell"]
    values = data["value"]
    stages = data["stage"]

    examples: List[dict] = []
    has_probs = "move_probs" in data
    for i in range(len(states)):
        ex = {
            "state": states[i],
            "switch": int(switches[i]),
            "move": int(moves[i]),
            "attack": int(attacks[i]),
            "spell": int(spells[i]),
            "value": float(values[i]),
            "stage": int(stages[i]),
        }
        if has_probs:
            ex["move_probs"] = data["move_probs"][i]
            ex["attack_probs"] = data["attack_probs"][i]
            ex["spell_probs"] = data["spell_probs"][i]
        examples.append(ex)
    return examples


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


def load_examples_from_env(
    env: Environment,
    action: ActionSet,
    processor: StateProcessor,
    player_team: int,
    visit_dists: dict = None,
) -> List[dict]:
    """从环境状态和已执行的动作构造训练样本。

    为动作的三个阶段（移动、攻击、施法）分别产生一条样本。
    每条样本包含该阶段的 state、switch 标签、动作索引标签、
    以及可选的 MCTS 访问分布（作为软策略标签）。

    Args:
        env: 执行动作前的环境。
        action: 已决定的完整 ActionSet。
        processor: StateProcessor 实例。
        player_team: 当前行动棋子的队伍编号（1 或 2）。
        visit_dists: 可选，MCTS 返回的访问分布字典，
            包含 'move_probs'(400,), 'attack_probs'(400,), 'spell_probs'(1600,)。

    Returns:
        List[dict]: 3 条样本（移动阶段、攻击阶段、施法阶段）。
    """
    if visit_dists is None:
        visit_dists = {}

    examples: List[dict] = []

    def build_partial_action(base_action: ActionSet, move=False, attack=False, spell=False) -> ActionSet:
        partial = ActionSet()
        partial.move = False
        partial.attack = False
        partial.spell = False
        if move and getattr(base_action, "move", False):
            partial.move = True
            partial.move_target = base_action.move_target
        if attack and getattr(base_action, "attack", False):
            partial.attack = True
            partial.attack_context = base_action.attack_context
        if spell and getattr(base_action, "spell", False):
            partial.spell = True
            partial.spell_context = base_action.spell_context
        return partial

    move_stage = processor.build_input(env, stage_value=0.3)
    if hasattr(action, "move") and action.move:
        move_index = processor._to_index(action.move_target.x, action.move_target.y)
    else:
        move_index = -1

    move_ex = {
        "state": move_stage,
        "switch": 1 if getattr(action, "move", False) else 0,
        "move": move_index,
        "attack": 0,
        "spell": 0,
        "value": 0.0,
        "stage": 0,
        "player_team": player_team,
    }
    if "move_probs" in visit_dists and visit_dists["move_probs"] is not None:
        move_ex["move_probs"] = visit_dists["move_probs"]
    examples.append(move_ex)

    env_move = fork_environment(env)
    env_move.execute_player_action(build_partial_action(action, move=True))
    if env_move.current_piece is None:
        env_move.begin_turn_host()

    attack_stage = processor.build_input(env_move, stage_value=0.6)
    if hasattr(action, "attack") and action.attack and action.attack_context is not None and action.attack_context.target is not None:
        attack_index = processor._to_index(
            action.attack_context.target.position.x,
            action.attack_context.target.position.y,
        )
    else:
        attack_index = -1

    attack_ex = {
        "state": attack_stage,
        "switch": 1 if getattr(action, "attack", False) else 0,
        "move": move_index,
        "attack": attack_index,
        "spell": 0,
        "value": 0.0,
        "stage": 1,
        "player_team": player_team,
    }
    if "attack_probs" in visit_dists and visit_dists["attack_probs"] is not None:
        attack_ex["attack_probs"] = visit_dists["attack_probs"]
    examples.append(attack_ex)

    env_attack = fork_environment(env_move)
    env_attack.execute_player_action(build_partial_action(action, attack=True))
    if env_attack.current_piece is None:
        env_attack.begin_turn_host()

    spell_stage = processor.build_input(env_attack, stage_value=1.0)
    spell_index = -1
    if getattr(action, "spell", False) and hasattr(action, "spell_context") and action.spell_context is not None:
        spell = action.spell_context.spell
        point = None
        if action.spell_context.target is not None:
            point = action.spell_context.target.position
        elif action.spell_context.target_area is not None:
            point = Point(action.spell_context.target_area.x, action.spell_context.target_area.y)
        if spell is not None and point is not None:
            NP = processor.width * processor.height
            spell_index = max(0, min(spell.id - 1, 3)) * NP + processor._to_index(point.x, point.y)

    spell_ex = {
        "state": spell_stage,
        "switch": 1 if getattr(action, "spell", False) else 0,
        "move": move_index,
        "attack": attack_index,
        "spell": spell_index,
        "value": 0.0,
        "stage": 2,
        "player_team": player_team,
    }
    if "spell_probs" in visit_dists and visit_dists["spell_probs"] is not None:
        spell_ex["spell_probs"] = visit_dists["spell_probs"]
    examples.append(spell_ex)

    return examples


def collect_self_play_examples(
    model: TacticalPolicyNet,
    processor: StateProcessor,
    player1_init: str,
    player2_init: str,
    player1_policy: str,
    player2_policy: str,
    games: int,
    device: torch.device,
    simulations: int,
    max_steps: int = 100,
    collect_for_team: int = None,
    temperature: float = 0.0,
) -> List[dict]:
    """收集自对弈数据。

    Args:
        model: 策略网络模型。
        processor: 模型输入处理器。
        player1_init: 玩家1初始配置名（逗号分隔可选多选一随机）。
        player2_init: 玩家2初始配置名。
        player1_policy: 玩家1行动策略名（puct/aggressive/defensive/random）。
        player2_policy: 玩家2行动策略名。
        games: 对弈游戏数量。
        device: 计算设备。
        simulations: MCTS 模拟次数。
        max_steps: 每局最大步数。
        collect_for_team: 若指定（1 或 2），只收集该队伍的数据；
            若为 None，收集双方数据。
        temperature: puct 策略的温度参数（0=确定性, >0=探索性）。

    Returns:
        List[dict]: 自对弈数据列表。每条样本包含 state/switch/move/attack/spell/
                    value/stage，若策略提供了 MCTS 访问分布则额外包含
                    move_probs/attack_probs/spell_probs。
    """
    examples: List[dict] = []
    model.eval()

    import random

    def _parse_candidates(s: str):
        if s is None:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts

    p1_init_candidates = _parse_candidates(player1_init)
    p2_init_candidates = _parse_candidates(player2_init)
    p1_policy_candidates = _parse_candidates(player1_policy)
    p2_policy_candidates = _parse_candidates(player2_policy)

    # 胜率 & 诊断统计
    p1_wins = 0
    p2_wins = 0
    draws = 0
    p1_physical_dmg = 0.0   # P1 普攻总伤害
    p2_physical_dmg = 0.0   # P2 普攻总伤害
    p1_spell_dmg = 0.0      # P1 法术总伤害
    p2_spell_dmg = 0.0      # P2 法术总伤害
    p1_first_kill = 0       # P1 先减员次数
    p2_first_kill = 0       # P2 先减员次数
    total_steps = 0

    def _choose_init(name: str):
        if name == "random":
            return StrategyFactory.get_random_init_strategy()
        return StrategyFactory.get_init_strategy_by_name(name)

    def _choose_policy(name: str):
        if name == "puct":
            return StrategyFactory.get_puct_action_strategy(
                model=model, processor=processor, device=device,
                simulations=simulations, temperature=temperature,
            )
        return StrategyFactory.get_action_strategy_by_name(
            name, model=model, processor=processor, device=device, simulations=simulations
        )

    game_bar = tqdm(range(games), desc="Self-play games", leave=False)

    for game_idx in game_bar:
        env = Environment(local_mode=True, if_log=0)
        env.init_board_only()

        # select init strategies
        if len(p1_init_candidates) > 1:
            sel = random.choice(p1_init_candidates)
        elif len(p1_init_candidates) == 1:
            sel = p1_init_candidates[0]
        else:
            sel = player1_init
        if len(p2_init_candidates) > 1:
            sel2 = random.choice(p2_init_candidates)
        elif len(p2_init_candidates) == 1:
            sel2 = p2_init_candidates[0]
        else:
            sel2 = player2_init

        p1_init_fn = _choose_init(sel)
        p2_init_fn = _choose_init(sel2)

        init1_args = p1_init_fn(_build_init_message(env, 1))
        init2_args = p2_init_fn(_build_init_message(env, 2))
        env.apply_init_policy(1, _wrap_piece_args(init1_args))
        env.apply_init_policy(2, _wrap_piece_args(init2_args))
        env.setup_battle_host()
        env.begin_turn_host()

        game_examples: List[dict] = []
        step = 0

        # ★ 每场比赛只创建一次策略
        if len(p1_policy_candidates) > 1:
            sel_pol = random.choice(p1_policy_candidates)
        elif len(p1_policy_candidates) == 1:
            sel_pol = p1_policy_candidates[0]
        else:
            sel_pol = player1_policy
        p1_strategy = _choose_policy(sel_pol)

        if len(p2_policy_candidates) > 1:
            sel_pol2 = random.choice(p2_policy_candidates)
        elif len(p2_policy_candidates) == 1:
            sel_pol2 = p2_policy_candidates[0]
        else:
            sel_pol2 = player2_policy
        p2_strategy = _choose_policy(sel_pol2)

        # 重置 PersistentMCTS
        if hasattr(p1_strategy, '_persistent_mcts'):
            p1_strategy._persistent_mcts.reset()
        if hasattr(p2_strategy, '_persistent_mcts'):
            p2_strategy._persistent_mcts.reset()

        # ★ 追踪每局首个减员
        first_blood_recorded = False
        initial_p1_alive = sum(1 for p in env.player1.pieces if p.is_alive)
        initial_p2_alive = sum(1 for p in env.player2.pieces if p.is_alive)

        while not env.is_game_over and step < max_steps:
            if env.current_piece is None:
                env.begin_turn_host()
                if env.current_piece is None:
                    break
            current_team = env.current_piece.team if env.current_piece is not None else 1
            if current_team == 1:
                strategy = p1_strategy
            else:
                strategy = p2_strategy

            action = strategy(env)

            # ★ 提取 MCTS 访问分布（仅 puct 策略有）
            visit_dists = None
            if hasattr(strategy, '_last_visit_dists'):
                visit_dists = strategy._last_visit_dists
            elif hasattr(strategy, '_persistent_mcts'):
                visit_dists = strategy._persistent_mcts.get_visit_distributions()

            # ★ 按队伍过滤：只收集指定队伍的数据
            if collect_for_team is None or current_team == collect_for_team:
                game_examples.extend(
                    load_examples_from_env(env, action, processor, current_team, visit_dists)
                )

            # ★ 诊断：追踪伤害
            if hasattr(action, 'attack') and action.attack and action.attack_context:
                dmg = 0
                attacker = action.attack_context.attacker
                target = action.attack_context.target
                if attacker and target:
                    if getattr(attacker, 'weapon_type', 0) == 4:
                        dmg = 4
                    else:
                        dmg = max(0, attacker.physical_damage + attacker.strength - target.physical_resist)
                if current_team == 1:
                    p1_physical_dmg += dmg
                else:
                    p2_physical_dmg += dmg

            if hasattr(action, 'spell') and action.spell and action.spell_context:
                dmg = action.spell_context.damage_value if hasattr(action.spell_context, 'damage_value') else 0
                if current_team == 1:
                    p1_spell_dmg += dmg
                else:
                    p2_spell_dmg += dmg

            # ★ 追踪首杀
            if not first_blood_recorded:
                cur_p1_alive = sum(1 for p in env.player1.pieces if p.is_alive)
                cur_p2_alive = sum(1 for p in env.player2.pieces if p.is_alive)
                if cur_p1_alive < initial_p1_alive:
                    p2_first_kill += 1
                    first_blood_recorded = True
                elif cur_p2_alive < initial_p2_alive:
                    p1_first_kill += 1
                    first_blood_recorded = True

            step_with_action(env, action)
            step += 1

        total_steps += step
        winner = 0
        if any(p.is_alive for p in env.player1.pieces) and not any(p.is_alive for p in env.player2.pieces):
            winner = 1
            p1_wins += 1
        elif any(p.is_alive for p in env.player2.pieces) and not any(p.is_alive for p in env.player1.pieces):
            winner = 2
            p2_wins += 1
        else:
            draws += 1

        for example in game_examples:
            if winner == 0:
                example["value"] = 0.0
            else:
                example["value"] = 1.0 if example["player_team"] == winner else -1.0
            example.pop("player_team", None)

        examples.extend(game_examples)

        # 更新进度条
        total_games = p1_wins + p2_wins + draws
        p1_win_rate = p1_wins / total_games if total_games > 0 else 0
        p2_win_rate = p2_wins / total_games if total_games > 0 else 0
        game_bar.set_postfix({
            'p1_wins': p1_wins, 'p2_wins': p2_wins, 'draws': draws,
            'p1_rate': f"{p1_win_rate:.1%}", 'p2_rate': f"{p2_win_rate:.1%}"
        })

        # 内存回收
        if hasattr(p1_strategy, '_persistent_mcts'):
            p1_strategy._persistent_mcts.reset()
        if hasattr(p2_strategy, '_persistent_mcts'):
            p2_strategy._persistent_mcts.reset()
        del env
        gc.collect()

    game_bar.close()

    total_games = p1_wins + p2_wins + draws
    p1_win_rate = p1_wins / total_games if total_games > 0 else 0
    p2_win_rate = p2_wins / total_games if total_games > 0 else 0
    print(f"Self-play results: P1 wins={p1_wins} ({p1_win_rate:.1%}), "
          f"P2 wins={p2_wins} ({p2_win_rate:.1%}), Draws={draws} ({draws/total_games:.1%})")
    # ★ 诊断输出
    print(f"  Diagnostics: avg_steps={total_steps/total_games:.1f}, "
          f"P1 first_kill={p1_first_kill}, P2 first_kill={p2_first_kill}")
    print(f"  Damage: P1 phys={p1_physical_dmg:.0f} spell={p1_spell_dmg:.0f}, "
          f"P2 phys={p2_physical_dmg:.0f} spell={p2_spell_dmg:.0f}")
    if collect_for_team is not None:
        print(f"  Data collection: team {collect_for_team} only, "
              f"{len(examples)} examples collected")
    has_probs = any("move_probs" in ex for ex in examples)
    print(f"  Visit distributions: {'yes' if has_probs else 'no'}")

    return examples


def collect_heuristic_examples(
    processor: StateProcessor,
    init_strategy: str = "archer29",
    action_strategy: str = "aggressive",
    games: int = 10,
    max_steps: int = 100,
) -> List[dict]:
    """使用纯启发式策略自我对弈收集训练数据（预训练用）。

    P1 和 P2 使用相同的初始化和行动策略。不支持模型策略。

    Args:
        processor: 状态处理器。
        init_strategy: 双方初始化策略名。
        action_strategy: 双方行动策略名。
        games: 对弈局数。
        max_steps: 每局最大步数。

    Returns:
        List[dict]: 训练样本列表。
    """
    examples: List[dict] = []
    p1_wins = 0
    p2_wins = 0
    draws = 0

    init_fn = StrategyFactory.get_init_strategy_by_name(init_strategy)
    action_fn = StrategyFactory.get_action_strategy_by_name(action_strategy)

    game_bar = tqdm(range(games), desc="Heuristic self-play", leave=False)

    for _ in game_bar:
        env = Environment(local_mode=True, if_log=0)
        env.init_board_only()

        init1_args = init_fn(_build_init_message(env, 1))
        init2_args = init_fn(_build_init_message(env, 2))
        env.apply_init_policy(1, _wrap_piece_args(init1_args))
        env.apply_init_policy(2, _wrap_piece_args(init2_args))
        env.setup_battle_host()
        env.begin_turn_host()

        game_examples: List[dict] = []
        step = 0

        while not env.is_game_over and step < max_steps:
            if env.current_piece is None:
                env.begin_turn_host()
                if env.current_piece is None:
                    break
            current_team = env.current_piece.team

            action = action_fn(env)
            # ★ 启发式策略无 MCTS 访问分布，传 None
            game_examples.extend(
                load_examples_from_env(env, action, processor, current_team, visit_dists=None)
            )
            step_with_action(env, action)
            step += 1

        winner = 0
        if any(p.is_alive for p in env.player1.pieces) and not any(p.is_alive for p in env.player2.pieces):
            winner = 1
            p1_wins += 1
        elif any(p.is_alive for p in env.player2.pieces) and not any(p.is_alive for p in env.player1.pieces):
            winner = 2
            p2_wins += 1
        else:
            draws += 1

        for ex in game_examples:
            if winner == 0:
                ex["value"] = 0.0
            else:
                ex["value"] = 1.0 if ex["player_team"] == winner else -1.0
            ex.pop("player_team", None)

        examples.extend(game_examples)
        del env
        gc.collect()

    game_bar.close()
    total = p1_wins + p2_wins + draws
    print(f"Heuristic self-play: P1={p1_wins}W P2={p2_wins}W Draws={draws}, "
          f"total examples={len(examples)}")
    return examples

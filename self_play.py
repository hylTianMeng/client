import os
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
    我们会按照 iteration 来保存 examples，每个 iteration 会保存一个 .npz 文件
    这里把字典的每一个要素都拿出来,做成单独的列表,放到npz里面/
    
    Args:
        path: Path to save the .npz file
        examples: List of example dictionaries containing state, switch, move, attack, spell, value, and stage
    
    """
    states = np.stack([example["state"] for example in examples])
    switches = np.array([example["switch"] for example in examples], dtype=np.int64)
    moves = np.array([example["move"] for example in examples], dtype=np.int64)
    attacks = np.array([example["attack"] for example in examples], dtype=np.int64)
    spells = np.array([example["spell"] for example in examples], dtype=np.int64)
    values = np.array([example["value"] for example in examples], dtype=np.float32)
    stages = np.array([example["stage"] for example in examples], dtype=np.int64)
    np.savez_compressed(
        path,
        states=states,
        switch=switches,
        move=moves,
        attack=attacks,
        spell=spells,
        value=values,
        stage=stages,
    )


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
) -> List[dict]:

    """
    我们返回的是一个字典的列表，表示训练数据，以下是返回的字典的形式，还要再进行进一步的处理，
            "state": spell_stage,
            "switch": 1 if getattr(action, "spell", False) else 0,
            "move": move_index,
            "attack": attack_index,
            "spell": spell_index,
            "value": 0.0,
            "stage": 2,
            "player_team": player_team,
    其中，play_team 最后会去掉，而 move attack spell 都是一个整型，进行了一定的编码，其中 spell 把原来 1 2 3 5 的法术 id 改为了 0 1 2 3，也分别对应 4 个通道。
    """
    examples: List[dict] = []

    def build_partial_action(base_action: ActionSet, move=False, attack=False, spell=False) -> ActionSet:
        """
        这个函数用于返回动作的部分，通过 move attack spell 的布尔值来控制返回哪一部分
        """
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
        move_index = StateProcessor._to_index(action.move_target.x, action.move_target.y)
    else:
        # 使用-1表示不移动，避免与位置索引(0,0)冲突
        move_index = -1

    examples.append(
        {
            "state": move_stage,
            "switch": 1 if getattr(action, "move", False) else 0,
            "move": move_index,
            "attack": 0,
            "spell": 0,
            "value": 0.0,
            "stage": 0,
            "player_team": player_team,
        }
    )

    env_move = fork_environment(env)
    env_move.execute_player_action(build_partial_action(action, move=True))
    if env_move.current_piece is None:
        env_move.begin_turn_host()

    attack_stage = processor.build_input(env_move, stage_value=0.6)
    if hasattr(action, "attack") and action.attack and action.attack_context is not None and action.attack_context.target is not None:
        attack_index = StateProcessor._to_index(
            action.attack_context.target.position.x,
            action.attack_context.target.position.y,
        )
    else:
        # 使用-1表示不攻击，避免与位置索引(0,0)冲突
        attack_index = -1

    examples.append(
        {
            "state": attack_stage,
            "switch": 1 if getattr(action, "attack", False) else 0,
            "move": move_index,
            "attack": attack_index,
            "spell": 0,
            "value": 0.0,
            "stage": 1,
            "player_team": player_team,
        }
    )

    env_attack = fork_environment(env_move)
    env_attack.execute_player_action(build_partial_action(action, attack=True))
    if env_attack.current_piece is None:
        env_attack.begin_turn_host()

    spell_stage = processor.build_input(env_attack, stage_value=1.0)
    spell_index = -1  # 使用-1表示不施法，避免与位置索引冲突
    if getattr(action, "spell", False) and hasattr(action, "spell_context") and action.spell_context is not None:
        spell = action.spell_context.spell
        point = None
        if action.spell_context.target is not None:
            point = action.spell_context.target.position
        elif action.spell_context.target_area is not None:
            point = Point(action.spell_context.target_area.x, action.spell_context.target_area.y)
        if spell is not None and point is not None:
            spell_index = max(0, min(spell.id - 1, 3)) * 400 + StateProcessor._to_index(point.x, point.y)

    examples.append(
        {
            "state": spell_stage,
            "switch": 1 if getattr(action, "spell", False) else 0,
            "move": move_index,
            "attack": attack_index,
            "spell": spell_index,
            "value": 0.0,
            "stage": 2,
            "player_team": player_team,
        }
    )

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
) -> List[dict]:
    """
    收集自对弈数据
    
    Args:
        model: 策略网络模型
        processor: 模型输入处理器
        player1_init: 玩家1初始配置
        player2_init: 玩家2初始配置
        player1_policy: 玩家1策略
        player2_policy: 玩家2策略
        games: 对弈游戏数量
        device: 设备
        simulations: 模拟次数
        max_steps: 最大步数
    
    Returns:
        List[dict]: 自对弈数据列表
    """
    examples: List[dict] = []
    model.eval()

    import random

    def _parse_candidates(s: str):
        """
        解析候选字符串，返回候选列表
        """
        if s is None:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts

    p1_init_candidates = _parse_candidates(player1_init)
    # print(p1_init_candidates, player1_init);
    p2_init_candidates = _parse_candidates(player2_init)
    p1_policy_candidates = _parse_candidates(player1_policy)
    p2_policy_candidates = _parse_candidates(player2_policy)

    # 胜率统计
    p1_wins = 0
    p2_wins = 0
    draws = 0

    def _choose_init(name: str):
        if name == "random":
            return StrategyFactory.get_random_init_strategy()
        return StrategyFactory.get_init_strategy_by_name(name)

    def _choose_policy(name: str):
        if name == "puct":
            return StrategyFactory.get_puct_action_strategy(
                model=model, processor=processor, device=device, simulations=simulations
            )
        return StrategyFactory.get_action_strategy_by_name(
            name, model=model, processor=processor, device=device, simulations=simulations
        )

    game_bar = tqdm(range(games), desc="Self-play games", leave=False)
    
    for game_idx in game_bar:
        env = Environment(local_mode=True, if_log=0)
        env.init_board_only()

        # select init strategies for this game (support randomized selection)
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
        env.begin_turn_host()  # 初始化current_piece

        game_examples: List[dict] = []
        step = 0
        
        # ★ 修复：每场比赛只创建一次策略（移到循环外），避免每步重建 MCTS
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
        
        while not env.is_game_over and step < max_steps:
            # ★ 防御 current_piece 为 None
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
            game_examples.extend(load_examples_from_env(env, action, processor, current_team))
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

        for example in game_examples:
            if winner == 0:
                example["value"] = 0.0
            else:
                example["value"] = 1.0 if example["player_team"] == winner else -1.0
            example.pop("player_team", None)

        # ★ 不过滤样本：value head 需要正负例才能学会区分好坏局面。
        # 策略头由 compute_loss 中的样本权重（按 value 加权）来处理。
        examples.extend(game_examples)

        # 更新进度条显示胜率
        total_games = p1_wins + p2_wins + draws
        p1_win_rate = p1_wins / total_games if total_games > 0 else 0
        p2_win_rate = p2_wins / total_games if total_games > 0 else 0
        game_bar.set_postfix({
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'draws': draws,
            'p1_rate': f"{p1_win_rate:.1%}",
            'p2_rate': f"{p2_win_rate:.1%}"
        })

    # 关闭进度条
    game_bar.close()
    
    # 打印最终胜率统计
    print(f"Self-play results: P1 wins={p1_wins} ({p1_win_rate:.1%}), P2 wins={p2_wins} ({p2_win_rate:.1%}), Draws={draws} ({draws/total_games:.1%})")

    return examples

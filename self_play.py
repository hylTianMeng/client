import os
from typing import List

import numpy as np
import torch

from env import Environment, Point, ActionSet, SpellContext, AttackContext
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory
from strategy_utils import fork_environment, step_with_action


def save_npz(path: str, examples: List[dict]):
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
        move_target=moves,
        attack_target=attacks,
        spell_target=spells,
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
        move_index = StateProcessor._to_index(action.move_target.x, action.move_target.y)
    elif env.current_piece is not None:
        move_index = StateProcessor._to_index(env.current_piece.position.x, env.current_piece.position.y)
    else:
        move_index = 0

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

    attack_stage = processor.build_input(env_move, stage_value=0.6)
    if hasattr(action, "attack") and action.attack and action.attack_context is not None and action.attack_context.target is not None:
        attack_index = StateProcessor._to_index(
            action.attack_context.target.position.x,
            action.attack_context.target.position.y,
        )
    else:
        attack_index = 0

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

    spell_stage = processor.build_input(env_attack, stage_value=1.0)
    spell_index = 0
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
    player2_policy: str,
    games: int,
    device: torch.device,
    simulations: int,
    max_steps: int = 100,
    puct_version: str = "v3",
) -> List[dict]:
    examples: List[dict] = []
    model.eval()

    p1_init_fn = StrategyFactory.get_init_strategy_by_name(player1_init)
    p2_init_fn = StrategyFactory.get_init_strategy_by_name(player2_init)
    p2_strategy_fn = StrategyFactory.get_action_strategy_by_name(
        player2_policy,
        model=model,
        processor=processor,
        device=device,
        simulations=simulations,
    )

    # Create and init all game environments
    game_envs: List[Environment] = []
    game_examples_lists: List[List[dict]] = []

    for _ in range(games):
        env = Environment(local_mode=True, if_log=0)
        env.init_board_only()
        init1_args = p1_init_fn(_build_init_message(env, 1))
        init2_args = p2_init_fn(_build_init_message(env, 2))
        env.apply_init_policy(1, _wrap_piece_args(init1_args))
        env.apply_init_policy(2, _wrap_piece_args(init2_args))
        env.setup_battle_host()
        env.begin_turn_host()
        game_envs.append(env)
        game_examples_lists.append([])

    # Shared MCTS batched strategy for player 1
    batched_mcts = None
    if puct_version == "v3":
        batched_mcts = StrategyFactory.get_puct_v3_batched_strategy(
            model=model, processor=processor, device=str(device),
            simulations=simulations,
        )

    active_indices = list(range(games))

    while active_indices:
        # Phase 1: collect player-1-turn envs for batched MCTS
        p1_indices = []
        p1_envs = []
        p2_queue = []  # (index, env) where player 2 acts

        for i in active_indices:
            env = game_envs[i]
            if env.is_game_over:
                continue
            cur = env.current_piece
            if cur is None:
                continue
            if cur.team == 1:
                p1_indices.append(i)
                p1_envs.append(env)
            else:
                p2_queue.append((i, env))

        # Batch player 1 actions
        if p1_envs:
            if batched_mcts is not None:
                p1_actions = batched_mcts(p1_envs)
            else:
                p1_actions = []
                for env in p1_envs:
                    strategy = StrategyFactory.get_puct_action_strategy(
                        model=model, processor=processor, device=str(device),
                        simulations=simulations,
                    )
                    p1_actions.append(strategy(env))

            for idx, env, action in zip(p1_indices, p1_envs, p1_actions):
                game_examples_lists[idx].extend(
                    load_examples_from_env(env, action, processor, 1))
                step_with_action(env, action)

        # Player 2 actions (individual)
        for idx, env in p2_queue:
            action = p2_strategy_fn(env)
            game_examples_lists[idx].extend(
                load_examples_from_env(env, action, processor, 2))
            step_with_action(env, action)

        # Remove finished or stuck games
        active_indices = [i for i in active_indices
                          if not game_envs[i].is_game_over]

    # Assign game-outcome values
    for i, ge in enumerate(game_examples_lists):
        env = game_envs[i]
        winner = 0
        if any(p.is_alive for p in env.player1.pieces) and not any(p.is_alive for p in env.player2.pieces):
            winner = 1
        elif any(p.is_alive for p in env.player2.pieces) and not any(p.is_alive for p in env.player1.pieces):
            winner = 2
        for example in ge:
            if winner == 0:
                example["value"] = 0.0
            else:
                example["value"] = 1.0 if example["player_team"] == winner else -1.0
            example.pop("player_team", None)
        examples.extend(ge)

    return examples

import gc
import torch
from typing import Tuple
from tqdm import tqdm

from env import Environment
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory
from strategy_utils import step_with_action


def evaluate_model(
    new_model: TacticalPolicyNet,
    old_model: TacticalPolicyNet,
    processor: StateProcessor,
    device: torch.device,
    init_strategy: str = "archer29",
    games_per_side: int = 5,
    simulations: int = 300,
    verbose: bool = False,
) -> Tuple[float, int, int, int]:
    """
    评估新模型与旧模型的对战性能。
    
    Args:
        new_model: 新模型（当前训练后的模型）
        old_model: 旧模型（基线模型，如初始未训练的模型）
        processor: 状态处理器
        device: 设备
        init_strategy: 初始化策略
        games_per_side: 每个模型作为player1的对战次数
        simulations: MCTS模拟次数
        verbose: 是否输出每局详细结果
    
    Returns:
        (win_rate, wins, losses, draws): 新模型的胜率、胜局数、负局数、平局数
    """
    new_model.eval()
    old_model.eval()
    
    wins = 0
    losses = 0
    draws = 0
    total_games = games_per_side * 2  # 两边各作为player1对战
    
    eval_bar = tqdm(range(total_games), desc="Eval games", leave=True)
    
    for game_idx in eval_bar:
        try:
            # 决定哪个模型是player1
            if game_idx < games_per_side:
                p1_model, p2_model = new_model, old_model
                p1_name, p2_name = "New", "Old"
            else:
                p1_model, p2_model = old_model, new_model
                p1_name, p2_name = "Old", "New"
            
            # 创建环境
            env = Environment(local_mode=True, if_log=0)
            env.init_board_only()
            
            # 初始化棋子
            p1_init_fn = StrategyFactory.get_init_strategy_by_name(init_strategy)
            p2_init_fn = StrategyFactory.get_init_strategy_by_name(init_strategy)
            
            init1_args = p1_init_fn(_build_init_message(env, 1))
            init2_args = p2_init_fn(_build_init_message(env, 2))
            env.apply_init_policy(1, _wrap_piece_args(init1_args))
            env.apply_init_policy(2, _wrap_piece_args(init2_args))
            env.setup_battle_host()
            env.begin_turn_host()
            
            # 创建策略
            p1_strategy = StrategyFactory.get_puct_action_strategy(
                p1_model, processor, device, simulations
            )
            p2_strategy = StrategyFactory.get_puct_action_strategy(
                p2_model, processor, device, simulations
            )
            
            # 重置持久化 MCTS
            if hasattr(p1_strategy, '_persistent_mcts'):
                p1_strategy._persistent_mcts.reset()
            if hasattr(p2_strategy, '_persistent_mcts'):
                p2_strategy._persistent_mcts.reset()
            
            # 对战（带 step 进度条）
            step = 0
            max_steps = 500
            step_bar = tqdm(total=max_steps, desc=f"  Game {game_idx+1}", leave=False, position=1)
            while not env.is_game_over and step < max_steps:
                if env.current_piece is None:
                    env.begin_turn_host()
                    if env.current_piece is None:
                        break
                
                if env.current_piece.team == 1:
                    action = p1_strategy(env)
                else:
                    action = p2_strategy(env)
                step_with_action(env, action)
                step += 1
                step_bar.update(1)
                step_bar.set_postfix({'piece': f"id={env.current_piece.id}" if env.current_piece else "None"})
            step_bar.close()
            
            # 判断胜负
            p1_alive = any(p.is_alive for p in env.player1.pieces)
            p2_alive = any(p.is_alive for p in env.player2.pieces)
            
            if p1_alive and not p2_alive:
                winner = 1
                result_str = "P1胜(全灭)"
            elif p2_alive and not p1_alive:
                winner = 2
                result_str = "P2胜(全灭)"
            elif step >= max_steps:
                winner = 0
                result_str = "平局(步尽)"
            else:
                winner = 0
                result_str = "平局"
            
            # 统计（从新模型角度）
            if game_idx < games_per_side:
                # 新模型=P1, 旧模型=P2
                if winner == 1:
                    wins += 1
                elif winner == 2:
                    losses += 1
                else:
                    draws += 1
            else:
                # 旧模型=P1, 新模型=P2
                if winner == 2:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
            
            if verbose:
                print(f"  Game {game_idx+1}/{total_games}: {p1_name}(P1) vs {p2_name}(P2) → {result_str} "
                      f"[累计: {wins}W/{losses}L/{draws}D]")
        
        except Exception as e:
            print(f"  Game {game_idx+1}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            # 出错的局算平局
            draws += 1
        
        # 更新进度条
        total_done = wins + losses + draws
        eval_bar.set_postfix({
            'New_W': wins,
            'New_L': losses,
            'D': draws,
            'New%': f"{wins/total_done:.1%}" if total_done > 0 else "0.0%"
        })
        
        # 每局后回收内存
        gc.collect()
    
    eval_bar.close()
    
    win_rate = wins / total_games if total_games > 0 else 0.0
    return win_rate, wins, losses, draws


def _build_init_message(env: Environment, player_id: int):
    """构建初始化消息"""
    init_message = type("InitGameMessage", (), {})()
    init_message.piece_cnt = env.player1.PIECE_CNT if player_id == 1 else env.player2.PIECE_CNT
    init_message.id = player_id
    init_message.board = env.board
    return init_message


def _wrap_piece_args(piece_args):
    """包装棋子参数"""
    policy = type("InitPolicyMessage", (), {})()
    policy.piece_args = piece_args
    return policy

import torch
from typing import Tuple
from tqdm import tqdm

from env import Environment
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory


def evaluate_model(
    new_model: TacticalPolicyNet,
    old_model: TacticalPolicyNet,
    processor: StateProcessor,
    device: torch.device,
    init_strategy: str = "archer29",
    games_per_side: int = 5,
    simulations: int = 300,
) -> Tuple[float, int, int, int]:
    """
    评估新模型与旧模型的对战性能
    
    Args:
        new_model: 新模型
        old_model: 旧模型
        processor: 状态处理器
        device: 设备
        init_strategy: 初始化策略
        games_per_side: 每个模型作为player1的对战次数
        simulations: MCTS模拟次数
    
    Returns:
        (win_rate, wins, losses, draws): 新模型的胜率、胜局数、负局数、平局数
    """
    new_model.eval()
    old_model.eval()
    
    wins = 0
    losses = 0
    draws = 0
    total_games = games_per_side * 2  # 两边各作为player1对战
    
    # 添加进度条
    eval_bar = tqdm(range(total_games), desc="Evaluation games", leave=False)
    
    for game_idx in eval_bar:
        # 决定哪个模型是player1
        if game_idx < games_per_side:
            # 新模型作为player1
            player1_model = new_model
            player2_model = old_model
        else:
            # 旧模型作为player1
            player1_model = old_model
            player2_model = new_model
        
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
        player1_strategy = StrategyFactory.get_puct_action_strategy(
            player1_model, processor, device, simulations
        )
        player2_strategy = StrategyFactory.get_puct_action_strategy(
            player2_model, processor, device, simulations
        )
        
        # 对战
        step = 0
        max_steps = 500
        while not env.is_game_over and step < max_steps:
            # 检查 current_piece 是否为 None
            if env.current_piece is None:
                env.begin_turn_host()
                if env.current_piece is None:
                    print(f"Warning: current_piece is still None after begin_turn_host at step {step}")
                    break
            
            if env.current_piece.team == 1:
                action = player1_strategy(env)
            else:
                action = player2_strategy(env)
            step_with_action(env, action)
            step += 1
        
        # 判断胜负
        if any(p.is_alive for p in env.player1.pieces) and not any(p.is_alive for p in env.player2.pieces):
            winner = 1
        elif any(p.is_alive for p in env.player2.pieces) and not any(p.is_alive for p in env.player1.pieces):
            winner = 2
        else:
            winner = 0  # 平局
        
        # 统计结果（从新模型的角度）
        if game_idx < games_per_side:
            # 新模型作为player1
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:
            # 新模型作为player2
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
        
        # 更新进度条
        eval_bar.set_postfix({
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': f"{wins/(wins+losses+draws):.1%}" if (wins+losses+draws) > 0 else "0.0%"
        })
    
    # 关闭进度条
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


# 导入必要的函数
from strategy_utils import step_with_action

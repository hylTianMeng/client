"""
两个模型对战脚本
用于比较两个不同模型的对战能力
"""

import argparse
import os
import torch
from datetime import datetime
from model import TacticalPolicyNet
from state_processor import StateProcessor
from strategy_factory import StrategyFactory
from env import Environment
from strategy_utils import step_with_action
from tqdm import tqdm
from utils import ActionSet


def _describe_piece(piece) -> str:
    """将棋子状态转为可读字符串。"""
    if piece is None:
        return "无棋子"
    return (f"棋子{piece.id}(队{piece.team}) "
            f"HP={piece.health}/{piece.max_health} "
            f"AP={piece.action_points}/{piece.max_action_points} "
            f"SP={piece.spell_slots}/{piece.max_spell_slots} "
            f"@({piece.position.x},{piece.position.y})")


def _describe_action(action: ActionSet) -> str:
    """将 ActionSet 转为可读字符串，用于调试输出。"""
    parts = []
    if hasattr(action, 'move') and action.move:
        t = action.move_target
        parts.append(f"移动→({t.x},{t.y})")
    else:
        parts.append("不移动")
    if hasattr(action, 'attack') and action.attack:
        ctx = action.attack_context
        if ctx and ctx.target:
            parts.append(f"攻击→棋子{ctx.target.id}@{ctx.target.position}")
        else:
            parts.append("攻击(无目标)")
    else:
        parts.append("不攻击")
    if hasattr(action, 'spell') and action.spell:
        ctx = action.spell_context
        if ctx and ctx.spell:
            parts.append(f"法术→{ctx.spell.name}")
        else:
            parts.append("法术(无效)")
    else:
        parts.append("不施法")
    return " | ".join(parts)


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


def battle_models(
    model1_path: str,
    model2_path: str,
    device: torch.device,
    init_strategy: str = "archer29",
    games_per_side: int = 5,
    simulations: int = 100,
    verbose: bool = False,
    log_file: str = None,
):
    """
    让两个模型对战，每一步的动作日志输出到文件。

    Args:
        model1_path: 模型1的路径
        model2_path: 模型2的路径
        device: 设备
        init_strategy: 初始化策略
        games_per_side: 每个模型作为player1的对战次数
        simulations: MCTS模拟次数
        verbose: 是否在控制台输出每步详情
        log_file: 日志文件路径（None=自动生成）
    """
    # ★ 设置日志文件
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = "battle_log/" + f"battle_log_{ts}.txt"
    log_fh = open(log_file, "w", encoding="utf-8")
    log_fh.write(f"Model1: {model1_path}\n")
    log_fh.write(f"Model2: {model2_path}\n")
    log_fh.write(f"Init: {init_strategy} | Sims: {simulations} | Games per side: {games_per_side}\n")
    log_fh.write(f"{'='*60}\n\n")
    log_fh.flush()
    print(f"Action log → {log_file}")
    # 加载模型
    # 创建处理器（40×40, 21通道）
    processor = StateProcessor()
    
    print(f"Loading model1 from {model1_path}")
    model1 = TacticalPolicyNet(in_channels=processor.num_channels)
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.to(device)
    model1.eval()
    
    print(f"Loading model2 from {model2_path}")
    model2 = TacticalPolicyNet(in_channels=processor.num_channels)
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.to(device)
    model2.eval()
    
    # 统计结果
    model1_wins = 0
    model2_wins = 0
    draws = 0
    total_games = games_per_side * 2

    # ★ 动作分布统计（诊断用）
    action_stats = {
        "model1": {"move": 0, "attack": 0, "spell": 0, "skip_all": 0, "total": 0},
        "model2": {"move": 0, "attack": 0, "spell": 0, "skip_all": 0, "total": 0},
    }
    
    # 进度条
    battle_bar = tqdm(range(total_games), desc="Battle games", leave=True)
    
    for game_idx in battle_bar:
        # 决定哪个模型是player1
        if game_idx < games_per_side:
            player1_model = model1
            player2_model = model2
            player1_name = "Model1"
            player2_name = "Model2"
        else:
            player1_model = model2
            player2_model = model1
            player1_name = "Model2"
            player2_name = "Model1"
        
        if verbose:
            print(f"\n--- Game {game_idx+1}/{total_games}: {player1_name}(P1) vs {player2_name}(P2) ---")
        
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

        # ★ 写入开局信息到日志
        log_fh.write(f"--- Game {game_idx+1}/{total_games}: {player1_name}(P1) vs {player2_name}(P2) ---\n")
        log_fh.write(f"  P1 pieces:\n")
        for p in env.action_queue:
            if p.team == 1:
                log_fh.write(f"    id={p.id} HP={p.health}/{p.max_health} STR={p.strength} DEX={p.dexterity} INT={p.intelligence} @({p.position.x},{p.position.y})\n")
        log_fh.write(f"  P2 pieces:\n")
        for p in env.action_queue:
            if p.team == 2:
                log_fh.write(f"    id={p.id} HP={p.health}/{p.max_health} STR={p.strength} DEX={p.dexterity} INT={p.intelligence} @({p.position.x},{p.position.y})\n")
        log_fh.flush()
        
        # 创建策略
        player1_strategy = StrategyFactory.get_puct_action_strategy(
            player1_model, processor, device, simulations
        )
        player2_strategy = StrategyFactory.get_puct_action_strategy(
            player2_model, processor, device, simulations
        )
        
        # ★ 重置持久化 MCTS 树（新游戏新树）
        if hasattr(player1_strategy, '_persistent_mcts'):
            player1_strategy._persistent_mcts.reset()
        if hasattr(player2_strategy, '_persistent_mcts'):
            player2_strategy._persistent_mcts.reset()
        
        game_result = "unknown"
        step = 0
        max_steps = 500
        try:
            while not env.is_game_over and step < max_steps:
                # ★ 防御 current_piece 为 None
                if env.current_piece is None:
                    env.begin_turn_host()
                    if env.current_piece is None:
                        if verbose:
                            print(f"  Step {step}: current_piece is None, breaking")
                        break
                
                if env.current_piece.team == 1:
                    action = player1_strategy(env)
                    actor_name = player1_name
                    stats_key = "model1" if player1_model is model1 else "model2"
                else:
                    action = player2_strategy(env)
                    actor_name = player2_name
                    stats_key = "model1" if player2_model is model1 else "model2"

                # ★ 动作统计
                action_stats[stats_key]["total"] += 1
                if hasattr(action, 'move') and action.move:
                    action_stats[stats_key]["move"] += 1
                if hasattr(action, 'attack') and action.attack:
                    action_stats[stats_key]["attack"] += 1
                if hasattr(action, 'spell') and action.spell:
                    action_stats[stats_key]["spell"] += 1
                if not (getattr(action, 'move', False) or getattr(action, 'attack', False) or getattr(action, 'spell', False)):
                    action_stats[stats_key]["skip_all"] += 1
                
                if verbose:
                    cp = env.current_piece
                    print(f"  Step {step}: 棋子{cp.id}(队{cp.team}) [{actor_name}] → {_describe_action(action)}")

                # ★ 写入动作日志到文件
                cp = env.current_piece
                log_fh.write(f"  Step {step}: {_describe_piece(cp)} [{actor_name}]\n")
                log_fh.write(f"    → {_describe_action(action)}\n")
                log_fh.flush()
                
                step_with_action(env, action)
                step += 1
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 判断胜负 ★ 修复：明确区分"全灭"和"步数耗尽"两种平局
        p1_alive = any(p.is_alive for p in env.player1.pieces)
        p2_alive = any(p.is_alive for p in env.player2.pieces)
        
        if p1_alive and not p2_alive:
            winner = 1
            game_result = "P1胜(全灭)"
        elif p2_alive and not p1_alive:
            winner = 2
            game_result = "P2胜(全灭)"
        elif step >= max_steps:
            winner = 0
            game_result = f"平局(步数耗尽, P1存活={p1_alive}, P2存活={p2_alive})"
        else:
            winner = 0
            game_result = f"平局(P1存活={p1_alive}, P2存活={p2_alive})"
        
        # 统计结果（从模型1的角度）
        if game_idx < games_per_side:
            if winner == 1:
                model1_wins += 1
            elif winner == 2:
                model2_wins += 1
            else:
                draws += 1
        else:
            if winner == 2:
                model1_wins += 1
            elif winner == 1:
                model2_wins += 1
            else:
                draws += 1
        
        if verbose:
            print(f"  结果: {game_result} | 累计: M1={model1_wins}W M2={model2_wins}W D={draws}")

        # ★ 写入比赛结果到日志
        log_fh.write(f"  结果: {game_result} (step={step}) | 累计: M1={model1_wins}W M2={model2_wins}W D={draws}\n\n")
        log_fh.flush()
        
        # 更新进度条
        total_done = model1_wins + model2_wins + draws
        battle_bar.set_postfix({
            'M1_W': model1_wins,
            'M2_W': model2_wins,
            'D': draws,
            'M1%': f"{model1_wins/total_done:.1%}" if total_done > 0 else "0.0%"
        })
    
    battle_bar.close()
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"=== Battle Results ===")
    print(f"Model1: {model1_path}")
    print(f"Model2: {model2_path}")
    print(f"Total games: {total_games}")
    print(f"Model1 wins: {model1_wins} ({model1_wins/total_games:.1%})")
    print(f"Model2 wins: {model2_wins} ({model2_wins/total_games:.1%})")
    print(f"Draws:      {draws} ({draws/total_games:.1%})")
    print(f"{'='*60}")

    # ★ 动作分布诊断
    print(f"\n--- Action Distribution ---")
    for mkey, mlabel in [("model1", "Model1"), ("model2", "Model2")]:
        stats = action_stats[mkey]
        tot = max(stats["total"], 1)
        print(f"  {mlabel}: total_actions={stats['total']}, "
              f"move={stats['move']}({stats['move']/tot:.0%}), "
              f"attack={stats['attack']}({stats['attack']/tot:.0%}), "
              f"spell={stats['spell']}({stats['spell']/tot:.0%}), "
              f"skip_all={stats['skip_all']}({stats['skip_all']/tot:.0%})")
    print(f"{'='*60}")

    # ★ 写入最终总结到日志并关闭
    log_fh.write(f"\n{'='*60}\n")
    log_fh.write(f"=== Battle Results ===\n")
    log_fh.write(f"Model1: {model1_path}\n")
    log_fh.write(f"Model2: {model2_path}\n")
    log_fh.write(f"Total games: {total_games}\n")
    log_fh.write(f"Model1 wins: {model1_wins} ({model1_wins/total_games:.1%})\n")
    log_fh.write(f"Model2 wins: {model2_wins} ({model2_wins/total_games:.1%})\n")
    log_fh.write(f"Draws:      {draws} ({draws/total_games:.1%})\n")
    log_fh.write(f"\n--- Action Distribution ---\n")
    for mkey, mlabel in [("model1", "Model1"), ("model2", "Model2")]:
        stats = action_stats[mkey]
        tot = max(stats["total"], 1)
        log_fh.write(f"  {mlabel}: total={stats['total']}, "
                     f"move={stats['move']}({stats['move']/tot:.0%}), "
                     f"attack={stats['attack']}({stats['attack']/tot:.0%}), "
                     f"spell={stats['spell']}({stats['spell']/tot:.0%}), "
                     f"skip={stats['skip_all']}({stats['skip_all']/tot:.0%})\n")
    log_fh.write(f"{'='*60}\n")
    log_fh.close()
    print(f"Log saved to: {log_file}")
    
    return model1_wins, model2_wins, draws


def parse_args():
    parser = argparse.ArgumentParser(description="Battle two models")
    parser.add_argument("--model1", required=True, help="Path to model1")
    parser.add_argument("--model2", required=True, help="Path to model2")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--init-strategy", default="archer29", help="Init strategy")
    parser.add_argument("--games-per-side", type=int, default=5, help="Games per side")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations")
    parser.add_argument("--verbose", action="store_true", help="Print ActionSet details at each step")
    parser.add_argument("--log-file", default=None, help="Path to action log file (default: auto-generated)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    battle_models(
        model1_path=args.model1,
        model2_path=args.model2,
        device=device,
        init_strategy=args.init_strategy,
        games_per_side=args.games_per_side,
        simulations=args.simulations,
        verbose=args.verbose,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()

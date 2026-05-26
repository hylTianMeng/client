import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm

import torch

from model import TacticalPolicyNet
from state_processor import StateProcessor
from dataset_utils import create_dataloader, build_model
from self_play import collect_self_play_examples, save_npz
from replay_buffer import ReplayBuffer
from model_evaluator import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play training for TacticalPolicyNet")
    parser.add_argument("--save-dir", default="training_data")
    parser.add_argument("--resume-model", help="Path to existing model checkpoint")
    parser.add_argument("--resume-best-model", help="Path to best model checkpoint for evaluation")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--player1-init", default="archer29,archer22")
    parser.add_argument("--player2-init", default="archer29")
    parser.add_argument("--player1-policy", default="puct")
    parser.add_argument("--player2-policy", default="aggressive")
    parser.add_argument("--puct-simulations", type=int, default=100)
    parser.add_argument("--val-data", help="Optional validation .npz file")
    # 新增参数
    parser.add_argument("--buffer-size", type=int, default=80000, help="Replay buffer size")
    parser.add_argument("--eval-games-per-side", type=int, default=5, help="Number of games per side for evaluation")
    parser.add_argument("--eval-win-rate-threshold", type=float, default=0.5, help="Win rate threshold to keep new model")
    parser.add_argument("--eval-init-strategy", default="archer29", help="Init strategy for evaluation")
    parser.add_argument("--enable-eval", action="store_true", default=True, help="Enable model evaluation")
    parser.add_argument("--disable-eval", action="store_true", help="Disable model evaluation")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate model every N iterations")
    args = parser.parse_args()
    # 处理评估开关
    if args.disable_eval:
        args.enable_eval = False
    return args


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    
    # 创建以运行时间为名字的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    _ensure_dir(run_dir)
    best_model_dir = os.path.join(run_dir, "best_model")
    _ensure_dir(best_model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")

    processor = StateProcessor()
    model = TacticalPolicyNet(in_channels=19)
    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
        print(f"Loaded model from {args.resume_model}")
    
    # 加载最佳模型用于评估
    best_model = None
    if args.resume_best_model:
        best_model = TacticalPolicyNet(in_channels=19)
        best_model.load_state_dict(torch.load(args.resume_best_model, map_location=device))
        best_model.to(device)
        print(f"Loaded best model from {args.resume_best_model}")
    
    # 创建样本池
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用进度条
    iteration_bar = tqdm(range(1, args.iterations + 1), desc="Training iterations")
    
    for iteration in iteration_bar:
        iteration_start_time = time.time()
        
        # 自对弈收集数据
        examples = collect_self_play_examples(
            model=model,
            processor=processor,
            player1_init=args.player1_init,
            player2_init=args.player2_init,
            player1_policy=args.player1_policy,
            player2_policy=args.player2_policy,
            games=args.games_per_iter,
            device=device,
            simulations=args.puct_simulations,
        )
        
        # 添加到样本池
        replay_buffer.add(examples)
        
        # 输出当前数据库长度
        print(f"Current replay buffer size: {replay_buffer.size()}")
        
        # 保存本次 iteration 的数据
        data_path = os.path.join(run_dir, f"iteration_{iteration}_data.npz")
        save_npz(data_path, examples)
        
        # 从样本池创建训练数据
        all_examples = replay_buffer.get_all()
        temp_data_path = os.path.join(run_dir, f"temp_buffer_data.npz")
        save_npz(temp_data_path, all_examples)
        
        train_loader = create_dataloader(temp_data_path, batch_size=args.batch_size, shuffle=True)
        val_loader = None
        if args.val_data:
            val_loader = create_dataloader(args.val_data, batch_size=args.batch_size, shuffle=False)
        
        # 训练模型
        model_path = os.path.join(run_dir, f"iteration_{iteration}_model.pt")
        build_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=model_path,
        )
        
        # 评估新模型（根据开关和间隔）
        if args.enable_eval and (iteration % args.eval_interval == 0 or iteration == 1):
            if best_model is not None:
                print(f"\n=== Evaluating new model vs best model ===")
                win_rate, wins, losses, draws = evaluate_model(
                    new_model=model,
                    old_model=best_model,
                    processor=processor,
                    device=device,
                    init_strategy=args.eval_init_strategy,
                    games_per_side=args.eval_games_per_side,
                    simulations=args.puct_simulations,
                )
                total_games = wins + losses + draws
                print(f"Evaluation results: Win rate={win_rate:.2%} ({wins}W/{losses}L/{draws}D out of {total_games} games)")
                
                # 如果胜率高于阈值，更新最佳模型
                if win_rate > args.eval_win_rate_threshold:
                    best_model = TacticalPolicyNet(in_channels=19)
                    best_model.load_state_dict(model.state_dict())
                    best_model.to(device)
                    best_model_path = os.path.join(best_model_dir, "best_model.pt")
                    torch.save(best_model.state_dict(), best_model_path)
                    print(f"New model accepted as best model (win rate {win_rate:.2%} > {args.eval_win_rate_threshold:.2%})")
                else:
                    print(f"New model rejected (win rate {win_rate:.2%} <= {args.eval_win_rate_threshold:.2%})")
            else:
                # 第一次迭代，直接保存为最佳模型
                best_model = TacticalPolicyNet(in_channels=19)
                best_model.load_state_dict(model.state_dict())
                best_model.to(device)
                best_model_path = os.path.join(best_model_dir, "best_model.pt")
                torch.save(best_model.state_dict(), best_model_path)
                print(f"First model saved as best model")
        else:
            print(f"Skipping evaluation (iteration {iteration}, enable_eval={args.enable_eval}, eval_interval={args.eval_interval})")
        
        # 计算时间和预测剩余时间
        iteration_time = time.time() - iteration_start_time
        elapsed_time = time.time() - start_time
        avg_time_per_iteration = elapsed_time / iteration
        remaining_iterations = args.iterations - iteration
        estimated_remaining_time = avg_time_per_iteration * remaining_iterations
        
        # 更新进度条
        iteration_bar.set_postfix({
            'buffer_size': replay_buffer.size(),
            'iter_time': f"{iteration_time:.1f}s",
            'elapsed': f"{elapsed_time/60:.1f}m",
            'remaining': f"{estimated_remaining_time/60:.1f}m"
        })
        
        # 保存最新模型
        latest_model_path = os.path.join(run_dir, "latest_model.pt")
        torch.save(model.state_dict(), latest_model_path)
    
    # 关闭进度条
    iteration_bar.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Final best model saved at {best_model_dir}/best_model.pt")


if __name__ == "__main__":
    main()

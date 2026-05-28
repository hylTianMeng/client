import argparse
import gc
import os
import sys
import time
import copy
import threading
import faulthandler
from datetime import datetime
from tqdm import tqdm

import torch

from model import TacticalPolicyNet
from state_processor import StateProcessor
from dataset_utils import create_dataloader, build_model
from self_play import collect_self_play_examples, save_npz
from replay_buffer import ReplayBuffer
from model_evaluator import evaluate_model

# ★ 增大 Python 递归限制（MCTS 树深度保护）
sys.setrecursionlimit(20000)

# ★ 增大线程栈空间（Windows 默认 1MB 不够深层 MCTS 调用链）
threading.stack_size(8 * 1024 * 1024)  # 8 MB

# ★ 启用 faulthandler：当发生 segfault 时输出 Python 调用栈
faulthandler.enable()


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play training for TacticalPolicyNet")
    parser.add_argument("--save-dir", default="training_data")
    parser.add_argument("--resume-model", help="Path to existing model checkpoint")
    parser.add_argument("--resume-baseline-model", help="Path to baseline model for evaluation (initial untrained)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games-per-iter", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--player1-init", default="archer29")
    parser.add_argument("--player2-init", default="archer29")
    parser.add_argument("--player1-policy", default="puct")
    parser.add_argument("--player2-policy", default="aggressive")
    parser.add_argument("--puct-simulations", type=int, default=100)
    parser.add_argument("--val-data", help="Optional validation .npz file")
    parser.add_argument("--load-data", help="Path to .npz file to pre-load into replay buffer before training")
    parser.add_argument("--buffer-size", type=int, default=5000, help="Replay buffer size")
    parser.add_argument("--eval-games-per-side", type=int, default=1, help="Games per side for evaluation")
    parser.add_argument("--eval-win-rate-threshold", type=float, default=0.55, help="Win rate to save as best model")
    parser.add_argument("--eval-init-strategy", default="archer29", help="Init strategy for evaluation")
    parser.add_argument("--enable-eval", action="store_true", default=True, help="Enable model evaluation")
    parser.add_argument("--disable-eval", action="store_true", help="Disable model evaluation")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N iterations")
    parser.add_argument("--eval-simulations", type=int, default=100, help="MCTS simulations for evaluation")
    args = parser.parse_args()
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
    print(f"Training: P1(puct) vs P2({args.player2_policy}), eval every {args.eval_interval} iters")
    print(f"Init: P1={args.player1_init}, P2={args.player2_init}")
    
    processor = StateProcessor()
    model = TacticalPolicyNet(in_channels=19)
    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
        print(f"Loaded model from {args.resume_model}")
    model.to(device)
    
    # ★ 保存初始未训练模型作为 baseline
    baseline_model = TacticalPolicyNet(in_channels=19)
    if args.resume_baseline_model:
        baseline_model.load_state_dict(torch.load(args.resume_baseline_model, map_location=device))
        print(f"Loaded baseline model from {args.resume_baseline_model}")
    else:
        baseline_model.load_state_dict(copy.deepcopy(model.state_dict()))
    baseline_model.to(device)
    baseline_model.eval()
    
    baseline_path = os.path.join(run_dir, "baseline_model.pt")
    torch.save(baseline_model.state_dict(), baseline_path)
    print(f"Baseline model saved to {baseline_path}")
    
    # 最佳模型跟踪（当前训练过程中历史最佳）
    best_model_state = copy.deepcopy(model.state_dict())
    best_win_rate = 0.0
    
    # 创建样本池
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # ★ 从已有 .npz 数据集预加载样本
    if args.load_data:
        if not os.path.exists(args.load_data):
            print(f"WARNING: --load-data file not found: {args.load_data}")
        else:
            print(f"Pre-loading replay buffer from: {args.load_data}")
            replay_buffer.load_from_file(args.load_data)
    
    start_time = time.time()
    iteration_bar = tqdm(range(1, args.iterations + 1), desc="Training iterations")
    
    for iteration in iteration_bar:

        print(f"\n{'='*25} Iteration {iteration} {'='*25}", flush=True)
        iteration_start_time = time.time()
        
        try:
            # === 自对弈收集数据 ===
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
            
            replay_buffer.add(examples)
            print(f"  Buffer size: {replay_buffer.size()}")
            
            # 保存本次 iteration 数据
            data_path = os.path.join(run_dir, f"iteration_{iteration}_data.npz")
            save_npz(data_path, examples)
            
            # === 训练模型 ===
            all_examples = replay_buffer.get_all()
            temp_data_path = os.path.join(run_dir, "temp_buffer_data.npz")
            save_npz(temp_data_path, all_examples)
            
            train_loader = create_dataloader(temp_data_path, batch_size=args.batch_size, shuffle=True)
            val_loader = None
            if args.val_data:
                val_loader = create_dataloader(args.val_data, batch_size=args.batch_size, shuffle=False)
            
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
            
            # ★ 训练后清理数据加载器引用
            del train_loader
            if val_loader is not None:
                del val_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # === 评估：当前模型 vs 初始 baseline ===
            if args.enable_eval and iteration % args.eval_interval == 0:
                print(f"\n{'='*50}")
                print(f"  Evaluation at iteration {iteration}: current model vs baseline (untrained)")
                print(f"{'='*50}")
                
                win_rate, wins, losses, draws = evaluate_model(
                    new_model=model,
                    old_model=baseline_model,
                    processor=processor,
                    device=device,
                    init_strategy=args.eval_init_strategy,
                    games_per_side=args.eval_games_per_side,
                    simulations=args.eval_simulations,
                    verbose=True,
                )
                total_games = wins + losses + draws
                print(f"  vs baseline: {wins}W/{losses}L/{draws}D | win_rate={win_rate:.2%}")
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_path = os.path.join(best_model_dir, "best_model.pt")
                    torch.save(best_model_state, best_path)
                    print(f"  >>> New best model! win_rate={win_rate:.2%} saved to {best_path}")
                else:
                    print(f"  Best so far: {best_win_rate:.2%}")
        
        except Exception as e:
            print(f"\n  !!! ERROR at iteration {iteration}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print(f"  !!! Saving checkpoint and continuing...", flush=True)
            # 保存崩溃时的模型
            crash_path = os.path.join(run_dir, f"crash_iter{iteration}_model.pt")
            torch.save(model.state_dict(), crash_path)
            # ★ 强制 GC 清理可能损坏的内存
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # 跳过这次 iteration，继续下一个
            continue
        
        # === 时间统计 ===
        iteration_time = time.time() - iteration_start_time
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iteration
        remaining = avg_time * (args.iterations - iteration)
        
        iteration_bar.set_postfix({
            'buf': replay_buffer.size(),
            'best': f"{best_win_rate:.1%}",
            'iter': f"{iteration_time:.0f}s",
            'remain': f"{remaining/60:.0f}m"
        })
        
        # 保存最新模型
        latest_path = os.path.join(run_dir, "latest_model.pt")
        torch.save(model.state_dict(), latest_path)
        
        # ★ 强制内存回收
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    iteration_bar.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best win rate vs baseline: {best_win_rate:.2%}")
    print(f"Baseline model: {baseline_path}")
    print(f"Best model: {best_model_dir}/best_model.pt")
    print(f"Latest model: {run_dir}/latest_model.pt")


if __name__ == "__main__":
    main()

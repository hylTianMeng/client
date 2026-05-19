import argparse
import os

import torch

from model import TacticalPolicyNet
from state_processor import StateProcessor
from dataset_utils import create_dataloader, build_model
from self_play import collect_self_play_examples, save_npz


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play training for TacticalPolicyNet")
    parser.add_argument("--save-dir", default="training_data")
    parser.add_argument("--resume-model", help="Path to existing model checkpoint")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games-per-iter", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--player1-init", default="archer30")
    parser.add_argument("--player2-init", default="archer22")
    parser.add_argument("--player2-policy", default="puct")
    parser.add_argument("--puct-simulations", type=int, default=16)
    parser.add_argument("--val-data", help="Optional validation .npz file")
    return parser.parse_args()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    _ensure_dir(args.save_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processor = StateProcessor()
    model = TacticalPolicyNet(in_channels=19)
    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))

    for iteration in range(1, args.iterations + 1):
        print(f"=== iteration {iteration} ===")
        examples = collect_self_play_examples(
            model=model,
            processor=processor,
            player1_init=args.player1_init,
            player2_init=args.player2_init,
            player2_policy=args.player2_policy,
            games=args.games_per_iter,
            device=device,
            simulations=args.puct_simulations,
        )

        data_path = os.path.join(args.save_dir, f"iteration_{iteration}_data.npz")
        model_path = os.path.join(args.save_dir, f"iteration_{iteration}_model.pt")
        save_npz(data_path, examples)
        print(f"Saved self-play examples to {data_path}")

        train_loader = create_dataloader(data_path, batch_size=args.batch_size, shuffle=True)
        val_loader = None
        if args.val_data:
            val_loader = create_dataloader(args.val_data, batch_size=args.batch_size, shuffle=False)

        build_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=model_path,
        )
        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest_model.pt"))
        print(f"Saved latest model checkpoint to {os.path.join(args.save_dir, 'latest_model.pt')}")


if __name__ == "__main__":
    main()

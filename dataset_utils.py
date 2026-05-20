import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from state_processor import StateProcessor


def target_to_index(target: np.ndarray, num_positions: int) -> np.ndarray:
    if target.ndim == 0:
        # 标量索引（-1 或 0-399）
        if target == -1:
            return np.array([-1], dtype=np.int64)  # 使用num_positions作为-1的映射
        return np.array([int(target)], dtype=np.int64)

    if target.ndim == 1:
        return target.astype(np.int64)

    if target.ndim == 2 and target.shape == (2,):
        return np.array([target[0] * 20 + target[1]], dtype=np.int64)

    if target.ndim == 2 and target.shape == (20, 20):
        if target.dtype != np.int64:
            target = target.astype(np.int64)
        coords = np.argwhere(target != 0)
        if coords.shape[0] == 0:
            raise ValueError("One-hot target has no positive entry")
        if coords.shape[0] > 1:
            coords = coords[:1]
        index = coords[0][0] * 20 + coords[0][1]
        return np.array([index], dtype=np.int64)

    raise ValueError("Unsupported target shape")


class TacticsDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)

        self.states = data["states"]
        self.switch = data["switch"]
        self.move_target = data["move"]
        self.attack_target = data["attack"]
        self.spell_target = data["spell"]
        self.value = data["value"]
        self.stage = data["stage"]

        if self.states.ndim != 4:
            raise ValueError("states must have shape (N, C, 20, 20)")
        if self.states.shape[1] != 19:
            raise ValueError("states must have 19 channels")
        if self.stage.ndim != 1:
            raise ValueError("stage must be a 1D array of stage ids")

        self.num_samples = self.states.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        state = self.states[idx] # 归一化重复,已修正
        switch_label = int(self.switch[idx])
        move_index = target_to_index(self.move_target[idx], 400)[0]
        attack_index = target_to_index(self.attack_target[idx], 400)[0]
        spell_index = target_to_index(self.spell_target[idx], 1600)[0]
        stage_label = int(self.stage[idx])

        return {
            "state": torch.tensor(state, dtype=torch.float32),
            "switch": torch.tensor(switch_label, dtype=torch.long),
            "move": torch.tensor(move_index, dtype=torch.long),
            "attack": torch.tensor(attack_index, dtype=torch.long),
            "spell": torch.tensor(spell_index, dtype=torch.long),
            "value": torch.tensor(float(self.value[idx]), dtype=torch.float32),
            "stage": torch.tensor(stage_label, dtype=torch.long),
        }


def compute_loss(
    outputs: dict,
    targets: dict,
    switch_loss_fn: nn.Module,
    action_loss_fn: nn.Module,
    value_loss_fn: nn.Module,
) -> tuple:
    switch_loss = switch_loss_fn(outputs["switch_logits"], targets["switch"])
    value_loss = value_loss_fn(outputs["value"].view(-1), targets["value"])

    stage = targets["stage"]
    active_mask = (targets["switch"] == 1).float()
    move_stage_mask = (stage == 0).float()
    attack_stage_mask = (stage == 1).float()
    spell_stage_mask = (stage == 2).float()

    # 创建mask来忽略-1索引（表示不执行该动作）
    valid_move_mask = (targets["move"] >= 0).float()
    valid_attack_mask = (targets["attack"] >= 0).float()
    valid_spell_mask = (targets["spell"] >= 0).float()

    move_loss = action_loss_fn(outputs["move_logits"], targets["move"])
    attack_loss = action_loss_fn(outputs["attack_logits"], targets["attack"])
    spell_logits = outputs["spell_logits"].view(outputs["spell_logits"].shape[0], -1)
    spell_loss = action_loss_fn(spell_logits, targets["spell"])

    # 只对有效索引（非-1）且激活的样本计算loss
    move_active = (active_mask * move_stage_mask * valid_move_mask).sum().clamp(min=1.0)
    attack_active = (active_mask * attack_stage_mask * valid_attack_mask).sum().clamp(min=1.0)
    spell_active = (active_mask * spell_stage_mask * valid_spell_mask).sum().clamp(min=1.0)

    move_loss = (move_loss * active_mask * move_stage_mask * valid_move_mask).sum() / move_active
    attack_loss = (attack_loss * active_mask * attack_stage_mask * valid_attack_mask).sum() / attack_active
    spell_loss = (spell_loss * active_mask * spell_stage_mask * valid_spell_mask).sum() / spell_active

    total_loss = switch_loss + value_loss
    if move_stage_mask.any():
        total_loss += move_loss
    if attack_stage_mask.any():
        total_loss += attack_loss
    if spell_stage_mask.any():
        total_loss += spell_loss

    return total_loss, {
        "switch": switch_loss.item(),
        "value": value_loss.item(),
        "move": move_loss.item(),
        "attack": attack_loss.item(),
        "spell": spell_loss.item(),
    }


def collate_batch(batch):
    """
    batch 是一个列表，每个元素是 __getitem__ 返回的字典
    """
    return {
        "state": torch.stack([item["state"] for item in batch]),
        "switch": torch.stack([item["switch"] for item in batch]),
        "move": torch.stack([item["move"] for item in batch]),
        "attack": torch.stack([item["attack"] for item in batch]),
        "spell": torch.stack([item["spell"] for item in batch]),
        "value": torch.stack([item["value"] for item in batch]),
        "stage": torch.stack([item["stage"] for item in batch]),
    }


def create_dataloader(path: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TacticsDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def build_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: str,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    switch_loss_fn = nn.CrossEntropyLoss()
    action_loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-1) # 忽略-1索引
    value_loss_fn = nn.MSELoss()

    model.to(device)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            for key in ("state", "switch", "move", "attack", "spell", "value", "stage"):
                batch[key] = batch[key].to(device)

            outputs = model(batch["state"])
            loss, _ = compute_loss(
                outputs,
                batch,
                switch_loss_fn,
                action_loss_fn,
                value_loss_fn,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    for key in ("state", "switch", "move", "attack", "spell", "value", "stage"):
                        batch[key] = batch[key].to(device)
                    outputs = model(batch["state"])
                    loss, _ = compute_loss(
                        outputs,
                        batch,
                        switch_loss_fn,
                        action_loss_fn,
                        value_loss_fn,
                    )
                    val_loss += loss.item()
                    val_steps += 1
            avg_val_loss = val_loss / max(val_steps, 1)
            print(f"Epoch {epoch}: val_loss={avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

    if val_loader is None:
        torch.save(model.state_dict(), save_path)
        print(f"Saved final model to {save_path}")

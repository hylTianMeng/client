import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from tqdm import tqdm

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

        # ★ 可选：MCTS 访问分布（软标签）
        self.has_probs = "move_probs" in data
        if self.has_probs:
            self.move_probs = data["move_probs"]
            self.attack_probs = data["attack_probs"]
            self.spell_probs = data["spell_probs"]
        else:
            self.move_probs = None
            self.attack_probs = None
            self.spell_probs = None

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
        state = self.states[idx]
        switch_label = int(self.switch[idx])
        move_index = target_to_index(self.move_target[idx], 400)[0]
        attack_index = target_to_index(self.attack_target[idx], 400)[0]
        spell_index = target_to_index(self.spell_target[idx], 1600)[0]
        stage_label = int(self.stage[idx])

        result = {
            "state": torch.tensor(state, dtype=torch.float32),
            "switch": torch.tensor(switch_label, dtype=torch.long),
            "move": torch.tensor(move_index, dtype=torch.long),
            "attack": torch.tensor(attack_index, dtype=torch.long),
            "spell": torch.tensor(spell_index, dtype=torch.long),
            "value": torch.tensor(float(self.value[idx]), dtype=torch.float32),
            "stage": torch.tensor(stage_label, dtype=torch.long),
        }

        # ★ 可选：软标签（访问分布）
        if self.has_probs:
            result["move_probs"] = torch.tensor(self.move_probs[idx], dtype=torch.float32)
            result["attack_probs"] = torch.tensor(self.attack_probs[idx], dtype=torch.float32)
            result["spell_probs"] = torch.tensor(self.spell_probs[idx], dtype=torch.float32)
        else:
            result["move_probs"] = torch.zeros(400, dtype=torch.float32)
            result["attack_probs"] = torch.zeros(400, dtype=torch.float32)
            result["spell_probs"] = torch.zeros(1600, dtype=torch.float32)

        return result


def compute_loss(
    outputs: dict,
    targets: dict,
    switch_loss_fn: nn.Module,
    action_loss_fn: nn.Module,
    value_loss_fn: nn.Module,
) -> tuple:
    """计算损失。

    支持两种模式：
    1. 硬标签模式（无 move_probs）：使用交叉熵，目标为单一动作索引。
    2. 软标签模式（有 move_probs）：使用 KL 散度，目标为 MCTS 访问分布。

    ★ 策略头按 value 加权：胜利方动作权重高，失败方动作权重低。
    ★ value head 不加权：需要学习区分好坏局面。
    """
    value = targets["value"]
    stage = targets["stage"]
    batch_size = value.shape[0]
    device = value.device

    # 策略权重：value 从 [-1,1] 映射到 [0.1, 1.0]
    policy_weight = torch.clamp((value + 1.0) / 2.0, 0.1, 1.0)
    pw_sum = policy_weight.sum().clamp(min=1.0)

    # Switch 损失（加权交叉熵）
    switch_per_sample = switch_loss_fn(outputs["switch_logits"], targets["switch"])
    switch_loss = (switch_per_sample * policy_weight).sum() / pw_sum

    # Value 损失（不加权）
    value_loss = value_loss_fn(outputs["value"].view(-1), targets["value"])

    active_mask = (targets["switch"] == 1).float()
    move_stage_mask = (stage == 0).float()
    attack_stage_mask = (stage == 1).float()
    spell_stage_mask = (stage == 2).float()

    valid_move_mask = (targets["move"] >= 0).float()
    valid_attack_mask = (targets["attack"] >= 0).float()
    valid_spell_mask = (targets["spell"] >= 0).float()

    # ★ 判断是否有软标签
    has_move_probs = "move_probs" in targets and targets["move_probs"].sum() > 0
    has_attack_probs = "attack_probs" in targets and targets["attack_probs"].sum() > 0
    has_spell_probs = "spell_probs" in targets and targets["spell_probs"].sum() > 0

    # --- Move loss ---
    if has_move_probs:
        # ★ KL 散度：以访问分布为目标
        move_log_probs = F.log_softmax(outputs["move_logits"], dim=-1)
        move_probs_target = targets["move_probs"].to(device).clamp(min=1e-8)
        move_per_sample = (move_probs_target * (move_probs_target.log() - move_log_probs)).sum(dim=-1)
    else:
        move_per_sample = action_loss_fn(outputs["move_logits"], targets["move"])

    # --- Attack loss ---
    if has_attack_probs:
        attack_log_probs = F.log_softmax(outputs["attack_logits"], dim=-1)
        attack_probs_target = targets["attack_probs"].to(device).clamp(min=1e-8)
        attack_per_sample = (attack_probs_target * (attack_probs_target.log() - attack_log_probs)).sum(dim=-1)
    else:
        attack_per_sample = action_loss_fn(outputs["attack_logits"], targets["attack"])

    # --- Spell loss ---
    spell_logits = outputs["spell_logits"].view(batch_size, -1)
    if has_spell_probs:
        spell_log_probs = F.log_softmax(spell_logits, dim=-1)
        spell_probs_target = targets["spell_probs"].to(device).clamp(min=1e-8)
        spell_per_sample = (spell_probs_target * (spell_probs_target.log() - spell_log_probs)).sum(dim=-1)
    else:
        spell_per_sample = action_loss_fn(spell_logits, targets["spell"])

    # 加权 mask
    move_mask = active_mask * move_stage_mask * valid_move_mask * policy_weight
    attack_mask = active_mask * attack_stage_mask * valid_attack_mask * policy_weight
    spell_mask = active_mask * spell_stage_mask * valid_spell_mask * policy_weight

    move_active = move_mask.sum().clamp(min=1.0)
    attack_active = attack_mask.sum().clamp(min=1.0)
    spell_active = spell_mask.sum().clamp(min=1.0)

    move_loss = (move_per_sample * move_mask).sum() / move_active
    attack_loss = (attack_per_sample * attack_mask).sum() / attack_active
    spell_loss = (spell_per_sample * spell_mask).sum() / spell_active

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
    """batch 是 __getitem__ 返回的字典列表。"""
    result = {
        "state": torch.stack([item["state"] for item in batch]),
        "switch": torch.stack([item["switch"] for item in batch]),
        "move": torch.stack([item["move"] for item in batch]),
        "attack": torch.stack([item["attack"] for item in batch]),
        "spell": torch.stack([item["spell"] for item in batch]),
        "value": torch.stack([item["value"] for item in batch]),
        "stage": torch.stack([item["stage"] for item in batch]),
    }
    # ★ 可选软标签
    if "move_probs" in batch[0]:
        result["move_probs"] = torch.stack([item["move_probs"] for item in batch])
        result["attack_probs"] = torch.stack([item["attack_probs"] for item in batch])
        result["spell_probs"] = torch.stack([item["spell_probs"] for item in batch])
    return result


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
    switch_loss_fn = nn.CrossEntropyLoss(reduction="none")  # reduction='none' 以支持策略加权
    action_loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    value_loss_fn = nn.MSELoss()  # ★ reduction='mean'（value loss 不加权）

    model.to(device)

    best_val_loss = float("inf")

    # Epoch 级别的进度条
    epoch_bar = tqdm(range(1, epochs + 1), desc="Training epochs")

    # ★ 诊断：累计各头损失
    diag = {"switch": 0.0, "value": 0.0, "move": 0.0, "attack": 0.0, "spell": 0.0}

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_steps = 0
        # 重置诊断累计
        for k in diag:
            diag[k] = 0.0

        # ★ 策略熵累积（仅第一个 batch 计算，避免开销）
        first_batch_entropy = None

        for batch_idx, batch in enumerate(train_loader):
            for key in ("state", "switch", "move", "attack", "spell", "value", "stage"):
                batch[key] = batch[key].to(device)
            # ★ 可选软标签
            for key in ("move_probs", "attack_probs", "spell_probs"):
                if key in batch:
                    batch[key] = batch[key].to(device)

            outputs = model(batch["state"])
            loss, loss_items = compute_loss(
                outputs,
                batch,
                switch_loss_fn,
                action_loss_fn,
                value_loss_fn,
            )

            # ★ NaN/Inf 检测
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at epoch {epoch}, batch {batch_idx}, skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            for k in diag:
                diag[k] += loss_items.get(k, 0.0)

            # ★ 第一个 batch 计算策略熵（诊断用）
            if batch_idx == 0:
                with torch.no_grad():
                    move_ent = -(F.softmax(outputs["move_logits"][:1], dim=-1) *
                                  F.log_softmax(outputs["move_logits"][:1], dim=-1)).sum(dim=-1).mean().item()
                    attack_ent = -(F.softmax(outputs["attack_logits"][:1], dim=-1) *
                                   F.log_softmax(outputs["attack_logits"][:1], dim=-1)).sum(dim=-1).mean().item()
                    switch_ent = -(F.softmax(outputs["switch_logits"][:1], dim=-1) *
                                   F.log_softmax(outputs["switch_logits"][:1], dim=-1)).sum(dim=-1).mean().item()
                    val_pred = outputs["value"][:1].mean().item()
                    first_batch_entropy = (move_ent, attack_ent, switch_ent, val_pred)

        avg_train_loss = train_loss / max(train_steps, 1)

        # ★ 构建进度条后缀
        postfix = {'loss': f"{avg_train_loss:.4f}"}
        if first_batch_entropy is not None:
            postfix['H_mv'] = f"{first_batch_entropy[0]:.2f}"
            postfix['H_at'] = f"{first_batch_entropy[1]:.2f}"
            postfix['V'] = f"{first_batch_entropy[3]:.2f}"
        epoch_bar.set_postfix(postfix)
        
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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

    epoch_bar.close()

    if val_loader is None:
        torch.save(model.state_dict(), save_path)

    # ★ 诊断：打印分头平均损失
    steps = max(train_steps, 1)
    print(f"  Training complete. Per-head avg loss: "
          f"switch={diag['switch']/steps:.4f}, value={diag['value']/steps:.4f}, "
          f"move={diag['move']/steps:.4f}, attack={diag['attack']/steps:.4f}, "
          f"spell={diag['spell']/steps:.4f}")

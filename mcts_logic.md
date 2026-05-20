# MCTS 逻辑说明

## 概述

`PUCTMCTS` 实现了一个基于 **PUCT（Predictor + UCT）** 的蒙特卡洛树搜索，用于在回合制战棋游戏中决策。搜索将每一"回合"拆分为三个阶段（stage 0/1/2），依次决定：移动、攻击、施法。

核心流程：`select_action(env)` 以当前环境为根节点，执行 N 次（默认 16）模拟，每次模拟经历 **选择 → 扩展 → 评估 → 回溯** 四个步骤，最终返回访问次数最多的子节点对应的动作。

---

## 三阶段决策（Stage）

每个棋子的完整行动被分解为三个阶段：

| Stage | 含义 | 候选动作来源 | skip 概率来源 |
|-------|------|-------------|-------------|
| 0 | 移动 | `get_legal_moves()` 返回的可达格子 + `None`（停留） | `switch[0]` |
| 1 | 攻击 | `get_attackable_targets()` 返回的射程内敌人 + `None`（跳过） | `switch[0]` |
| 2 | 法术 | `env.get_available_spells()` × 合法目标 + `None`（跳过） | `switch[0]` |

模型为每个 stage 输出对应的动作概率分布（`move_logits` / `attack_logits` / `spell_logits`）和一个二分类 `switch_logits`（`[skip, execute]`），最终每个候选动作的 prior = `softmax(switch)[1] * softmax(action_head)[idx]`。

`is_leaf()` 条件：`stage >= 3` 或游戏结束。

---

## 核心数据结构：Node

```python
class Node:
    env        # 当前节点对应的环境快照（fork 出来的副本）
    stage      # 当前阶段 (0/1/2)
    parent     # 父节点
    action     # 从父节点到达此节点所执行的动作
    children   # {action_key: Node}
    visits     # 被选中次数
    value_sum  # 累计价值（从当前玩家视角）
    prior      # {action_key: float} 神经网络输出的先验概率
    expanded   # 是否已展开
```

`value` 属性返回 `value_sum / visits`，即平均价值。未访问过时返回 0。

---

## 一次模拟的完整流程

### 1. Selection（选择） — `_select_child(node)`

从当前节点沿树向下，选择 PUCT 分数最大的子节点：

```
PUCT(child) = Q(child) + c_puct * P(child) * sqrt(Σ visits) / (1 + child.visits)
```

- **Q**: 子节点的平均价值（exploitation）
- **P**: 神经网络给该动作的先验概率（来自 `node.prior`）
- **分母 √(Σvisits+1) / (1+visits)**: UCB 风格的探索奖励，访问少的节点会被优先探索
- **c_puct**: 探索系数，默认 1.0

一直选择到遇到未展开节点或叶节点为止。

### 2. Expansion（扩展） — `_expand(node)`

对非叶且未展开的节点：
1. 调用神经网络，获取该 stage 下所有候选动作的 prior 分布
2. 遍历所有候选动作，为每个动作 `fork_environment` 创建子环境，并 `execute_player_action` 执行该动作，生成子节点
3. 将 `expanded` 标记为 True

### 3. Evaluation（评估） — `_evaluate(node)`

三种情况：
- **游戏已结束**：根据存活队伍返回 +1.0（当前队伍胜）或 -1.0（败）或 0.0
- **叶节点（stage >= 3）**：调用神经网络输出 `value` 头，返回标量价值
- **未展开**：先 expand，再随机选一个子节点递归评估

### 4. Backup（回溯） — `_backup(node, value)`

从评估节点向上回溯到根节点，每层：
- `visits += 1`
- `value_sum += value`
- `value = -value`（翻转价值，因为交替视角）

---

## 动作构建细节

### `_make_move_action`（stage 0）
- 如果 candidate 非 None 且行动力 > 0：设置 `move=True, move_target=move`
- 否则 `move=False`

### `_make_attack_action`（stage 1）
- 如果 candidate 非 None 且行动力 > 0：构建 `AttackContext`，设置 attacker 和 target
- 否则 `attack=False`

### `_make_spell_action`（stage 2）
- 如果 candidate 非 None 且行动力 > 0 且有法术位：
  - **AOE 法术**：target 为 None，target_area 以施法者位置为中心
  - **单体法术**：从 `action_queue` 中查找目标棋子，找不到则 `spell=False`

所有动作在构建后通过 `_execute_partial_action` 调用 `env.execute_player_action(action)` 执行。

---

## 最终决策 — `select_action(env)`

1. `fork_environment(env)` 创建根节点（stage=0），立即展开
2. 执行 `simulations` 次模拟（while 循环 + expand/select/evaluate/backup）
3. 选 `root.children` 中 `visits` 最大的子节点，返回其 `action`
4. 若无子节点，返回空 `ActionSet()`

---

## 关键辅助函数

| 函数 | 来源 | 作用 |
|------|------|------|
| `fork_environment(env)` | `strategy_utils.py` | 深拷贝整个 Environment（棋盘、棋子、队列、法术等） |
| `get_legal_moves(env)` | `strategy_utils.py` | 返回当前棋子可达的所有 Point |
| `get_attackable_targets(env)` | `strategy_utils.py` | 返回射程内可攻击的敌方棋子列表 |
| `_build_model_output(env, stage_value)` | `mcts.py` | 构建 17×20×20 状态张量，送入神经网络，输出 logits + value |

---

## 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `simulations` | 16 | 每次决策执行的 MCTS 模拟次数 |
| `c_puct` | 1.0 | 探索系数，越大越倾向探索未访问节点 |
| `stage_value` | 0.3/0.6/1.0 | 输入给网络的状态标记，区分当前处于哪个阶段 |
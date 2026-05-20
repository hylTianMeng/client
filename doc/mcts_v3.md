# MCTS v3（分阶段 PUCT）说明（ai生成）

本文档说明 `mcts_v3.py` 中的 `MCTS`：一个将“单个棋子的回合”拆分为三段（移动→攻击→施法）的 PUCT 蒙特卡洛树搜索实现，并支持多环境批量推理（batch）以提升 GPU 利用率

> 适用场景：自博弈训练、对局决策；希望网络在 move/attack/spell 的**中间局面**上反复查询，从而让后续阶段的先验与价值评估真正以“已移动/已攻击后的局面”为条件。

---

## 1. 核心设计：把一个回合拆成 3 个 stage

在该项目的规则里，一个棋子在自己的回合内通常会尝试按顺序执行：

1. 移动（stage=0）
2. 攻击（stage=1）
3. 施法（stage=2）

`MCTS v3` 的树结构按 stage 串联：

- `stage=0 (move)` → `stage=1 (attack)` → `stage=2 (spell)` → `stage=0 (next piece)`

其中前三个阶段属于**同一个棋子**（同队），在 stage=2 之后才推进到下一个棋子（队伍可能变化）。

### 为什么要拆 stage？

- 让模型在 **move 后的真实局面**上再输出 attack priors；在 **move+attack 后的真实局面**上再输出 spell priors。
- 避免“一次性输出 move+attack+spell”带来的条件错配：后续动作先验没有看到前序动作的真实影响。

---

## 2. 状态如何逐步更新（small step update）

每个树节点 `_Node` 都持有一个独立的 `env`（环境快照）。扩展子节点时：

1. `child_env = fork_environment(node.env)` 复制环境
2. 构造当前 stage 的 **partial action**（只包含 move 或 attack 或 spell）
3. `child_env.execute_player_action(partial)` 在拷贝环境上执行该小动作
4. 若当前 stage 为 2（spell），再调用 `_advance_turn(child_env)` 进行队列轮转与回合推进

关键点：

- `env.execute_player_action` 只会执行 `ActionSet` 中标记为 True 的部分（move/attack/spell），不会自动轮转到下一个棋子。
- 因此 stage=0/1/2 的环境会按小动作“逐步累积变化”。

---

## 3. 节点结构（_Node）

节点 `_Node` 主要字段：

- `env`: 当前节点局面（Environment 快照）
- `stage`: 0/1/2
- `team`: 当前节点所属棋子（当前阶段行动者）的队伍
- `children`: `Dict[action_key, child_node]`
- `prior`: `Dict[action_key, P(s,a)]`（模型先验）
- `visits`, `value_sum`: 访问次数与累计价值
- `partial_action`: 从父节点到该节点所执行的小动作（ActionSet，仅一小步）

`value` 由 `value_sum / visits` 得到。

---

## 4. 候选动作生成与剪枝

### 4.1 候选生成

按 stage 生成候选：

- stage 0（move）：`[None(skip), stay, legal_moves...]`
- stage 1（attack）：`[None(skip)] + attackable_targets`
- stage 2（spell）：`[None(skip)] + [(spell,target)...]`
  - area spell：用 `(spell, None)` 代表“以施法者位置为中心”的选择
  - single-target spell：遍历 `env.get_spell_targets` 返回的目标

### 4.2 用模型先验构建 `prior`

`_infer(env, stage)` 输出：

- `switch`: 二分类概率（skip / exec）
- `move`, `attack`: 400 格子的概率分布（20×20 展平）
- `spell`: `num_spells × 400` 概率
- `value`: 标量局面价值

`_build_priors` 将其组合为 `prior[action_key]`：

- `p_skip = switch[0]`
- `p_exec = switch[1]`
- 对每个候选动作：
  - skip 用 `p_skip`
  - 非 skip 用 `p_exec * head_prob`（并做 EPS 下限）

### 4.3 move 的 top-k 剪枝（关键）

移动空间很大（最多 400），因此 stage=0 会按模型 `move` 概率排序，仅保留：

- `skip`
- `stay`
- 概率最高的 `top_k_move` 个 move（默认 12）

扩展时用 `prior` 过滤：

- 只有 `action_key` 在 `prior` 里的候选才会进入 `candidates` 并被扩展。

因此 stage=0 的子节点数通常 ≤ `1(skip)+1(stay)+top_k_move`。

---

## 5. 扩展（expand）：为每个候选建立子节点

`_expand(node)` 的核心流程：

1. 调模型 `_infer(node.env, node.stage)`
2. 构建 `node.prior`
3. 生成 `candidates`（按 `prior` 过滤后的候选）
4. 对每个 `c`：
   - 构造 `partial ActionSet`
   - fork 环境并执行该小动作
   - 若 stage==2，执行 `_advance_turn`
   - 创建 child node（stage 递进或回到 0）

注意：

- 扩展会把**过滤后的** `candidates` 全部扩展成子节点。
- 不会扩展被剪掉的 move（不在 `prior` 里）。

---

## 6. 选择（select）：PUCT 从子节点里挑一个继续走

`_select_child(node)` 使用：

- 利用项（exploitation）：`Q = child.value`
- 探索项（exploration）：

\[
U = c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b) + 1}}{1 + N(s,a)}
\]

最终分数：

\[
score = Q + U
\]

选择 `score` 最大的 child。

---

## 7. 评估（evaluate）：叶节点价值

`_evaluate(node)`：

- 若终局：根据 `node.team` 与胜负返回 `+1/-1/0`
- 否则：调用模型输出的 `value`

---

## 8. 回传（backup）：价值沿路径更新，并在换队时翻转

`_backup(leaf, value)` 从叶子往根回传：

- 每个节点：`visits += 1`, `value_sum += value`
- 仅当父子节点 `team` 不同（跨队）时，`value = -value`

为什么这样做？

- stage=0→1→2 属于同一队（同一个棋子回合），价值不应翻转。
- stage=2→下一棋子 stage=0 可能换队，此时才需要对抗翻转。

---

## 9. 输出动作：把 3 段 partial action 合并为一个 ActionSet

`_collect_full_action(root)`：

- 从根沿“访问次数最大”的路径走：stage0→stage1→stage2
- 把途中 child 的 `partial_action` 合并成一个完整 `ActionSet`（move/attack/spell 三段都可能被设置）
- 走到 stage=2 后再遇到 stage=0（下一个棋子）则停止

---

## 10. 批量模式：select_actions_batch

`select_actions_batch(envs)` 用于同时对多个局面做 MCTS，关键点：

- 选择/回传在 CPU 上独立进行
- **模型推理（_infer_batch）**在 GPU 上批量执行，减少推理开销

单轮 simulation 的结构：

1. 每棵树各自 select 到 leaf
2. 收集需要 expand 的 leaf，batch 推理并 expand
3. 收集需要 eval 的 leaf，batch 推理得到 value
4. 各自 backup

---

## 11. 关键参数与调参建议

- `simulations`：每次决策的模拟次数
  - 更大更强，但更慢
- `c_puct`：探索强度
  - 大：更愿意试先验高但访问少的动作
  - 小：更偏向当前 Q
- `top_k_move`：移动阶段保留的候选数
  - 大：更全面但分支更大
  - 小：更快但可能错过关键走位
- `max_depth`：限制树深，防止过深导致耗时和局面重复

---

## 12. 常见问题（FAQ）

### Q1：网络输入的多平面会随着小动作更新吗？

会。每个节点持有独立 `env`，并在扩展时对 `child_env` 执行 partial action；推理时调用 `StateProcessor.build_input(node.env, stage_value)`，因此输入平面随 `env` 改变而改变。

### Q2：为什么 stage=0 子节点通常是 14 个？

默认 `top_k_move=12`，并保留 `skip` 与 `stay`，因此通常 `1+1+12=14`（但若合法走位不足则更少）。

### Q3：攻击阶段 children 有多少？

`1(skip) + 可攻击目标数`（最多一般是对方存活且在射程内的棋子数）。

---

## 13. 相关入口（训练/自博弈）

- 自博弈收集：`self_play.collect_self_play_examples`
- 策略工厂：`StrategyFactory.get_puct_v3_action_strategy`、`get_puct_v3_batched_strategy`

> 若你希望训练时默认使用 v3，请确保 `player2-policy` 选择 `puct_v3`，并在策略工厂中统一将旧 `puct` 转发到 v3。



我的思考：
- device：cpu/cuda，记得看看是不是cuda
- 训练时：由于这个游戏空间太大，建议先用有一定策略的模型对战之后的到的片段来训练，否则模型需要很长时间才有可能开始学会一些东西
- 建议训练的时候采取某种随机手段，来切换对战双方的组合种类，目前天梯上普遍使用的是29*3的弓箭手集火最低血量的逻辑（我很难想到其他比这个好的逻辑，弓箭手手长，所以走位很难有效果），29-30的弓箭手只需要5箭杀死对手，而28及以下的弓箭手需要6箭，导致天梯上现在的胜负基本看先攻，谁先能够打满5箭基本就赢了，所以天梯上现在基本趋同。
- 训练的模拟次数也是需要调整的
- selfplay里边前期用策略对战，后期用模型对战或者混合对战均可（和你的实现基本差不多）


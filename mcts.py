import gc
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import Environment, Piece, Point, ActionSet, SpellContext, AttackContext, Area
from state_processor import StateProcessor
from strategy_utils import fork_environment, get_legal_moves, get_attackable_targets


class MCTS:
    """PUCT tree search with per-stage decomposition and correct turn progression.

    Tree structure
    --------------
    stage=0 (move) ──→ stage=1 (attack) ──→ stage=2 (spell) ──→ stage=0 (next piece)
         same team         same team           same team            team may differ

    - Each partial action forks the env and executes only that sub-action via
      ``Environment.execute_player_action``.  AP / spell_slots are decremented
      naturally as side-effects of those calls.
    - After stage=2, ``_advance_turn`` rotates the action queue, resets AP for
      the new current piece, advances delayed spells, and checks game-over.
      No further action is executed during queue rotation – the turn's three
      sub-actions have already been applied.
    - The model is re-queried at every node so attack/spell priors are
      conditioned on the *actual* intermediate state (move → attack → spell).

    Backup
    ------
    Value propagates up through all three stages *without* flipping (same
    team).  It flips only when stepping from one stage-0 node to its parent
    (which is a stage-2 node belonging to the *previous* piece, potentially
    the opponent).
    """

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model,
        processor: StateProcessor,
        device: torch.device,
        simulations: int = 160,
        c_puct: float = 1.0,
        max_depth: int = 60,
        top_k_move: int = 12,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        skip_prior_scale: float = 0.3,
    ):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.top_k_move = top_k_move
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.skip_prior_scale = skip_prior_scale  # ★ 降低 skip 的先验权重
        self._last_visit_dists = None  # ★ 存储最近一次搜索的访问分布

    # ------------------------------------------------------------------
    #  Node
    # ------------------------------------------------------------------

    class _Node:
        __slots__ = (
            "env",
            "stage",          # 0=move, 1=attack, 2=spell
            "team",           # team of the piece whose sub-turn this node belongs to
            "depth",
            "parent",
            "partial_action", # the ActionSet that brought us here from parent
            "children",
            "visits",
            "value_sum",
            "prior",
            "is_expanded",
        )

        def __init__(
            self,
            env: Environment,
            stage: int,
            team: int,
            depth: int = 0,
            parent=None,
            partial_action: Optional[ActionSet] = None,
        ):
            self.env = env
            self.stage = stage
            self.team = team
            self.depth = depth
            self.parent = parent
            self.partial_action = partial_action
            self.children: Dict[Tuple, "MCTS._Node"] = {}
            self.visits = 0
            self.value_sum = 0.0
            self.prior: Dict[Tuple, float] = {}
            self.is_expanded = False

        @property
        def value(self) -> float:
            if self.visits == 0:
                return 0.0
            return self.value_sum / self.visits

        def is_terminal(self) -> bool:
            return self.env.is_game_over

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        s = np.sum(exp)
        if s <= 0:
            return np.ones_like(logits) / logits.size
        return exp / s

    def _idx_of(self, pos) -> int:
        """位置→线性索引，基于当前棋盘宽度。"""
        w = self.processor.width
        return pos.x * w + pos.y

    @staticmethod
    def _normalize_priors(priors: Dict[Tuple, float]) -> Dict[Tuple, float]:
        total = sum(priors.values())
        if total <= 0:
            return priors
        return {k: v / total for k, v in priors.items()}

    # ------------------------------------------------------------------
    #  model inference
    # ------------------------------------------------------------------

    def _infer(self, env: Environment, stage: int) -> dict:
        """Run the model and return numpy dict of probs + scalar value.
        ★ 在 softmax 前对 move/attack/spell logits 应用合法动作掩码（AlphaZero 风格）。
        """
        stage_value = 0.3 if stage == 0 else 0.6 if stage == 1 else 1.0
        state = self.processor.build_input(env, stage_value)
        with torch.no_grad():
            x = torch.tensor(state[None, ...], dtype=torch.float32, device=self.device)
            outputs = self.model(x)

        switch = self._softmax(outputs["switch_logits"].cpu().numpy()[0])

        # ★ 移动掩码
        move_logits = outputs["move_logits"].cpu().numpy()[0]
        move_mask = self._build_move_mask(env)
        move = self._masked_softmax(move_logits, move_mask)

        # ★ 攻击掩码
        attack_logits = outputs["attack_logits"].cpu().numpy()[0]
        attack_mask = self._build_attack_mask(env)
        attack = self._masked_softmax(attack_logits, attack_mask)

        # ★ 法术掩码
        raw_spell = outputs["spell_logits"].cpu().numpy()[0]
        spell_mask = self._build_spell_mask(env)
        spell_logits_flat = raw_spell.reshape(-1)
        spell = self._masked_softmax(spell_logits_flat, spell_mask).reshape(raw_spell.shape)

        value = float(outputs["value"].cpu().numpy()[0])

        return {"switch": switch, "move": move, "attack": attack, "spell": spell, "value": value}

    def _infer_batch(self, items: List[Tuple[Environment, int]]) -> List[dict]:
        """Batch-infer a list of (env, stage) pairs. Returns list of output dicts."""
        if not items:
            return []

        states = []
        for env, stage in items:
            sv = 0.3 if stage == 0 else 0.6 if stage == 1 else 1.0
            states.append(self.processor.build_input(env, sv))
        batch = np.stack(states, axis=0)

        with torch.no_grad():
            x = torch.tensor(batch, dtype=torch.float32, device=self.device)
            outputs = self.model(x)

        switch_all = outputs["switch_logits"].cpu().numpy()
        move_all = outputs["move_logits"].cpu().numpy()
        attack_all = outputs["attack_logits"].cpu().numpy()
        spell_all = outputs["spell_logits"].cpu().numpy()
        value_all = outputs["value"].cpu().numpy()

        results = []
        for i in range(len(items)):
            env, stage = items[i]
            # ★ 对每个环境应用掩码
            move_mask = self._build_move_mask(env)
            attack_mask = self._build_attack_mask(env)
            spell_mask = self._build_spell_mask(env)
            raw_spell = spell_all[i]
            spell = self._masked_softmax(raw_spell.reshape(-1), spell_mask).reshape(raw_spell.shape)
            results.append({
                "switch": self._softmax(switch_all[i]),
                "move": self._masked_softmax(move_all[i], move_mask),
                "attack": self._masked_softmax(attack_all[i], attack_mask),
                "spell": spell,
                "value": float(value_all[i]),
            })
        return results

    # ------------------------------------------------------------------
    #  action masks (AlphaZero-style: mask logits before softmax)
    # ------------------------------------------------------------------

    def _num_positions(self) -> int:
        """棋盘格总数（如 40×40 = 1600）。"""
        return self.processor.width * self.processor.height

    def _pos_idx(self, x: int, y: int) -> int:
        """(x,y) → 线性索引。"""
        return x * self.processor.width + y

    def _get_friendly_positions(self, env: Environment) -> "set":
        """返回当前棋子同队所有存活棋子的线性索引集合。"""
        piece = env.current_piece
        if piece is None:
            return set()
        friendly = set()
        for p in env.action_queue:
            if p.is_alive and p.team == piece.team:
                friendly.add(self._pos_idx(p.position.x, p.position.y))
        return friendly

    def _get_enemy_pieces(self, env: Environment) -> "list":
        """返回当前棋子敌队所有存活棋子列表。"""
        piece = env.current_piece
        if piece is None:
            return []
        enemies = []
        for p in env.action_queue:
            if p.is_alive and p.team != piece.team:
                enemies.append(p)
        return enemies

    def _build_attack_mask(self, env: Environment) -> np.ndarray:
        """构建攻击合法掩码 (N,)，N = 棋盘格数。

        核心原则：
        - 只有敌人在攻击范围内，该位置才标记为 1。
        - 所有友方位置（包括自己）强制置 0。
        - 如果没有任何敌人在射程内，返回全 0 掩码（强制跳过攻击）。
        """
        NP = self._num_positions()
        mask = np.zeros(NP, dtype=np.float32)
        piece = env.current_piece
        if piece is None:
            return mask

        friendly_indices = self._get_friendly_positions(env)

        has_enemy_in_range = False
        for enemy in self._get_enemy_pieces(env):
            if env.is_in_attack_range(piece, enemy):
                idx = self._pos_idx(enemy.position.x, enemy.position.y)
                mask[idx] = 1.0
                has_enemy_in_range = True

        for fidx in friendly_indices:
            mask[fidx] = 0.0

        if not has_enemy_in_range:
            mask[:] = 0.0

        return mask

    def _build_move_mask(self, env: Environment) -> np.ndarray:
        """构建移动合法掩码 (N,)，N = 棋盘格数。"""
        NP = self._num_positions()
        mask = np.zeros(NP, dtype=np.float32)
        piece = env.current_piece
        if piece is None:
            return mask

        friendly_occupied = set()
        for p in env.action_queue:
            if p.is_alive and p.team == piece.team and p.id != piece.id:
                friendly_occupied.add(self._pos_idx(p.position.x, p.position.y))

        moves = get_legal_moves(env)
        for m in moves:
            idx = self._pos_idx(m.x, m.y)
            if idx not in friendly_occupied:
                mask[idx] = 1.0

        if piece.position is not None:
            idx = self._pos_idx(piece.position.x, piece.position.y)
            mask[idx] = 1.0

        return mask

    def _build_spell_mask(self, env: Environment) -> np.ndarray:
        """构建法术合法掩码 (4*N,)，N = 棋盘格数。按 4×N 编码。"""
        NP = self._num_positions()
        mask = np.zeros(4 * NP, dtype=np.float32)
        piece = env.current_piece
        if piece is None:
            return mask

        friendly_indices = self._get_friendly_positions(env)
        enemy_indices = set()
        for enemy in self._get_enemy_pieces(env):
            enemy_indices.add(self._pos_idx(enemy.position.x, enemy.position.y))

        spells = env.get_available_spells(piece)
        for spell in spells:
            s_idx = max(0, min(spell.id - 1, 3))

            is_hostile = spell.effect_type is not None and str(spell.effect_type) in (
                "SpellEffectType.DAMAGE", "DAMAGE", "DEBUFF"
            )

            if spell.is_area_effect:
                for tx in range(max(0, piece.position.x - int(spell.range)),
                                min(env.board.width, piece.position.x + int(spell.range) + 1)):
                    for ty in range(max(0, piece.position.y - int(spell.range)),
                                    min(env.board.height, piece.position.y + int(spell.range) + 1)):
                        if abs(piece.position.x - tx) + abs(piece.position.y - ty) <= spell.range:
                            idx = s_idx * NP + self._pos_idx(tx, ty)
                            mask[idx] = 1.0
            else:
                targets = env.get_spell_targets(spell, piece)
                for t in targets:
                    tidx = self._pos_idx(t.position.x, t.position.y)
                    idx = s_idx * NP + tidx
                    if is_hostile and tidx in friendly_indices:
                        continue
                    if not is_hostile and tidx in enemy_indices:
                        continue
                    mask[idx] = 1.0

        return mask

    @staticmethod
    def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """对 logits 应用掩码后做 softmax。mask=0 的位置设为 -1e9，确保 softmax 后概率 ≈ 0。"""
        # 使用足够大的负数确保 exp 后为 0
        MASK_NEG = -1e9
        masked = np.where(mask > 0.5, logits, MASK_NEG)
        return MCTS._softmax(masked)

    @staticmethod
    def _make_move_action(env: Environment, target: Optional[Point]) -> ActionSet:
        a = ActionSet()
        a.move = False
        a.attack = False
        a.spell = False
        if target is not None and env.current_piece is not None and env.current_piece.get_action_points() > 0:
            a.move = True
            a.move_target = target
        return a

    @staticmethod
    def _make_attack_action(env: Environment, target: Optional[Piece]) -> ActionSet:
        a = ActionSet()
        a.move = False
        a.attack = False
        a.spell = False
        if target is not None and env.current_piece is not None and env.current_piece.get_action_points() > 0:
            a.attack = True
            ctx = AttackContext()
            ctx.attacker = env.current_piece
            ctx.target = target
            a.attack_context = ctx
        return a

    @staticmethod
    def _make_spell_action(env: Environment, spell, target) -> ActionSet:
        a = ActionSet()
        a.move = False
        a.attack = False
        a.spell = False
        if (
            spell is not None
            and env.current_piece is not None
            and env.current_piece.get_action_points() > 0
            and env.current_piece.spell_slots > 0
        ):
            a.spell = True
            ctx = SpellContext()
            ctx.caster = env.current_piece
            ctx.spell = spell
            ctx.target = target
            ctx.target_area = (
                Area(env.current_piece.position.x, env.current_piece.position.y, spell.area_radius)
                if spell.is_area_effect
                else Area(target.position.x, target.position.y, 0)
                if target is not None
                else None
            )
            a.spell_context = ctx
        return a

    # ------------------------------------------------------------------
    #  candidate generation + priors
    # ------------------------------------------------------------------

    def _move_candidates(self, env: Environment):
        piece = env.current_piece
        if piece is None:
            return [None]
        moves = get_legal_moves(env)
        cur = piece.position if piece is not None else None
        candidates = [None]  # skip
        if cur is not None:
            candidates.append(cur)  # stay in place
        for m in moves:
            if cur is None or m.x != cur.x or m.y != cur.y:
                candidates.append(m)
        return candidates

    def _attack_candidates(self, env: Environment):
        if env.current_piece is None:
            return [None]
        return [None] + get_attackable_targets(env)

    def _spell_candidates(self, env: Environment):
        piece = env.current_piece
        if piece is None:
            return [None]
        spells = env.get_available_spells(piece)
        result: List = [None]
        for spell in spells:
            if spell.is_area_effect:
                result.append((spell, None))
            else:
                for t in env.get_spell_targets(spell, piece):
                    result.append((spell, t))
        return result

    @staticmethod
    def _action_key(stage: int, candidate):
        if stage == 0:
            if candidate is None:
                return ("move", "skip")
            return ("move", candidate.x, candidate.y)
        if stage == 1:
            if candidate is None:
                return ("attack", "skip")
            return ("attack", candidate.id)
        if stage == 2:
            if candidate is None:
                return ("spell", "skip")
            spell, target = candidate
            tx = target.position.x if target is not None else -1
            ty = target.position.y if target is not None else -1
            return ("spell", spell.id, tx, ty)
        return ("noop",)

    @staticmethod
    def _make_partial_action(env: Environment, stage: int, candidate) -> ActionSet:
        if stage == 0:
            return MCTS._make_move_action(env, candidate)
        if stage == 1:
            return MCTS._make_attack_action(env, candidate)
        if stage == 2:
            if candidate is None:
                return MCTS._make_spell_action(env, None, None)
            return MCTS._make_spell_action(env, candidate[0], candidate[1])
        return ActionSet()

    def _build_priors(self, env: Environment, stage: int, output: dict) -> Dict[Tuple, float]:
        """Return normalised prior dict over candidate action-keys.

        ★ 修复：先分离 skip/execute 概率，execute 概率再按 action head 分配给各候选动作。
        避免 skip 因"概率集中在 1 个 key"而在归一化后压倒执行动作。
        """
        p_skip = max(float(output["switch"][0]), 1e-4)
        p_exec = max(float(output["switch"][1]), 1e-4)
        # ★ 不再缩小 skip 先验，而是通过正确的概率分解来解决
        EPS = 1e-6

        if stage == 0:
            candidates = self._move_candidates(env)
            move_p = output["move"]

            # ★ 先收集执行候选的先验（按 action head 输出的比例分配 execute 概率）
            ranked = []
            for c in candidates:
                if c is None:
                    continue
                ranked.append((c, float(move_p[self._idx_of(c)])))
            ranked.sort(key=lambda x: -x[1])
            top = {c for c, _ in ranked[: self.top_k_move]}
            cur = env.current_piece.position if env.current_piece is not None else None
            stay_key = (cur.x, cur.y) if cur is not None else None

            # 收集执行候选的原始分数
            exec_scores = {}
            for c in candidates:
                if c is None:
                    continue
                if c in top or (c.x == stay_key[0] and c.y == stay_key[1] if stay_key else False):
                    exec_scores[c] = max(EPS, float(move_p[self._idx_of(c)]))

            # ★ 按 action head 分数分配 execute 概率
            exec_total = sum(exec_scores.values())
            priors: Dict[Tuple, float] = {}
            for c, score in exec_scores.items():
                key = self._action_key(stage, c)
                priors[key] = p_exec * (score / exec_total)
            # skip 的 prior = switch 头的 skip 概率
            priors[("move", "skip")] = p_skip

            return self._normalize_priors(priors)

        if stage == 1:
            candidates = self._attack_candidates(env)
            attack_p = output["attack"]
            # 收集执行候选分数
            exec_scores = {}
            for c in candidates:
                if c is None:
                    continue
                exec_scores[c] = max(EPS, float(attack_p[self._idx_of(c.position)]))
            exec_total = sum(exec_scores.values())
            priors = {}
            for c, score in exec_scores.items():
                key = self._action_key(stage, c)
                priors[key] = p_exec * (score / exec_total)
            priors[("attack", "skip")] = p_skip
            return self._normalize_priors(priors)

        if stage == 2:
            candidates = self._spell_candidates(env)
            spell_p = output["spell"]
            exec_scores = {}
            for c in candidates:
                if c is None:
                    continue
                spell, target = c
                s_idx = max(0, min(spell.id - 1, spell_p.shape[0] - 1))
                if target is not None:
                    exec_scores[c] = max(EPS, float(spell_p[s_idx, self._idx_of(target.position)]))
                else:
                    cp = env.current_piece.position if env.current_piece is not None else Point(0, 0)
                    exec_scores[c] = max(EPS, float(spell_p[s_idx, self._idx_of(cp)]))
            exec_total = sum(exec_scores.values())
            priors = {}
            for c, score in exec_scores.items():
                key = self._action_key(stage, c)
                priors[key] = p_exec * (score / exec_total)
            priors[("spell", "skip")] = p_skip
            return self._normalize_priors(priors)

        return {}

    def _add_dirichlet_noise(self, node: "_Node") -> None:
        """在根节点先验上添加 Dirichlet 噪声以鼓励探索（AlphaZero 风格）。"""
        if not node.prior or self.dirichlet_frac <= 0:
            return
        keys = list(node.prior.keys())
        n = len(keys)
        if n <= 1:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        frac = self.dirichlet_frac
        for i, key in enumerate(keys):
            node.prior[key] = (1.0 - frac) * node.prior[key] + frac * noise[i]

    # ------------------------------------------------------------------
    #  turn advance  (called after all three sub-actions are done)
    # ------------------------------------------------------------------

    @staticmethod
    def _advance_turn(env: Environment):
        """Rotate the action queue and prepare the next piece's turn.

        Does NOT execute any action – the turn's move/attack/spell have
        already been applied via the staged partial actions.
        ★ 全部使用 Python list，杜绝 np.array(dtype=object) 导致的内存访问违规。
        """
        env.round_number += 1

        # ★ 过滤死亡棋子，使用 Python list
        alive = [p for p in env.action_queue if p.is_alive]
        if not alive:
            env.current_piece = None
            env.is_game_over = True
            return
        env.action_queue = alive  # 直接使用 list，不转为 np.array(dtype=object)

        for piece in env.action_queue:
            if piece.is_alive:
                piece.set_action_points(piece.max_action_points)

        # ★ 处理延时法术（list pop 替代 np.delete）
        ds_list = env.delayed_spells if isinstance(env.delayed_spells, list) else list(env.delayed_spells)
        for i in range(len(ds_list) - 1, -1, -1):
            spell = ds_list[i]
            spell.spell_lifespan -= 1
            if spell.spell_lifespan == 0:
                env.execute_spell(spell)
                ds_list.pop(i)
            elif spell.spell_lifespan < 0:
                ds_list.pop(i)
        env.delayed_spells = ds_list  # 保持为 list

        # ★ 防御 current_piece 为 None
        if env.current_piece is None:
            env.current_piece = env.action_queue[0]

        # ★ 旋转队列（list 切片 + 拼接替代 np.append）
        aq = env.action_queue if isinstance(env.action_queue, list) else list(env.action_queue)
        if len(aq) > 0:
            aq = aq[1:] + [env.current_piece]
            env.action_queue = aq  # 保持为 list
            env.current_piece = aq[0]
        else:
            env.current_piece = None

        env.is_game_over = (
            not any(p.is_alive for p in env.player1.pieces)
            or not any(p.is_alive for p in env.player2.pieces)
        )

        # ★ 死亡棋子追踪：使用 list
        env.last_round_dead_pieces = list(env.new_dead_this_round) if hasattr(env.new_dead_this_round, '__iter__') else []
        env.new_dead_this_round = []

    # ------------------------------------------------------------------
    #  tree operations
    # ------------------------------------------------------------------

    def _expand(self, node: "_Node"):
        if node.is_expanded or node.is_terminal():
            return

        output = self._infer(node.env, node.stage)
        node.prior = self._build_priors(node.env, node.stage, output)
        candidates = [c for c in self._all_candidates(node.env, node.stage)
                      if self._action_key(node.stage, c) in node.prior]

        for c in candidates:
            key = self._action_key(node.stage, c)
            partial = self._make_partial_action(node.env, node.stage, c)
            child_env = fork_environment(node.env)
            child_env.execute_player_action(partial)

            if node.stage == 2:
                # After the spell sub-action, advance the turn so the next
                # piece becomes current.
                self._advance_turn(child_env)

            child_stage = 0 if node.stage == 2 else node.stage + 1
            child_team = (
                child_env.current_piece.team
                if child_env.current_piece is not None
                else node.team
            )
            child = MCTS._Node(
                child_env,
                stage=child_stage,
                team=child_team,
                depth=node.depth + 1,
                parent=node,
                partial_action=partial,
            )
            node.children[key] = child

        node.is_expanded = True

    def _all_candidates(self, env: Environment, stage: int) -> List:
        if stage == 0:
            return self._move_candidates(env)
        if stage == 1:
            return self._attack_candidates(env)
        return self._spell_candidates(env)

    def _select_child(self, node: "_Node") -> "_Node":
        total_sqrt = math.sqrt(sum(c.visits for c in node.children.values()) + 1)
        best_score = -float("inf")
        best = None
        for key, child in node.children.items():
            q = child.value
            p = node.prior.get(key, 0.0)
            u = self.c_puct * p * total_sqrt / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        return best

    def _evaluate(self, node: "_Node") -> float:
        if node.is_terminal():
            team1 = any(p.is_alive for p in node.env.player1.pieces)
            team2 = any(p.is_alive for p in node.env.player2.pieces)
            if team1 and not team2:
                return 1.0 if node.team == 1 else -1.0
            if team2 and not team1:
                return 1.0 if node.team == 2 else -1.0
            return 0.0
        return float(self._infer(node.env, node.stage)["value"])

    def _backup(self, leaf: "_Node", value: float):
        node = leaf
        while node is not None:
            node.visits += 1
            node.value_sum += value
            if node.parent is None:
                break
            # Flip only when team changes (stage-0 → parent stage-2 may
            # cross to opponent).
            if node.team != node.parent.team:
                value = -value
            node = node.parent

    # ------------------------------------------------------------------
    #  principal variation → full ActionSet
    # ------------------------------------------------------------------

    def _collect_full_action(self, root: "_Node", temperature: float = 0.0) -> ActionSet:
        """Trace through stages 0→1→2 and merge into one ActionSet.

        Args:
            root: Root node after MCTS search.
            temperature: 0.0 = argmax (pick most-visited child),
                1.0 = sample proportional to visit counts.
        """
        full = ActionSet()
        full.move = False
        full.attack = False
        full.spell = False

        node = root
        while node is not None and node.stage < 3:
            if not node.children:
                break
            best_child = self._sample_child(node, temperature)
            pa = best_child.partial_action
            if pa is not None:
                if getattr(pa, "move", False):
                    full.move = True
                    full.move_target = pa.move_target
                if getattr(pa, "attack", False):
                    full.attack = True
                    full.attack_context = pa.attack_context
                if getattr(pa, "spell", False):
                    full.spell = True
                    full.spell_context = pa.spell_context
            node = best_child
            # After stage 2, the child is stage 0 (next piece) – stop
            if node.stage == 0 and node is not root:
                break

        return full

    @staticmethod
    def _sample_child(node: "_Node", temperature: float) -> "_Node":
        """Sample a child node based on visit counts and temperature.

        temperature=0 → argmax (most visited).
        temperature>0 → sample ∝ visits^(1/temperature).
        """
        if temperature <= 0.0 or len(node.children) <= 1:
            return max(node.children.values(), key=lambda c: c.visits)

        children_list = list(node.children.values())
        visits = np.array([c.visits for c in children_list], dtype=np.float64)
        visits = np.maximum(visits, 1e-8)  # avoid zeros

        if temperature < 1e-6:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            probs = visits ** (1.0 / temperature)
            probs /= probs.sum()

        idx = np.random.choice(len(children_list), p=probs)
        return children_list[idx]

    # ------------------------------------------------------------------
    #  batched expansion (uses pre-computed output)
    # ------------------------------------------------------------------

    def _expand_with_output(self, node: "_Node", output: dict):
        """Like _expand but uses externally computed model output."""
        if node.is_expanded or node.is_terminal():
            return
        node.prior = self._build_priors(node.env, node.stage, output)
        candidates = [c for c in self._all_candidates(node.env, node.stage)
                      if self._action_key(node.stage, c) in node.prior]

        for c in candidates:
            key = self._action_key(node.stage, c)
            partial = self._make_partial_action(node.env, node.stage, c)
            child_env = fork_environment(node.env)
            child_env.execute_player_action(partial)

            if node.stage == 2:
                self._advance_turn(child_env)

            child_stage = 0 if node.stage == 2 else node.stage + 1
            child_team = (
                child_env.current_piece.team
                if child_env.current_piece is not None
                else node.team
            )
            child = MCTS._Node(
                child_env,
                stage=child_stage,
                team=child_team,
                depth=node.depth + 1,
                parent=node,
                partial_action=partial,
            )
            node.children[key] = child

        node.is_expanded = True

    # ------------------------------------------------------------------
    #  batched search: N environments in lockstep, one batch per simulation
    # ------------------------------------------------------------------

    def select_actions_batch(self, envs: List[Environment]) -> List[ActionSet]:
        """Run MCTS for multiple environments in parallel, batching GPU inference.

        All N trees advance through simulations together.  Each simulation
        round: (1) select leaves independently, (2) batch-infer expand nodes,
        (3) batch-infer leaf values, (4) backup independently.
        """
        N = len(envs)
        if N == 0:
            return []
        if N == 1:
            return [self.select_action(envs[0])]

        # --- init roots ---
        roots = []
        for env in envs:
            piece = env.current_piece
            if piece is None or not piece.is_alive:
                roots.append(None)
            else:
                r = MCTS._Node(fork_environment(env), stage=0, team=piece.team, depth=0)
                roots.append(r)

        # --- batch-expand all roots ---
        expand_items = [(r.env, r.stage) for r in roots if r is not None]
        expand_outputs = self._infer_batch(expand_items)
        idx = 0
        for r in roots:
            if r is not None:
                self._expand_with_output(r, expand_outputs[idx])
                idx += 1

        # --- simulation loop ---
        active = [i for i in range(N) if roots[i] is not None and roots[i].children]

        for _ in range(self.simulations):
            if not active:
                break
            leaves: List[Tuple[int, "_Node"]] = []

            # Phase 1: each active tree selects down to a leaf
            for i in active:
                node = roots[i]
                while node.is_expanded and node.children and not node.is_terminal():
                    node = self._select_child(node)
                leaves.append((i, node))

            # Phase 2: collect nodes needing expansion
            expand_idx = []        # which leaf index
            expand_nodes = []      # nodes to expand
            expand_map = {}        # leaf_idx -> node_index in batch request

            for ei, (_, leaf) in enumerate(leaves):
                if not leaf.is_expanded and not leaf.is_terminal() and leaf.depth < self.max_depth:
                    expand_idx.append(ei)
                    expand_nodes.append(leaf)

            if expand_nodes:
                batch_items = [(n.env, n.stage) for n in expand_nodes]
                batch_outputs = self._infer_batch(batch_items)
                for ei, node, output in zip(expand_idx, expand_nodes, batch_outputs):
                    self._expand_with_output(node, output)
                    if node.children:
                        # replace leaf with random child
                        leaves[ei] = (leaves[ei][0],
                                     random.choice(list(node.children.values())))
                    else:
                        # expansion produced no children → node stays as-is
                        pass

            # Phase 3: batch-evaluate all leaf nodes
            eval_items = []
            eval_indices = []
            for i, leaf in leaves:
                if leaf.is_terminal():
                    # evaluate inline – no model needed
                    pass
                else:
                    eval_items.append((leaf.env, leaf.stage))
                    eval_indices.append(i)

            term_values = {}
            for i, leaf in leaves:
                if leaf.is_terminal():
                    team1 = any(p.is_alive for p in leaf.env.player1.pieces)
                    team2 = any(p.is_alive for p in leaf.env.player2.pieces)
                    if team1 and not team2:
                        term_values[i] = 1.0 if leaf.team == 1 else -1.0
                    elif team2 and not team1:
                        term_values[i] = 1.0 if leaf.team == 2 else -1.0
                    else:
                        term_values[i] = 0.0

            eval_outputs = {}
            if eval_items:
                outputs = self._infer_batch(eval_items)
                for i, out in zip(eval_indices, outputs):
                    eval_outputs[i] = float(out["value"])

            # Phase 4: backup all leaves
            for i, leaf in leaves:
                if i in term_values:
                    value = term_values[i]
                elif i in eval_outputs:
                    value = eval_outputs[i]
                else:
                    value = 0.0
                self._backup(leaf, value)

            # Prune inactive trees (terminal or no children)
            active = [i for i in active
                      if roots[i] is not None and roots[i].children
                      and not roots[i].is_terminal()]

        # --- collect results ---
        results = []
        for i, r in enumerate(roots):
            if r is None or not r.children:
                results.append(ActionSet())
            else:
                action = self._collect_full_action(r)
                # ★ 重映射 fork 环境的棋子引用到对应的原始 env
                action = self._remap_action_targets(action, envs[i])
                results.append(action)
        return results

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def select_action(
        self,
        env: Environment,
        temperature: float = 0.0,
    ) -> ActionSet:
        """Run MCTS and return the best action.

        Args:
            env: Current game environment.
            temperature: 0.0 = argmax (deterministic), 1.0 = sample proportional
                to visit counts.  Higher values encourage exploration.

        Returns:
            ActionSet with the chosen move/attack/spell sub-actions.
            Also stores visit distributions in self._last_visit_dists for
            later retrieval via get_visit_distributions().
        """
        piece = env.current_piece
        if piece is None or not piece.is_alive:
            self._last_visit_dists = None
            return ActionSet()

        root = MCTS._Node(fork_environment(env), stage=0, team=piece.team, depth=0)
        self._expand(root)
        self._add_dirichlet_noise(root)

        if not root.children:
            self._last_visit_dists = None
            return ActionSet()

        for _ in range(self.simulations):
            node = root

            # select
            while node.is_expanded and node.children and not node.is_terminal():
                node = self._select_child(node)

            # expand
            if not node.is_expanded and not node.is_terminal() and node.depth < self.max_depth:
                self._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))

            # evaluate & backup
            value = self._evaluate(node)
            self._backup(node, value)

        # ★ 记录访问分布（用于训练）
        self._last_visit_dists = self._extract_visit_distributions(root)

        # 收集完整动作（支持温度采样）
        full_action = self._collect_full_action(root, temperature)
        # ★ 关键修复：将 fork 环境的棋子引用重映射到真实环境
        full_action = self._remap_action_targets(full_action, env)
        # ★ 内存回收：搜索完成后清理树引用
        self._clear_node_recursive(root)
        return full_action

    # ------------------------------------------------------------------
    #  remap fork-environment piece references to real environment
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_action_targets(action: ActionSet, env: Environment) -> ActionSet:
        """将 ActionSet 中的棋子引用从 fork 环境重映射到真实环境，并做友军火力安全检查。"""
        id_to_piece = {}
        for p in env.action_queue:
            id_to_piece[p.id] = p
        attacker_ref = env.current_piece
        my_team = attacker_ref.team if attacker_ref is not None else -1

        # ── 重映射 attack_context ──
        if hasattr(action, 'attack_context') and action.attack_context is not None:
            ctx = action.attack_context

            # 重映射 attacker
            if ctx.attacker is not None and ctx.attacker.id in id_to_piece:
                ctx.attacker = id_to_piece[ctx.attacker.id]
            elif attacker_ref is not None:
                ctx.attacker = attacker_ref

            # 重映射 target + 安全检查
            if ctx.target is not None and ctx.target.id in id_to_piece:
                ctx.target = id_to_piece[ctx.target.id]

                # ★ 硬安全检查：绝不允许攻击友方或自己
                if ctx.target.team == my_team or ctx.target.id == ctx.attacker.id:
                    action.attack = False
                    action.attack_context = None
            elif ctx.target is not None:
                # ID 不在真实环境中（棋子已死）→ 禁用攻击
                action.attack = False
                action.attack_context = None

        # ── 重映射 spell_context ──
        if hasattr(action, 'spell_context') and action.spell_context is not None:
            ctx = action.spell_context
            if ctx.caster is not None and ctx.caster.id in id_to_piece:
                ctx.caster = id_to_piece[ctx.caster.id]
            elif attacker_ref is not None:
                ctx.caster = attacker_ref
            if ctx.target is not None and ctx.target.id in id_to_piece:
                ctx.target = id_to_piece[ctx.target.id]

        return action

    # ------------------------------------------------------------------
    #  visit distribution extraction (for training targets)
    # ------------------------------------------------------------------

    def _extract_visit_distributions(self, root: "_Node") -> dict:
        """从根节点提取各阶段的访问计数分布。

        Returns:
            dict with keys:
            - 'move_probs': shape (N,) 归一化访问分布（N=棋盘格数）
            - 'attack_probs': shape (N,) 同上
            - 'spell_probs': shape (4*N,) 归一化访问分布（4种法术×N位置）
            - 'move_keys': list of (stage, candidate_key) for each move child
            - 'attack_keys': list
            - 'spell_keys': list
        """
        result = {
            "move_probs": None,
            "attack_probs": None,
            "spell_probs": None,
        }

        w = self.processor.width
        NP = w * self.processor.height  # num positions

        # stage 0: move
        if root.stage == 0 and root.children:
            move_visits = np.zeros(NP, dtype=np.float32)
            total_v = 0
            for key, child in root.children.items():
                if key[0] == "move" and key[1] != "skip":
                    idx = key[1] * w + key[2]
                    move_visits[idx] = float(child.visits)
                    total_v += child.visits
            if total_v > 0:
                result["move_probs"] = move_visits / total_v

            # stage 1: attack (traverse best move child)
            best_move = max(root.children.values(), key=lambda c: c.visits)
            if best_move.children:
                attack_visits = np.zeros(NP, dtype=np.float32)
                total_v = 0
                for key, child in best_move.children.items():
                    if key[0] == "attack" and key[1] != "skip":
                        if child.partial_action is not None and child.partial_action.attack_context is not None:
                            tgt = child.partial_action.attack_context.target
                            if tgt is not None:
                                idx = tgt.position.x * w + tgt.position.y
                                attack_visits[idx] = float(child.visits)
                                total_v += child.visits
                if total_v > 0:
                    result["attack_probs"] = attack_visits / total_v

                # stage 2: spell (traverse best attack child)
                best_attack = max(best_move.children.values(), key=lambda c: c.visits)
                if best_attack.children:
                    spell_visits = np.zeros(4 * NP, dtype=np.float32)
                    total_v = 0
                    for key, child in best_attack.children.items():
                        if key[0] == "spell" and key[1] != "skip":
                            spell_id = int(key[1])
                            sidx = max(0, min(spell_id - 1, 3))
                            tx = int(key[2]) if key[2] >= 0 else 0
                            ty = int(key[3]) if key[3] >= 0 else 0
                            idx = sidx * NP + tx * w + ty
                            spell_visits[idx] = float(child.visits)
                            total_v += child.visits
                    if total_v > 0:
                        result["spell_probs"] = spell_visits / total_v

        return result

    def get_visit_distributions(self) -> dict:
        """返回最近一次 select_action 调用产生的访问分布。

        Returns:
            dict 或 None（若尚未搜索）。
            包含 'move_probs', 'attack_probs', 'spell_probs'（均为 np.ndarray 或 None）。
        """
        return self._last_visit_dists

    @staticmethod
    def _clear_node_recursive(node: "_Node") -> None:
        """迭代清理节点及其子节点，释放内存。（避免递归栈溢出导致 segfault）"""
        if node is None:
            return
        stack = [node]
        while stack:
            cur = stack.pop()
            # 将子节点加入栈中处理
            for child in list(cur.children.values()):
                stack.append(child)
            cur.children.clear()
            cur.parent = None
            cur.partial_action = None
            cur.prior.clear()
            if hasattr(cur, 'env'):
                cur.env = None


class PersistentMCTS:
    """持久化 MCTS 包装器 —— 一场比赛维护一棵树。

    设计意图：
    - 第一次调用时建立完整的 MCTS 树。
    - 执行动作后，不重建树，而是沿树走到对应子节点，将其作为新根。
    - 只在树中没有对应子节点时才重新搜索。
    """

    def __init__(
        self,
        model,
        processor,
        device,
        simulations: int = 160,
        c_puct: float = 1.0,
        max_depth: int = 60,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        skip_prior_scale: float = 0.3,
    ):
        self.mcts = MCTS(
            model=model,
            processor=processor,
            device=device,
            simulations=simulations,
            c_puct=c_puct,
            max_depth=max_depth,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
            skip_prior_scale=skip_prior_scale,
        )
        self._root: Optional[MCTS._Node] = None
        self._last_env = None
        self._simulations = simulations
        self._last_executed_action = None  # ★ 上次执行的动作，用于树导航

    def select_action(self, env: Environment, temperature: float = 0.0) -> ActionSet:
        """获取当前环境下的动作。优先尝试复用已有树。

        Args:
            env: 当前游戏环境。
            temperature: 温度参数（0=确定性, 1=按访问比例采样）。

        Returns:
            ActionSet。同时可通过 get_visit_distributions() 获取访问分布。
        """
        import gc

        piece = env.current_piece
        if piece is None or not piece.is_alive:
            self._root = None
            self.mcts._last_visit_dists = None
            return ActionSet()

        # 尝试从已有树中导航到匹配当前 env 的节点
        reused = False
        if self._root is not None and self._root.children:
            reused = self._try_navigate_by_action(env)

        if not reused:
            # 无法复用，重新搜索
            self._cleanup_root()
            self._root = MCTS._Node(fork_environment(env), stage=0, team=piece.team, depth=0)
            self.mcts._expand(self._root)
            self.mcts._add_dirichlet_noise(self._root)
            if not self._root.children:
                self._root = None
                self.mcts._last_visit_dists = None
                return ActionSet()
            self._run_simulations(self._root)
            gc.collect()

        if self._root is None or not self._root.children:
            self.mcts._last_visit_dists = None
            return ActionSet()

        # ★ 记录访问分布
        self.mcts._last_visit_dists = self.mcts._extract_visit_distributions(self._root)

        full_action = self.mcts._collect_full_action(self._root, temperature)
        # ★ 关键修复：将 fork 环境的棋子引用重映射到真实环境
        full_action = self.mcts._remap_action_targets(full_action, env)
        # ★ 存储实际执行的动作，供后续导航使用
        self._last_executed_action = full_action
        self._last_env = env
        return full_action

    def get_visit_distributions(self) -> dict:
        """返回最近一次搜索的访问分布。"""
        return self.mcts._last_visit_dists

    def _try_navigate_by_action(self, env: Environment) -> bool:
        """根据上次实际执行的动作在树中导航，而非 max-visit 路径。

        沿 stage 0→1→2 查找与上次执行动作匹配的子节点，找到后提升 stage-0 孙子为新根。
        """
        import gc

        if self._last_executed_action is None:
            return False

        last_action = self._last_executed_action
        node = self._root
        if node is None or not node.children:
            return False

        # 逐阶段匹配子节点
        for stage in range(3):
            if not node.children:
                return False

            matched = None

            if stage == 0 and getattr(last_action, "move", False):
                target = last_action.move_target
                for key, child in node.children.items():
                    if key[0] == "move" and key[1] != "skip":
                        if key[1] == target.x and key[2] == target.y:
                            matched = child
                            break
                # 如果 move 是 skip，匹配 skip 子节点
                if matched is None:
                    for key, child in node.children.items():
                        if key == ("move", "skip"):
                            matched = child
                            break
            elif stage == 0:
                # 没有 move，匹配 skip
                for key, child in node.children.items():
                    if key == ("move", "skip"):
                        matched = child
                        break

            elif stage == 1 and getattr(last_action, "attack", False):
                tgt_id = last_action.attack_context.target.id if last_action.attack_context and last_action.attack_context.target else None
                for key, child in node.children.items():
                    if key[0] == "attack" and key[1] != "skip":
                        if key[1] == tgt_id:
                            matched = child
                            break
                if matched is None:
                    for key, child in node.children.items():
                        if key == ("attack", "skip"):
                            matched = child
                            break
            elif stage == 1:
                for key, child in node.children.items():
                    if key == ("attack", "skip"):
                        matched = child
                        break

            elif stage == 2 and getattr(last_action, "spell", False):
                ctx = last_action.spell_context
                spell_id = ctx.spell.id if ctx and ctx.spell else -1
                tx = ctx.target.position.x if ctx and ctx.target else -1
                ty = ctx.target.position.y if ctx and ctx.target else -1
                for key, child in node.children.items():
                    if key[0] == "spell" and key[1] != "skip":
                        if key[1] == spell_id and key[2] == tx and key[3] == ty:
                            matched = child
                            break
                if matched is None:
                    for key, child in node.children.items():
                        if key == ("spell", "skip"):
                            matched = child
                            break
            elif stage == 2:
                for key, child in node.children.items():
                    if key == ("spell", "skip"):
                        matched = child
                        break

            if matched is None:
                return False

            # 清理兄弟节点
            for key, child in list(node.children.items()):
                if child is not matched:
                    self.mcts._clear_node_recursive(child)
            node.children.clear()
            node = matched

        # node 现在是 stage-0（下一棋子）的节点
        node.parent = None
        self._root = node

        # ★ 关键修复：旧根的 env 是上一回合 fork 的副本，里面的棋子可能已死亡。
        #    用当前真实环境重新 fork，替换旧 env。这确保了 remap 时 ID 映射正确。
        self._root.env = fork_environment(env)

        # 清空旧子节点（它们引用过期 fork 棋子），重新展开
        for old_child in list(self._root.children.values()):
            self.mcts._clear_node_recursive(old_child)
        self._root.children.clear()
        self._root.is_expanded = False

        self.mcts._expand(self._root)
        if self._root.children and not self._root.is_terminal():
            self._run_simulations(self._root)
        gc.collect()
        return True

    def _run_simulations(self, root: "MCTS._Node") -> None:
        """在给定根节点上运行 MCTS 模拟。"""
        import gc

        for sim_i in range(self._simulations):
            node = root
            while node.is_expanded and node.children and not node.is_terminal():
                node = self.mcts._select_child(node)
            if not node.is_expanded and not node.is_terminal() and node.depth < self.mcts.max_depth:
                self.mcts._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))
            value = self.mcts._evaluate(node)
            self.mcts._backup(node, value)

            # 定期触发 GC 防止内存积累
            if sim_i % 50 == 49:
                gc.collect()

        gc.collect()

    def _cleanup_root(self) -> None:
        """清理旧树根节点。"""
        import gc
        if self._root is not None:
            self.mcts._clear_node_recursive(self._root)
            self._root = None
        gc.collect()

    def reset(self) -> None:
        """重置树（新游戏开始时调用）。"""
        self._cleanup_root()
        self._last_env = None
        self._last_executed_action = None
        self.mcts._last_visit_dists = None
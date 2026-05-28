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

    @staticmethod
    def _idx_of(pos) -> int:
        return pos.x * 20 + pos.y

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
        """Run the model and return numpy dict of probs + scalar value."""
        stage_value = 0.3 if stage == 0 else 0.6 if stage == 1 else 1.0
        state = self.processor.build_input(env, stage_value)
        with torch.no_grad():
            x = torch.tensor(state[None, ...], dtype=torch.float32, device=self.device)
            outputs = self.model(x)

        switch = self._softmax(outputs["switch_logits"].cpu().numpy()[0])
        move = self._softmax(outputs["move_logits"].cpu().numpy()[0])
        attack = self._softmax(outputs["attack_logits"].cpu().numpy()[0])
        raw_spell = outputs["spell_logits"].cpu().numpy()[0]
        spell = self._softmax(raw_spell.reshape(-1)).reshape(raw_spell.shape)
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
            spell = self._softmax(spell_all[i].reshape(-1)).reshape(spell_all[i].shape)
            results.append({
                "switch": self._softmax(switch_all[i]),
                "move": self._softmax(move_all[i]),
                "attack": self._softmax(attack_all[i]),
                "spell": spell,
                "value": float(value_all[i]),
            })
        return results

    # ------------------------------------------------------------------
    #  partial-action factories  (single sub-action only)
    # ------------------------------------------------------------------

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
        
        ★ 修复：skip 先验乘以 skip_prior_scale 降低权重，鼓励探索实际动作。
        """
        p_skip = max(float(output["switch"][0]), 1e-4) * self.skip_prior_scale
        p_exec = max(float(output["switch"][1]), 1e-4)
        EPS = 1e-6

        if stage == 0:
            candidates = self._move_candidates(env)
            move_p = output["move"]

            ranked = []
            for c in candidates:
                if c is None:
                    continue
                ranked.append((c, float(move_p[self._idx_of(c)])))
            ranked.sort(key=lambda x: -x[1])
            top = {c for c, _ in ranked[: self.top_k_move]}
            cur = env.current_piece.position if env.current_piece is not None else None
            stay_key = (cur.x, cur.y) if cur is not None else None

            priors: Dict[Tuple, float] = {}
            for c in candidates:
                key = self._action_key(stage, c)
                if c is None:
                    priors[key] = p_skip
                elif c in top or (c.x == stay_key[0] and c.y == stay_key[1] if stay_key else False):
                    priors[key] = max(EPS, p_exec * float(move_p[self._idx_of(c)]))
                # else: dropped by top-k

            return self._normalize_priors(priors)

        if stage == 1:
            candidates = self._attack_candidates(env)
            attack_p = output["attack"]
            priors = {}
            for c in candidates:
                key = self._action_key(stage, c)
                if c is None:
                    priors[key] = p_skip
                else:
                    priors[key] = max(EPS, p_exec * float(attack_p[self._idx_of(c.position)]))
            return self._normalize_priors(priors)

        if stage == 2:
            candidates = self._spell_candidates(env)
            spell_p = output["spell"]
            priors = {}
            for c in candidates:
                key = self._action_key(stage, c)
                if c is None:
                    priors[key] = p_skip
                else:
                    spell, target = c
                    s_idx = max(0, min(spell.id - 1, spell_p.shape[0] - 1))
                    if target is not None:
                        priors[key] = max(EPS, p_exec * float(spell_p[s_idx, self._idx_of(target.position)]))
                    else:
                        cp = env.current_piece.position if env.current_piece is not None else Point(0, 0)
                        priors[key] = max(EPS, p_exec * float(spell_p[s_idx, self._idx_of(cp)]))
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

    def _collect_full_action(self, root: "_Node") -> ActionSet:
        """Trace max-visit path through stages 0→1→2 and merge into one ActionSet."""
        full = ActionSet()
        full.move = False
        full.attack = False
        full.spell = False

        node = root
        while node is not None and node.stage < 3:
            if not node.children:
                break
            best_child = max(node.children.values(), key=lambda c: c.visits)
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
                results.append(self._collect_full_action(r))
        return results

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def select_action(self, env: Environment) -> ActionSet:
        piece = env.current_piece
        if piece is None or not piece.is_alive:
            return ActionSet()

        root = MCTS._Node(fork_environment(env), stage=0, team=piece.team, depth=0)
        self._expand(root)
        # ★ 在根节点添加 Dirichlet 噪声以鼓励探索
        self._add_dirichlet_noise(root)

        if not root.children:
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

        # 收集完整动作
        full_action = self._collect_full_action(root)
        # ★ 内存回收：搜索完成后清理树引用
        self._clear_node_recursive(root)
        return full_action

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

    def select_action(self, env: Environment) -> ActionSet:
        """获取当前环境下的动作。优先尝试复用已有树。"""
        import gc

        piece = env.current_piece
        if piece is None or not piece.is_alive:
            self._root = None
            return ActionSet()

        # 尝试从已有树中导航到匹配当前 env 的节点
        reused = False
        if self._root is not None and self._root.children:
            reused = self._try_navigate_to_child(env)

        if not reused:
            # 无法复用，重新搜索
            self._cleanup_root()
            self._root = MCTS._Node(fork_environment(env), stage=0, team=piece.team, depth=0)
            self.mcts._expand(self._root)
            if not self._root.children:
                self._root = None
                return ActionSet()
            self._run_simulations(self._root)
            gc.collect()

        if self._root is None or not self._root.children:
            return ActionSet()

        full_action = self.mcts._collect_full_action(self._root)
        self._last_env = env
        return full_action

    def _try_navigate_to_child(self, env: Environment) -> bool:
        """沿最大访问量路径追踪到下一棋子的 stage-0 节点，将其提升为新根。"""
        import gc

        node = self._root
        if node is None or not node.children:
            return False

        # 沿最大访问量路径追踪：stage 0 → 1 → 2 → 下一棋子的 stage 0
        trace_path = [node]
        current = node
        max_trace = 20  # ★ 防止死循环
        while current.children and len(trace_path) < max_trace:
            best = max(current.children.values(), key=lambda c: c.visits)
            trace_path.append(best)
            current = best
            # 到达下一棋子的 stage 0 时停止
            if current.stage == 0 and current is not node:
                break

        if len(trace_path) < 2:
            return False

        # 最终节点（下一棋子的 stage 0）作为新根
        new_root = trace_path[-1]
        if new_root is node:
            return False  # 没有前进

        # 清理所有不在路径上的子树
        for i, path_node in enumerate(trace_path):
            for key, child in list(path_node.children.items()):
                if i + 1 < len(trace_path) and child is trace_path[i + 1]:
                    continue  # 保留路径上的子节点
                self.mcts._clear_node_recursive(child)
            path_node.children.clear()

        # 解除新根的父引用
        new_root.parent = None
        self._root = new_root

        # 在新根上继续搜索
        if not self._root.is_expanded:
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
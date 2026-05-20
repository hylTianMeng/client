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
    ):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.top_k_move = top_k_move

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
        """Return normalised prior dict over candidate action-keys."""
        p_skip = max(float(output["switch"][0]), 1e-4)
        p_exec = max(float(output["switch"][1]), 1e-4)
        EPS = 1e-6

        if stage == 0:
            candidates = self._move_candidates(env)
            move_p = output["move"]

            # rank moves by model prob, keep top_k_move + skip + stay
            ranked = []
            for c in candidates:
                if c is None:
                    continue
                ranked.append((c, float(move_p[self._idx_of(c)])))
            ranked.sort(key=lambda x: -x[1])
            top = {c for c, _ in ranked[: self.top_k_move]}
            # always keep skip and stay-in-place
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
                        # area-effect with no target — use caster position
                        cp = env.current_piece.position if env.current_piece is not None else Point(0, 0)
                        priors[key] = max(EPS, p_exec * float(spell_p[s_idx, self._idx_of(cp)]))
            return self._normalize_priors(priors)

        return {}

    # ------------------------------------------------------------------
    #  turn advance  (called after all three sub-actions are done)
    # ------------------------------------------------------------------

    @staticmethod
    def _advance_turn(env: Environment):
        """Rotate the action queue and prepare the next piece's turn.

        Does NOT execute any action – the turn's move/attack/spell have
        already been applied via the staged partial actions.
        """
        env.round_number += 1

        for piece in env.action_queue:
            if piece.is_alive:
                piece.set_action_points(piece.max_action_points)

        for i in range(len(env.delayed_spells) - 1, -1, -1):
            spell = env.delayed_spells[i]
            spell.spell_lifespan -= 1
            if spell.spell_lifespan == 0:
                env.execute_spell(spell)
                env.delayed_spells = np.delete(env.delayed_spells, i)
            elif spell.spell_lifespan < 0:
                env.delayed_spells = np.delete(env.delayed_spells, i)

        if len(env.action_queue) > 0:
            env.action_queue = np.append(env.action_queue[1:], [env.current_piece])
            env.current_piece = env.action_queue[0]
        else:
            env.current_piece = None

        env.is_game_over = (
            not any(p.is_alive for p in env.player1.pieces)
            or not any(p.is_alive for p in env.player2.pieces)
        )

        env.last_round_dead_pieces = np.array(env.new_dead_this_round, dtype=object)
        env.new_dead_this_round = np.array([], dtype=object)

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

        return self._collect_full_action(root)
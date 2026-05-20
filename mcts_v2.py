import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import Environment, Point, ActionSet, SpellContext, AttackContext, Area
from state_processor import StateProcessor
from strategy_utils import fork_environment, get_legal_moves, get_attackable_targets, step_with_action
from utils import SpellEffectType


class MCTS:
    """True cross-turn MCTS with PUCT selection and neural-network priors.

    Unlike the original PUCTMCTS which decomposed a single piece's turn into
    3 fixed-depth stages, this tree spans the full action queue.  Each edge
    is a complete ActionSet (move + attack + spell for one piece); after the
    action the queue rotates and the next piece – possibly the opponent's –
    becomes current.  Perspective flips only when the acting team changes.
    """

    def __init__(
        self,
        model,
        processor: StateProcessor,
        device: torch.device,
        simulations: int = 160,
        c_puct: float = 1.0,
        max_depth: int = 200,
        top_k_move: int = 16,
        top_k_composite: int = 8,
        max_children: int = 120,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.top_k_move = top_k_move
        self.top_k_composite = top_k_composite
        self.max_children = max_children

    # ------------------------------------------------------------------
    #  Node
    # ------------------------------------------------------------------

    class _Node:
        __slots__ = (
            "env",
            "team",
            "depth",
            "parent",
            "action",
            "children",
            "visits",
            "value_sum",
            "prior",
            "is_expanded",
        )

        def __init__(
            self,
            env: Environment,
            team: int,
            depth: int = 0,
            parent=None,
            action: Optional[ActionSet] = None,
        ):
            self.env = env
            self.team = team          # team of the piece that will act at this node
            self.depth = depth
            self.parent = parent
            self.action = action      # action that led from parent to this node
            self.children: Dict[object, "MCTS._Node"] = {}
            self.visits = 0
            self.value_sum = 0.0
            self.prior: Dict[object, float] = {}
            self.is_expanded = False

        @property
        def value(self) -> float:
            if self.visits == 0:
                return 0.0
            return self.value_sum / self.visits

        def is_terminal(self) -> bool:
            return self.env.is_game_over

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        denom = np.sum(exp)
        if denom <= 0:
            return np.ones_like(logits) / logits.size
        return exp / denom

    @staticmethod
    def _idx_of(point: Point) -> int:
        return point.x * 20 + point.y

    # ------------------------------------------------------------------
    #  Model wrapper
    # ------------------------------------------------------------------

    def _infer(self, env: Environment) -> dict:
        """Run the model and return numpy dict of {switch, move, attack, spell, value}."""
        state = self.processor.build_input(env, stage_value=0.3)
        with torch.no_grad():
            x = torch.tensor(state[None, ...], dtype=torch.float32, device=self.device)
            outputs = self.model(x)

        switch = self._softmax(outputs["switch_logits"].cpu().numpy()[0])
        move = self._softmax(outputs["move_logits"].cpu().numpy()[0])
        att = self._softmax(outputs["attack_logits"].cpu().numpy()[0])
        raw_spell = outputs["spell_logits"].cpu().numpy()[0]  # (num_spell_types, 400)
        spell = self._softmax(raw_spell.reshape(-1)).reshape(raw_spell.shape)
        value = float(outputs["value"].cpu().numpy()[0])

        return {"switch": switch, "move": move, "attack": att, "spell": spell, "value": value}

    # ------------------------------------------------------------------
    #  Action generation
    # ------------------------------------------------------------------

    def _generate_actions(
        self, env: Environment, priors: dict
    ) -> Dict[object, Tuple[ActionSet, float]]:
        """Return {action_key: (ActionSet, unnormalised_prior)} for the current piece."""
        piece = env.current_piece
        if piece is None or not piece.is_alive:
            return {}

        ap = max(0, piece.action_points)
        legal_moves = get_legal_moves(env)
        attackable = get_attackable_targets(env)
        spells = env.get_available_spells(piece) if piece.spell_slots > 0 else []

        switch = priors["switch"]
        move_p = priors["move"]
        att_p = priors["attack"]
        spell_p = priors["spell"]

        p_skip = max(1e-4, float(switch[0]))
        p_exec = max(1e-4, float(switch[1]))
        EPS = 1e-6

        actions: Dict[object, Tuple[ActionSet, float]] = {}

        # -- 1.  pass ------------------------------------------------
        pass_action = ActionSet()
        pass_action.move = False
        pass_action.attack = False
        pass_action.spell = False
        actions[("pass",)] = (pass_action, p_skip)

        if ap < 1:
            return actions

        # -- 2.  move-only  (top-k) ----------------------------------
        ranked_moves: List[Tuple[Point, float]] = []
        for m in legal_moves:
            ranked_moves.append((m, float(move_p[self._idx_of(m)])))
        ranked_moves.sort(key=lambda x: -x[1])
        top_moves = ranked_moves[: self.top_k_move]

        for m, prob in top_moves:
            a = ActionSet()
            a.move = True
            a.move_target = m
            a.attack = False
            a.spell = False
            actions[("move", m.x, m.y)] = (a, max(EPS, p_exec * prob))

        # -- 3.  attack-only -----------------------------------------
        for t in attackable:
            prob = float(att_p[self._idx_of(t.position)])
            a = ActionSet()
            a.move = False
            a.attack = True
            ctx = AttackContext()
            ctx.attacker = piece
            ctx.target = t
            a.attack_context = ctx
            a.spell = False
            actions[("attack", t.id)] = (a, max(EPS, p_exec * prob))

        # -- 4.  spell-only ------------------------------------------
        spell_candidates: List[Tuple[object, object, float]] = []  # (spell, target, prob)
        for spell in spells:
            for t in env.get_spell_targets(spell, piece):
                s_idx = min(spell.id - 1, spell_p.shape[0] - 1)
                prob = float(spell_p[s_idx, self._idx_of(t.position)])
                spell_candidates.append((spell, t, prob))
        for spell, t, prob in spell_candidates:
            a = ActionSet()
            a.move = False
            a.attack = False
            a.spell = True
            ctx = SpellContext()
            ctx.caster = piece
            ctx.spell = spell
            ctx.target = t
            ctx.target_area = (
                Area(piece.position.x, piece.position.y, spell.area_radius)
                if spell.is_area_effect
                else Area(t.position.x, t.position.y, 0)
            )
            a.spell_context = ctx
            actions[("spell", spell.id, t.position.x, t.position.y)] = (a, max(EPS, p_exec * prob))

        if ap < 2:
            return self._truncate(actions)

        # -- 5.  move + attack  (top composite × attackable) ----------
        top_comp = top_moves[: self.top_k_composite]
        for m, m_prob in top_comp:
            for t in attackable:
                if abs(m.x - t.position.x) + abs(m.y - t.position.y) > piece.attack_range:
                    continue
                prob = max(EPS, p_exec * m_prob * float(att_p[self._idx_of(t.position)]))
                a = self._build_composite_move_attack(piece, m, t)
                if a is not None:
                    actions[("ma", m.x, m.y, t.id)] = (a, prob)

        # -- 6.  move + spell -----------------------------------------
        for m, m_prob in top_comp:
            for spell, t, sp_prob in spell_candidates:
                if abs(m.x - t.position.x) + abs(m.y - t.position.y) > spell.range:
                    continue
                prob = max(EPS, p_exec * m_prob * sp_prob)
                a = self._build_composite_move_spell(piece, m, spell, t)
                if a is not None:
                    actions[("ms", m.x, m.y, spell.id, t.position.x, t.position.y)] = (a, prob)

        if ap < 3:
            return self._truncate(actions)

        # -- 7.  move + attack + spell  (only top-4 moves) ------------
        top4 = top_moves[:4]
        for m, m_prob in top4:
            for t in attackable:
                if abs(m.x - t.position.x) + abs(m.y - t.position.y) > piece.attack_range:
                    continue
                for spell, st, sp_prob in spell_candidates:
                    if abs(m.x - st.position.x) + abs(m.y - st.position.y) > spell.range:
                        continue
                    prob = max(EPS, p_exec * m_prob * float(att_p[self._idx_of(t.position)]) * sp_prob)
                    a = self._build_composite_move_attack_spell(piece, m, t, spell, st)
                    if a is not None:
                        actions[("mas", m.x, m.y, t.id, spell.id, st.position.x, st.position.y)] = (a, prob)

        return self._truncate(actions)

    def _truncate(self, actions: Dict[object, Tuple[ActionSet, float]]) -> Dict[object, Tuple[ActionSet, float]]:
        if len(actions) <= self.max_children:
            return actions
        # Keep pass, then top by prior
        pass_key = ("pass",)
        pass_entry = actions.pop(pass_key, None)
        ranked = sorted(actions.items(), key=lambda kv: -kv[1][1])
        result = dict(ranked[: self.max_children - 1])
        if pass_entry is not None:
            result[pass_key] = pass_entry
        return result

    # -- composite-action builders -----------------------------------

    @staticmethod
    def _build_composite_move_attack(piece, move: Point, target) -> Optional[ActionSet]:
        a = ActionSet()
        a.move = True
        a.move_target = move
        a.attack = True
        ctx = AttackContext()
        ctx.attacker = piece
        ctx.target = target
        a.attack_context = ctx
        a.spell = False
        return a

    @staticmethod
    def _build_composite_move_spell(piece, move: Point, spell, target) -> Optional[ActionSet]:
        a = ActionSet()
        a.move = True
        a.move_target = move
        a.attack = False
        a.spell = True
        ctx = SpellContext()
        ctx.caster = piece
        ctx.spell = spell
        ctx.target = target
        ctx.target_area = (
            Area(move.x, move.y, spell.area_radius)
            if spell.is_area_effect
            else Area(target.position.x, target.position.y, 0)
        )
        a.spell_context = ctx
        return a

    @staticmethod
    def _build_composite_move_attack_spell(piece, move: Point, target, spell, st) -> Optional[ActionSet]:
        a = ActionSet()
        a.move = True
        a.move_target = move
        a.attack = True
        a.attack_context = AttackContext()
        a.attack_context.attacker = piece
        a.attack_context.target = target
        a.spell = True
        ctx = SpellContext()
        ctx.caster = piece
        ctx.spell = spell
        ctx.target = st
        ctx.target_area = (
            Area(move.x, move.y, spell.area_radius)
            if spell.is_area_effect
            else Area(st.position.x, st.position.y, 0)
        )
        a.spell_context = ctx
        return a

    # ------------------------------------------------------------------
    #  Tree operations
    # ------------------------------------------------------------------

    def _expand(self, node: "_Node") -> None:
        """Compute priors via model and populate children lazily."""
        if node.is_expanded or node.is_terminal():
            return
        priors = self._infer(node.env)
        candidates = self._generate_actions(node.env, priors)
        for key, (action, prior_value) in candidates.items():
            node.prior[key] = prior_value
            child_env = fork_environment(node.env)
            step_with_action(child_env, action)
            child_team = (
                child_env.current_piece.team
                if child_env.current_piece is not None
                else node.team
            )
            node.children[key] = MCTS._Node(
                child_env, child_team, depth=node.depth + 1, parent=node, action=action
            )
        node.is_expanded = True

    def _select_child(self, node: "_Node") -> "_Node":
        total_visits_sqrt = math.sqrt(sum(c.visits for c in node.children.values()) + 1)
        best_score = -float("inf")
        best = None
        for key, child in node.children.items():
            q = child.value
            p = node.prior.get(key, 0.0)
            u = self.c_puct * p * total_visits_sqrt / (1 + child.visits)
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
        return float(self._infer(node.env)["value"])

    def _backup(self, leaf: "_Node", value: float) -> None:
        node = leaf
        while node is not None:
            node.visits += 1
            node.value_sum += value
            if node.parent is None:
                break
            # Flip value only when perspective changes
            if node.team != node.parent.team:
                value = -value
            node = node.parent

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def select_action(self, env: Environment) -> ActionSet:
        piece = env.current_piece
        if piece is None or not piece.is_alive:
            return ActionSet()

        root = MCTS._Node(fork_environment(env), piece.team, depth=0)
        self._expand(root)

        if not root.children:
            return ActionSet()

        for _ in range(self.simulations):
            node = root

            # Select
            while node.is_expanded and node.children and not node.is_terminal():
                node = self._select_child(node)

            # Expand
            if not node.is_expanded and not node.is_terminal() and node.depth < self.max_depth:
                self._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))

            # Evaluate & backup
            value = self._evaluate(node)
            self._backup(node, value)

        best = max(root.children.values(), key=lambda c: c.visits)
        return best.action if best.action is not None else ActionSet()
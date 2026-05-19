import math
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from env import Environment, Point, ActionSet, SpellContext, AttackContext, Area
from state_processor import StateProcessor
from strategy_utils import fork_environment, get_legal_moves, get_attackable_targets


class PUCTMCTS:
    def __init__(
        self,
        model,
        processor: StateProcessor,
        device: torch.device,
        simulations: int = 16,
        c_puct: float = 1.0,
    ):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

    class Node:
        def __init__(self, env: Environment, stage: int, parent=None, action: Optional[ActionSet] = None):
            self.env = env
            self.stage = stage
            self.parent = parent
            self.action = action
            self.children: Dict[Tuple, "PUCTMCTS.Node"] = {}
            self.visits = 0
            self.value_sum = 0.0
            self.prior: Dict[Tuple, float] = {}
            self.expanded = False

        @property
        def value(self) -> float:
            return self.value_sum / self.visits if self.visits > 0 else 0.0

        def is_leaf(self) -> bool:
            return self.stage >= 3 or self.env.is_game_over

    @staticmethod
    def _to_index(x: int, y: int) -> int:
        return x * 20 + y

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        denom = np.sum(exp)
        if denom <= 0:
            return np.ones_like(logits) / logits.size
        return exp / denom

    def _build_model_output(self, env: Environment, stage_value: float):
        state = self.processor.build_input(env, stage_value)
        with torch.no_grad():
            x = torch.tensor(state[None, ...], dtype=torch.float32, device=self.device)
            outputs = self.model(x)
        switch_logits = outputs["switch_logits"].detach().cpu().numpy()[0]
        move_logits = outputs["move_logits"].detach().cpu().numpy()[0]
        attack_logits = outputs["attack_logits"].detach().cpu().numpy()[0]
        spell_logits = outputs["spell_logits"].detach().cpu().numpy()[0]
        value = outputs["value"].detach().cpu().numpy()[0]

        return {
            "switch": self._softmax(switch_logits),
            "move": self._softmax(move_logits),
            "attack": self._softmax(attack_logits),
            "spell": self._softmax(spell_logits.reshape(-1)).reshape(spell_logits.shape),
            "value": float(value),
        }

    def _make_move_action(self, env: Environment, move: Optional[Point]) -> ActionSet:
        action = ActionSet()
        if move is not None and env.current_piece.get_action_points() > 0:
            action.move = True
            action.move_target = move
        else:
            action.move = False
        action.attack = False
        action.spell = False
        return action

    def _make_attack_action(self, env: Environment, target: Optional[object]) -> ActionSet:
        action = ActionSet()
        action.move = False
        if target is not None and env.current_piece.get_action_points() > 0:
            action.attack = True
            action.attack_context = AttackContext()
            action.attack_context.attacker = env.current_piece
            action.attack_context.target = target
        else:
            action.attack = False
        action.spell = False
        return action

    def _make_spell_action(
        self,
        env: Environment,
        spell_choice: Optional[Tuple[object, Optional[Point]]],
    ) -> ActionSet:
        action = ActionSet()
        action.move = False
        action.attack = False
        if spell_choice is not None and env.current_piece.get_action_points() > 0 and env.current_piece.spell_slots > 0:
            spell, target_point = spell_choice
            action.spell = True
            action.spell_context = SpellContext()
            action.spell_context.caster = env.current_piece
            action.spell_context.spell = spell

            if spell.is_area_effect:
                action.spell_context.target = None
                action.spell_context.target_area = Area(
                    env.current_piece.position.x,
                    env.current_piece.position.y,
                    spell.area_radius,
                )
            else:
                if target_point is None:
                    action.spell = False
                else:
                    action.spell_context.target = next(
                        (
                            p
                            for p in env.action_queue
                            if p.position.x == target_point.x
                            and p.position.y == target_point.y
                            and p.is_alive
                        ),
                        None,
                    )
                    action.spell_context.target_area = Area(target_point.x, target_point.y, 0)
                    if action.spell_context.target is None:
                        action.spell = False
        else:
            action.spell = False
        return action

    def _candidate_actions(self, env: Environment, stage: int):
        if stage == 0:
            moves = get_legal_moves(env)
            current_pos = env.current_piece.position if env.current_piece is not None else None
            candidates = [None]
            if current_pos is not None:
                candidates.append(current_pos)
            for move in moves:
                if current_pos is None or move.x != current_pos.x or move.y != current_pos.y:
                    candidates.append(move)
            return candidates

        if stage == 1:
            attackable = get_attackable_targets(env)
            return [None] + attackable

        if stage == 2:
            spells = env.get_available_spells(env.current_piece)
            candidates = [None]
            for spell in spells:
                if spell.is_area_effect:
                    candidates.append((spell, env.current_piece.position))
                else:
                    targets = env.get_spell_targets(spell, env.current_piece)
                    for target in targets:
                        candidates.append((spell, Point(target.position.x, target.position.y)))
            return candidates

        return [None]

    def _action_key(self, stage: int, candidate):
        if stage == 0:
            if candidate is None:
                return ("move", None)
            return ("move", candidate.x, candidate.y)
        if stage == 1:
            if candidate is None:
                return ("attack", None)
            return ("attack", candidate.id)
        if stage == 2:
            if candidate is None:
                return ("spell", None)
            spell, point = candidate
            return ("spell", spell.id, point.x if point is not None else None, point.y if point is not None else None)
        return ("noop",)

    def _make_action_for_stage(self, env: Environment, stage: int, candidate):
        if stage == 0:
            return self._make_move_action(env, candidate)
        if stage == 1:
            return self._make_attack_action(env, candidate)
        if stage == 2:
            return self._make_spell_action(env, candidate)
        return ActionSet()

    def _execute_partial_action(self, env: Environment, stage: int, candidate):
        action = self._make_action_for_stage(env, stage, candidate)
        env.execute_player_action(action)
        return action

    def _build_priors(self, env: Environment, stage: int):
        stage_value = 0.3 if stage == 0 else 0.6 if stage == 1 else 1.0
        output = self._build_model_output(env, stage_value)
        candidates = self._candidate_actions(env, stage)
        priors: Dict[Tuple, float] = {}

        if stage == 0:
            move_probs = output["move"]
            skip_prob = float(output["switch"][0])
            exec_prob = float(output["switch"][1])
            for candidate in candidates:
                key = self._action_key(stage, candidate)
                if candidate is None:
                    priors[key] = max(1e-4, skip_prob)
                else:
                    idx = self._to_index(candidate.x, candidate.y)
                    priors[key] = max(1e-6, exec_prob * float(move_probs[idx]))
            return priors

        if stage == 1:
            attack_probs = output["attack"]
            skip_prob = float(output["switch"][0])
            exec_prob = float(output["switch"][1])
            for candidate in candidates:
                key = self._action_key(stage, candidate)
                if candidate is None:
                    priors[key] = max(1e-4, skip_prob)
                else:
                    idx = self._to_index(candidate.position.x, candidate.position.y)
                    priors[key] = max(1e-6, exec_prob * float(attack_probs[idx]))
            return priors

        if stage == 2:
            spell_probs = output["spell"]
            skip_prob = float(output["switch"][0])
            exec_prob = float(output["switch"][1])
            for candidate in candidates:
                key = self._action_key(stage, candidate)
                if candidate is None:
                    priors[key] = max(1e-4, skip_prob)
                else:
                    spell, point = candidate
                    idx = self._to_index(point.x, point.y)
                    spell_index = max(0, min(spell.id - 1, spell_probs.shape[0] - 1))
                    priors[key] = max(1e-6, exec_prob * float(spell_probs[spell_index, idx]))
            return priors

        return priors

    def _create_child(self, parent: "PUCTMCTS.Node", candidate):
        new_env = fork_environment(parent.env)
        self._execute_partial_action(new_env, parent.stage, candidate)
        child_stage = parent.stage + 1
        action = self._make_action_for_stage(parent.env, parent.stage, candidate)
        node = PUCTMCTS.Node(new_env, child_stage, parent=parent, action=action)
        return node

    def _select_child(self, node: "PUCTMCTS.Node") -> "PUCTMCTS.Node":
        best_score = -float("inf")
        best_child = None
        total_visits = math.sqrt(sum(child.visits for child in node.children.values()) + 1)
        for key, child in node.children.items():
            q_value = child.value
            prior = node.prior.get(key, 0.0)
            u_value = self.c_puct * prior * total_visits / (1 + child.visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child if best_child is not None else random.choice(list(node.children.values()))

    def _expand(self, node: "PUCTMCTS.Node"):
        if node.expanded or node.is_leaf():
            return
        node.prior = self._build_priors(node.env, node.stage)
        candidates = self._candidate_actions(node.env, node.stage)
        for candidate in candidates:
            key = self._action_key(node.stage, candidate)
            if key not in node.children:
                node.children[key] = self._create_child(node, candidate)
        node.expanded = True

    def _evaluate(self, node: "PUCTMCTS.Node") -> float:
        if node.is_leaf():
            if node.env.is_game_over:
                current_team = node.env.current_piece.team if node.env.current_piece is not None else 1
                team1_alive = any(p.is_alive for p in node.env.player1.pieces)
                team2_alive = any(p.is_alive for p in node.env.player2.pieces)
                if team1_alive and not team2_alive:
                    return 1.0 if current_team == 1 else -1.0
                if team2_alive and not team1_alive:
                    return 1.0 if current_team == 2 else -1.0
                return 0.0
            output = self._build_model_output(node.env, 0.3)
            return float(output["value"])
        if not node.expanded:
            self._expand(node)
        return self._evaluate(random.choice(list(node.children.values())))

    def _backup(self, node: "PUCTMCTS.Node", value: float):
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def select_action(self, env: Environment) -> ActionSet:
        root = PUCTMCTS.Node(fork_environment(env), stage=0)
        self._expand(root)

        for _ in range(self.simulations):
            node = root
            while not node.is_leaf() and node.expanded:
                node = self._select_child(node)
            if not node.expanded and not node.is_leaf():
                self._expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))
            value = self._evaluate(node)
            self._backup(node, value)

        if not root.children:
            return ActionSet()
        best_child = max(root.children.values(), key=lambda c: c.visits)
        return best_child.action if best_child.action is not None else ActionSet()

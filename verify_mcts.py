"""Verify that mcts_v3.MCTS produces valid ActionSet outputs.

Usage: python verify_mcts.py
"""

import os
import sys
import torch

# setup_battle_host imports log_converter which isn't in the repo,
# so provide a minimal stub before importing env.
import types
log_converter_stub = types.ModuleType("log_converter")
class _StubLogConverter:
    def init(self, *args, **kwargs): pass
    def add_round(self, *args, **kwargs): pass
    def add_move(self, *args, **kwargs): pass
    def add_attack(self, *args, **kwargs): pass
    def add_spell(self, *args, **kwargs): pass
log_converter_stub.LogConverter = _StubLogConverter
sys.modules["log_converter"] = log_converter_stub

from env import Environment, Point, ActionSet, InitGameMessage
from model import TacticalPolicyNet
from state_processor import StateProcessor
from utils import InitPolicyMessage, PieceArg
from strategy_factory import StrategyFactory


def _build_init_message(env: Environment, player_id: int):
    msg = InitGameMessage()
    msg.piece_cnt = 3
    msg.id = player_id
    msg.board = env.board
    return msg


def _wrap_piece_args(piece_args):
    policy = InitPolicyMessage()
    policy.piece_args = piece_args
    return policy


def main():
    device = torch.device("cpu")
    board_file = os.path.join(os.path.dirname(__file__), "BoardCase", "case1.txt")

    # --- setup environment ---
    env = Environment(local_mode=True, if_log=0)
    env.init_board_only(board_file)

    init_fn = StrategyFactory.get_init_strategy_by_name("archer22")
    for pid in (1, 2):
        args = init_fn(_build_init_message(env, pid))
        env.apply_init_policy(pid, _wrap_piece_args(args))
    env.setup_battle_host()
    env.begin_turn_host()

    # --- setup model ---
    model = TacticalPolicyNet(in_channels=19)
    model.eval()
    processor = StateProcessor()

    print("=== MCTS v3 verification ===")

    # --- test mcts_v3 ---
    from mcts_v3 import MCTS
    mcts_v3 = MCTS(model, processor, device, simulations=8)
    action = mcts_v3.select_action(env)

    assert isinstance(action, ActionSet), f"Expected ActionSet, got {type(action)}"
    print(f"v3 action: move={getattr(action, 'move', False)}, attack={action.attack}, spell={action.spell}")

    if getattr(action, "move", False):
        assert action.move_target is not None, "move=True but move_target is None"
        assert isinstance(action.move_target, Point), f"move_target type: {type(action.move_target)}"
        print(f"  move_target: ({action.move_target.x}, {action.move_target.y})")

    if action.attack:
        assert action.attack_context is not None, "attack=True but attack_context is None"
        assert action.attack_context.target is not None, "attack=True but target is None"
        print(f"  attack_target: id={action.attack_context.target.id}")

    if action.spell:
        assert action.spell_context is not None, "spell=True but spell_context is None"
        assert action.spell_context.spell is not None, "spell=True but spell is None"
        print(f"  spell: {action.spell_context.spell.name}")

    print("v3 OK")

    # --- test mcts v1 (backward compat) ---
    print("\n=== MCTS v1 verification ===")
    from mcts import PUCTMCTS
    mcts_v1 = PUCTMCTS(model, processor, device, simulations=8)
    action_v1 = mcts_v1.select_action(env)

    assert isinstance(action_v1, ActionSet), f"Expected ActionSet, got {type(action_v1)}"
    print(f"v1 action: move={getattr(action_v1, 'move', False)}, attack={action_v1.attack}, spell={action_v1.spell}")
    print("v1 OK")

    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
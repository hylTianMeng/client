# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

THUAI9 ("苍穹棋域") Python AI client — a tactical grid-based 1v1 strategy game. Each player controls 3 pieces on a 20×20 board with varying terrain heights. Pieces have RPG-like stats (strength/dexterity/intelligence), equipment (weapon + armor), action points, spell slots, and can move/attack/cast spells each turn. The game ends when one side loses all pieces.

## Commands

```bash
# Local console PvP (test board/rules)
python local_client.py --mode local --board ./BoardCase/case1.txt

# AI vs AI with predefined strategy
python local_client.py --mode function --strategy aggressive
python local_client.py --mode function --strategy defensive
python local_client.py --mode function --strategy mcts --mcts-simulations 25

# Saiblo competition entry point
python main.py --strategy aggressive
python main.py --strategy mcts --mcts-simulations 25

# Self-play training loop
python model_train.py --device cpu --iterations 5 --games-per-iter 5
```

No build/lint/test tooling is configured (no setup.py, pyproject.toml, or test framework).

## Architecture

### Game engine (`env.py`)
- **`Environment`** — central game controller. Two modes: `local_mode=True` (step-based simulation) and `local_mode=False` (Saiblo host mode with `init_board_only` / `setup_battle_host` / `begin_turn_host` / `apply_action_host` / `end_turn_host`).
- **`Board`** — grid + height_map. Pathfinding uses Dijkstra with Manhattan distance; move cost is `1 + max(0, height_diff)`. Attack range is also Manhattan distance.
- **`Piece`** / **`Player`** — piece has stats, equipment, resources, position. Player owns 3 pieces (`PIECE_CNT = 3`).
- **`SpellFactory`** (in `utils.py`) — 4 built-in spells: Fireball (area damage), Heal (area heal), Arrow Hit (single-target damage), Teleport (self-move). Trap (ID 4) is disabled.
- Turn flow: reset action points → advance delayed spells → current piece acts (move → attack → spell, each checked for remaining AP/resources) → rotate queue → check game over.

### Neural network (`model.py`)
- **`TacticalPolicyNet`** — ~471K params. Input: `(B, 17, 20, 20)`. Architecture: stem conv → 6 residual blocks (64 channels, 3×3 convs) → 4 heads:
  - `switch_head`: binary "execute/skip" per action stage (FC 64→128→2)
  - `move_head`, `attack_head`: spatial value maps via 1×1 conv → flattened to 400 logits
  - `spell_head`: `num_spell_types × 400` logits (spell type × target position)
  - `value_head`: scalar board evaluation (tanh clamped to [-1,1])

### State representation (`state_processor.py`)
- **`StateProcessor.build_input()`** produces a `(17, 20, 20)` normalized float32 tensor:
  - Channel 0: terrain passability, 1: height, 2-3: allied/enemy attack threat maps, 4-5: health, 6-7: physical resist, 8: queue order, 9-10: spell slots, 11-12: spell threat, 13: current piece indicator, 14: stage (0.3/0.6/1.0), 15: max movement, 16: action points.

### PUCT MCTS (`mcts.py`)
- **`PUCTMCTS`** — neural-guided tree search. Uses 3-stage sequential decision: stage 0 (move), stage 1 (attack), stage 2 (spell). Each stage the model outputs priors via softmax over legal actions + a switch probability. Node selection uses PUCT formula `Q + c_puct * P * sqrt(sum_visits) / (1 + visits)`. Leaf evaluation uses the value head or game-over outcome.

### Self-play training (`self_play.py` + `model_train.py` + `dataset_utils.py`)
- Data format: `.npz` files with `states` (N,17,20,20), `switch`, `move_target`, `attack_target`, `spell_target`, `value`, `stage`.
- `collect_self_play_examples()` — runs games with model-controlled player 1 vs opponent, stores 3 samples per turn (move/attack/spell stages).
- `compute_loss()` — weighted sum of cross-entropy (switch + action logits) + MSE (value), masked per stage.
- Training loop: self-play → save data → train for N epochs → repeat.

### Strategy system (`strategy_factory.py`)
- Predefined init strategies: aggressive (20/8/2 + short sword/heavy armor), defensive (5/15/10 + bow/light armor), archer30, archer22, 2archer1mage, 1archer2mage, mage3.
- Action strategies: aggressive (close in + attack), defensive (keep distance + snipe), random, AlphaBeta (depth-limited minimax), MCTS (simple UCB1 with random rollouts), PUCT (neural-guided, in `mcts.py`).
- **To add a custom strategy:** add `get_custom_init_strategy()` and `get_custom_action_strategy()` static methods to `StrategyFactory`, wire them into `local_client.py` or `main.py`.

### Entry points
- **`main.py`** — Saiblo stdin/stdout client. Receives JSON from judger, calls strategy functions, sends JSON back. Handles handshake (seat assignment), init phase (piece placement), and per-turn actions.
- **`local_client.py`** — local testing with console or function-mode AI vs AI.
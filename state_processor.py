import numpy as np
from typing import Tuple

from env import Environment, Piece
from utils import SpellEffectType

class StateProcessor:
    def __init__(self, width: int = 20, height: int = 20):
        """
        width 是地图的宽
        height: the height of board
        """
        self.width = width
        self.height = height
        self._offset_cache = {}

    @staticmethod
    def _to_index(x: int, y: int) -> int:
        """
        Args:
            x, y : the position
        Returns:
            the hard-code x*20+y
        """
        return x * 20 + y

    @staticmethod
    def _attack_potential(piece: Piece) -> float:
        """
        Args:
            piece: Piece
        Returns:
            the attack damage
        """
        if getattr(piece, "weapon_type", 0) == 4:
            return 0.0  # spell damage is handled in spell channels
        return float(piece.physical_damage + piece.strength)

    def _manhattan_offsets(self, radius: int):
        if radius in self._offset_cache:
            return self._offset_cache[radius]

        offsets = []
        for dx in range(-radius, radius + 1):
            max_dy = radius - abs(dx)
            for dy in range(-max_dy, max_dy + 1):
                offsets.append((dx, dy))

        self._offset_cache[radius] = offsets
        return offsets

    def _apply_spell_mask(self, positions, radius: int, board_width: int, board_height: int):
        result = set()
        for x, y in positions:
            for dx, dy in self._manhattan_offsets(radius):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < board_width and 0 <= ny < board_height:
                    result.add((nx, ny))
        return result

    def _collect_spell_targets(self, env: Environment, caster: Piece, spell):
        targets = []
        if spell.is_area_effect:
            for tx in range(max(0, caster.position.x - int(spell.range)), min(env.board.width, caster.position.x + int(spell.range) + 1)):
                for ty in range(max(0, caster.position.y - int(spell.range)), min(env.board.height, caster.position.y + int(spell.range) + 1)):
                    if abs(caster.position.x - tx) + abs(caster.position.y - ty) > spell.range:
                        continue
                    targets.append((tx, ty))
        else:
            # 对于单体/非范围法术（只能对人释放），在 state 设计上仍然按照其 range 来染格：
            # 把施法者周围所有在 range 内的格子都视为 potential targets
            for tx in range(max(0, caster.position.x - int(spell.range)), min(env.board.width, caster.position.x + int(spell.range) + 1)):
                for ty in range(max(0, caster.position.y - int(spell.range)), min(env.board.height, caster.position.y + int(spell.range) + 1)):
                    if abs(caster.position.x - tx) + abs(caster.position.y - ty) > spell.range:
                        continue
                    targets.append((tx, ty))
        return targets

    def build_raw_state(self, env: Environment, stage_value: float = 0.3) -> np.ndarray:
        """
        Args:
            env: Environment
            stage_value: 0.3 0.6 1.0
        Returns:
            numpy arraw, shape 19 * 20 * 20
        """
        state = np.zeros((19, self.width, self.height), dtype=np.float32)

        for x in range(min(env.board.width, self.width)):
            for y in range(min(env.board.height, self.height)):
                cell = env.board.grid[x][y]
                state[0, x, y] = 0.0 if cell.state == -1 else 1.0
                state[1, x, y] = float(env.board.height_map[x][y])

        valid_heights = state[1, :, :][state[0, :, :] == 1]
        if valid_heights.size > 0:
            hmin = valid_heights.min()
            state[1] = np.maximum(state[1] - hmin, 0.0)

        current_piece = env.current_piece
        current_team = current_piece.team
        assert current_team != None
        queue_size = len(env.action_queue)

        for order_index, piece in enumerate(env.action_queue):
            if not piece.is_alive:
                continue
            x = piece.position.x
            y = piece.position.y
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue

            team_offset = 0 if piece.team == current_team else 1
            attack_potential = self._attack_potential(piece)
            if attack_potential > 0.0:
                for tx in range(min(env.board.width, self.width)):
                    for ty in range(min(env.board.height, self.height)):
                        if abs(piece.position.x - tx) + abs(piece.position.y - ty) <= piece.attack_range:
                            state[2 + team_offset, tx, ty] += attack_potential
            else:
                for tx in range(min(env.board.width, self.width)):
                    for ty in range(min(env.board.height, self.height)):
                        if abs(piece.position.x - tx) + abs(piece.position.y - ty) <= piece.attack_range:
                            state[13 + team_offset, tx, ty] += 4

            state[4 + team_offset, x, y] = float(piece.health)
            state[6 + team_offset, x, y] = float(piece.physical_resist)
            state[8, x, y] = float(queue_size - order_index)
            state[9 + team_offset, x, y] = float(piece.spell_slots)

            spells = env.get_available_spells(piece) if piece.spell_slots > 0 and piece.action_points > 0 else []
            if spells:
                # 对于同一施法者：先取每格的最大法术伤害（同人取最大）
                piece_spell_map = np.zeros((self.width, self.height), dtype=np.float32)
                for spell in spells:
                    if spell.effect_type != SpellEffectType.DAMAGE:
                        continue

                    target_positions = self._collect_spell_targets(env, piece, spell)
                    if not target_positions:
                        continue
                    if spell.is_area_effect:
                        damaged_positions = self._apply_spell_mask(target_positions, int(spell.area_radius), env.board.width, env.board.height)
                    elif spell.is_locking_spell:
                        damaged_positions = self._apply_spell_mask(target_positions, 0, env.board.width, env.board.height)
                    for tx, ty in damaged_positions:
                        if 0 <= tx < self.width and 0 <= ty < self.height:
                            piece_spell_map[tx, ty] = max(piece_spell_map[tx, ty], float(spell.base_value))

                # 同队之间累加（不同施法者造成的伤害相加）
                if piece.team == current_team:
                    state[11, :, :] += piece_spell_map
                else:
                    state[12, :, :] += piece_spell_map

            if current_piece is not None and piece == current_piece:
                state[15, x, y] = 1.0
                state[16, :, :] = float(piece.action_points)

            state[17, x, y] = float(piece.max_movement)
        state[18, :, :] = float(stage_value)
        return state

    @staticmethod
    def normalize_state(state: np.ndarray) -> np.ndarray:
        """
        对输入的 state numpy 数组（模型输入）进行归一化
        输出归一化后的结果。当然，归一化是逐通道的。
        """
        if state.ndim != 3 or state.shape[1:] != (20, 20):
            raise ValueError("Expect state shape (C, 20, 20)")

        state = state.astype(np.float32, copy=True)
        if state.shape[0] != 19:
            raise ValueError("Expect exactly 19 channels for the current network configuration")

        state[0] = (state[0] > 0).astype(np.float32)

        height = state[1]
        height = height - height.min()
        state[1] = height / 10.0

        state[2] = state[2] / 150.0
        state[3] = state[3] / 150.0

        state[4] = state[4] / 150.0
        state[5] = state[5] / 150.0

        state[6] = state[6] / 150.0
        state[7] = state[7] / 150.0

        state[8] = state[8] / 6.0

        state[9] = state[9] / 5.0
        state[10] = state[10] / 5.0

        state[11] = state[11] / 150.0
        state[12] = state[12] / 150.0
        state[13] = state[13] / 150.0
        state[14] = state[14] / 150.0

        state[15] = state[15].astype(np.float32)
        state[16] = state[16] / 40.0
        state[17] = state[17] / 3.0

        return state.clip(0.0, 1.0)

    def build_input(self, env: Environment, stage_value: float = 0.3) -> np.ndarray:
        raw = self.build_raw_state(env, stage_value)
        return self.normalize_state(raw)

    @staticmethod
    def print_state_visualization(state: np.ndarray) -> None:
        """
        可视化打印 19 个 20x20 的通道状态
        Args:
            state: numpy array, shape (19, 20, 20)
        """
        if state.shape != (19, 20, 20):
            raise ValueError(f"Expected shape (19, 20, 20), got {state.shape}")
        
        channel_names = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18"
        ]
        
        print("\n" + "=" * 80)
        print("19通道状态数值 (20x20)")
        print("=" * 80)
        
        np.set_printoptions(suppress=True, precision=3, linewidth=200)
        
        for ch in range(19):
            print(f"\n【通道 {ch}】{channel_names[ch]}")
            print(state[ch])
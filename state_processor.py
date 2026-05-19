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
            return 0.0 # fashu damage is in fashu tongdao
        return float(piece.physical_damage + piece.strength)

    def build_raw_state(self, env: Environment, stage_value: float = 0.3) -> np.ndarray:
        """
        Args:
            env: Environment
            stage_value: 0.3 0.6 1.0
        Returns:
            numpy arraw, shape 17 * 20 * 20
        """
        state = np.zeros((17, self.height, self.width), dtype=np.float32)

        for x in range(min(env.board.width, self.width)):
            for y in range(min(env.board.height, self.height)):
                cell = env.board.grid[x][y]
                state[0, x, y] = 0.0 if cell.state == 1 else 0.0
                height_value = int(env.board.height_map[x][y])
                state[1, x, y] = float(height_value)

        valid_heights = state[1, :, :][state[0, :, :] == 1]
        if valid_heights.size > 0:
            hmin = valid_heights.min()
            state[1] = np.maximum(state[1] - hmin, 0.0)
        # tongdao 0,1 no problem

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
            # print("[Process] team_offset = ", team_offset);
            attack_potential = self._attack_potential(piece)
            for tx in range(env.board.width):
                for ty in range(env.board.height):
                    distance = abs(piece.position.x - tx) + abs(piece.position.y - ty)
                    if distance <= piece.attack_range:
                        if piece.team == current_team:
                            state[2, tx, ty] += attack_potential
                        else:
                            state[3, tx, ty] += attack_potential

            state[4 + team_offset * 1, x, y] = float(piece.health)
            state[6 + team_offset * 1, x, y] = float(piece.physical_resist)
            state[8, x, y] = float(queue_size - order_index)
            state[9 + team_offset, x, y] = float(piece.spell_slots)

            spells = env.get_available_spells(piece) if piece.spell_slots > 0 and piece.action_points > 0 else []
            for spell in spells:
                if spell.effect_type != SpellEffectType.DAMAGE:
                    continue

                if spell.is_area_effect:
                    for tx in range(max(0, x - int(spell.range)), min(env.board.width, self.width, x + int(spell.range) + 1)):
                        for ty in range(max(0, y - int(spell.range)), min(env.board.height, self.height, y + int(spell.range) + 1)):
                            if abs(x - tx) + abs(y - ty) > spell.range:
                                continue

                            for ax in range(max(0, tx - int(spell.area_radius)), min(env.board.width, self.width, tx + int(spell.area_radius) + 1)):
                                for ay in range(max(0, ty - int(spell.area_radius)), min(env.board.height, self.height, ty + int(spell.area_radius) + 1)):
                                    if abs(ax - tx) + abs(ay - ty) > spell.area_radius:
                                        continue
                                    state[11, ax, ay] += float(spell.range)
                                    state[12, ax, ay] += float(spell.base_value)
                else:
                    for tx in range(max(0, x - int(spell.range)), min(env.board.width, self.width, x + int(spell.range) + 1)):
                        for ty in range(max(0, y - int(spell.range)), min(env.board.height, self.height, y + int(spell.range) + 1)):
                            if abs(x - tx) + abs(y - ty) > spell.range:
                                continue
                            state[11, tx, ty] += float(spell.range)
                            state[12, tx, ty] += float(spell.base_value)

            if current_piece is not None and piece == current_piece:
                state[13, x, y] = 1.0

            state[15, x, y] = float(piece.max_movement)
            state[16, x, y] = float(piece.action_points)

        state[14, :, :] = float(stage_value)
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
        if state.shape[0] != 17:
            raise ValueError("Expect exactly 17 channels for the current network configuration")

        state[0] = (state[0] > 0).astype(np.float32)

        height = state[1]
        height = height - height.min()
        state[1] = height / 10.0

        state[2] = state[2] / 110.0
        state[3] = state[3] / 110.0

        state[4] = state[4] / 110.0
        state[5] = state[5] / 110.0

        state[6] = state[6] / 23.0
        state[7] = state[7] / 23.0

        state[8] = state[8] / 6.0

        state[9] = state[9] / 5.0
        state[10] = state[10] / 5.0

        state[11] = state[11] / 110.0
        state[12] = state[12] / 110.0

        state[13] = state[13].astype(np.float32)
        state[14] = state[14].astype(np.float32)
        state[15] = state[15] / 40.0
        state[16] = state[16] / 3.0

        return state.clip(0.0, 1.0)
    def build_input(self, env: Environment, stage_value: float = 0.3) -> np.ndarray:
        raw = self.build_raw_state(env, stage_value)
        return self.normalize_state(raw)

    @staticmethod
    def print_state_visualization(state: np.ndarray) -> None:
        """
        可视化打印 17 个 20x20 的通道状态
        Args:
            state: numpy array, shape (17, 20, 20)
        """
        if state.shape != (17, 20, 20):
            raise ValueError(f"Expected shape (17, 20, 20), got {state.shape}")
        
        channel_names = [
            "0-地形(0空/1可走/2占据)",
            "1-高度",
            "2-玩家1血量",
            "3-玩家2血量",
            "4-玩家1物抗",
            "5-玩家2物抗",
            "6-玩家1属性",
            "7-玩家2属性",
            "8-道具",
            "9-玩家1法伤",
            "10-玩家2法伤",
            "11-玩家1魔抗",
            "12-玩家2魔抗",
            "13-单位类型",
            "14-游戏阶段",
            "15-坐标特征",
            "16-行动点数"
        ]
        
        print("\n" + "=" * 80)
        print("17通道状态数值 (20x20)")
        print("=" * 80)
        
        np.set_printoptions(suppress=True, precision=3, linewidth=200)
        
        for ch in range(17):
            print(f"\n【通道 {ch}】{channel_names[ch]}")
            print(state[ch])
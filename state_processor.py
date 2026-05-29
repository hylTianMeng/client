import numpy as np
from typing import Tuple

from env import Environment, Piece
from utils import SpellEffectType

class StateProcessor:
    def __init__(self, width: int = 40, height: int = 40, num_channels: int = 21):
        """
        width 是地图的宽
        height: the height of board
        num_channels: 状态通道数（新规则=21：原19 + 安全区 + 高地优势）
        """
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self._offset_cache = {}

    def _to_index(self, x: int, y: int) -> int:
        """
        Args:
            x, y : the position
        Returns:
            flat index = x * width + y
        """
        return x * self.width + y

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
            numpy array, shape (num_channels, height, width)
            Channels:
              0: walkable mask
              1: height map
              2: friendly attack potential (physical)
              3: enemy attack potential (physical)
              4: friendly HP
              5: enemy HP
              6: friendly physical resist
              7: enemy physical resist
              8: turn order (queue position)
              9: friendly spell slots
             10: enemy spell slots
             11: friendly spell damage map
             12: enemy spell damage map
             13: friendly magic attack potential (staff)
             14: enemy magic attack potential (staff)
             15: current piece indicator
             16: action points
             17: max movement
             18: stage value
             19: ★ zone safety (1.0=safe, decreasing to 0.0)
             20: ★ height advantage (current piece's height advantage per position)
        """
        state = np.zeros((self.num_channels, self.width, self.height), dtype=np.float32)

        bw = min(env.board.width, self.width)
        bh = min(env.board.height, self.height)

        # ── ch0: walkable mask ──
        for x in range(bw):
            for y in range(bh):
                cell = env.board.grid[x][y]
                state[0, x, y] = 0.0 if cell.state == -1 else 1.0
                state[1, x, y] = float(env.board.height_map[x][y])

        # ── ch1: height map (subtract min walkable height) ──
        valid_heights = state[1, :, :][state[0, :, :] == 1]
        if valid_heights.size > 0:
            hmin = valid_heights.min()
            state[1] = np.maximum(state[1] - hmin, 0.0)

        current_piece = env.current_piece
        if current_piece is None:
            current_team = 1
        else:
            current_team = current_piece.team
        assert current_team is not None
        queue_size = len(env.action_queue)

        # ── ch19: zone safety ──
        zone_radius = getattr(env, 'zone_radius', float("inf"))
        if zone_radius == float("inf"):
            state[19, :, :] = 1.0  # no zone, always safe
        else:
            cx = env.board.zone_cx
            cy = env.board.zone_cy
            for x in range(bw):
                for y in range(bh):
                    dist = max(abs(x - cx), abs(y - cy))
                    if dist <= zone_radius:
                        state[19, x, y] = 1.0
                    else:
                        # 圈外：距离越远越危险
                        state[19, x, y] = float(max(0.0, 1.0 - (dist - zone_radius) / 10.0))

        # ── ch20: height advantage ──
        attacker_h = current_piece.height if current_piece is not None else 0
        for x in range(bw):
            for y in range(bh):
                if state[0, x, y] > 0:
                    target_h = env.board.height_map[x][y]
                    dh = max(0, attacker_h - int(target_h))
                    state[20, x, y] = float(dh)

        # ── piece-specific channels ──
        for order_index, piece in enumerate(env.action_queue):
            if not piece.is_alive:
                continue
            x = piece.position.x
            y = piece.position.y
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue

            team_offset = 0 if piece.team == current_team else 1
            attack_potential = self._attack_potential(piece)
            # effective_attack_range includes height bonus
            eff_range = env.effective_attack_range(piece, piece) if hasattr(env, 'effective_attack_range') else piece.attack_range
            if attack_potential > 0.0:
                for tx in range(bw):
                    for ty in range(bh):
                        if abs(piece.position.x - tx) + abs(piece.position.y - ty) <= eff_range:
                            state[2 + team_offset, tx, ty] += attack_potential
            else:
                for tx in range(bw):
                    for ty in range(bh):
                        if abs(piece.position.x - tx) + abs(piece.position.y - ty) <= piece.attack_range:
                            state[13 + team_offset, tx, ty] += 4

            state[4 + team_offset, x, y] = float(piece.health)
            state[6 + team_offset, x, y] = float(piece.physical_resist)
            state[8, x, y] = float(queue_size - order_index)
            state[9 + team_offset, x, y] = float(piece.spell_slots)

            spells = env.get_available_spells(piece) if piece.spell_slots > 0 and piece.action_points > 0 else []
            if spells:
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

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        对输入的 state numpy 数组（模型输入）进行归一化
        输出归一化后的结果。逐通道归一化。
        """
        h, w = self.height, self.width
        if state.ndim != 3 or state.shape[1:] != (h, w):
            raise ValueError(f"Expect state shape (C, {h}, {w}), got {state.shape}")

        state = state.astype(np.float32, copy=True)
        C = state.shape[0]
        if C != self.num_channels:
            raise ValueError(f"Expect exactly {self.num_channels} channels, got {C}")

        state[0] = (state[0] > 0).astype(np.float32)

        height = state[1]
        height = height - height.min()
        state[1] = height / 10.0

        state[2] = state[2] / 150.0
        state[3] = state[3] / 150.0

        # HP 固定 80
        state[4] = state[4] / 80.0
        state[5] = state[5] / 80.0

        # 抗性最大 26（重甲）
        state[6] = state[6] / 26.0
        state[7] = state[7] / 26.0

        state[8] = state[8] / 6.0

        state[9] = state[9] / 5.0
        state[10] = state[10] / 5.0

        state[11] = state[11] / 150.0
        state[12] = state[12] / 150.0
        state[13] = state[13] / 150.0
        state[14] = state[14] / 150.0

        state[15] = state[15].astype(np.float32)
        # 行动点固定 2
        state[16] = state[16] / 2.0
        # 移动力 = dex//2+6+护甲，最大约 25
        state[17] = state[17] / 25.0

        # stage value 已在 [0, 1] 范围
        state[18] = state[18]  # 保持不变

        # ch19: zone safety 已经在 [0, 1] 范围
        state[19] = state[19]

        # ch20: height advantage, 最大高度差约 2~3，除 3 归一化
        state[20] = state[20] / 3.0

        return state.clip(0.0, 1.0)

    def build_input(self, env: Environment, stage_value: float = 0.3) -> np.ndarray:
        raw = self.build_raw_state(env, stage_value)
        return self.normalize_state(raw)

    @staticmethod
    def print_state_visualization(state: np.ndarray) -> None:
        """
        可视化打印各通道状态
        Args:
            state: numpy array, shape (C, H, W)
        """
        C, H, W = state.shape
        print(f"\nState shape: ({C}, {H}, {W})")
        
        channel_names = [
            "0-可行走", "1-高度", "2-友方物攻", "3-敌方物攻",
            "4-友方HP", "5-敌方HP", "6-友方物抗", "7-敌方物抗",
            "8-队列顺序", "9-友方法术位", "10-敌方法术位",
            "11-友方魔法", "12-敌方魔法", "13-友方法杖", "14-敌方法杖",
            "15-当前棋子", "16-行动点", "17-最大移动", "18-阶段值",
            "19-安全区", "20-高地优势"
        ]
        
        print("\n" + "=" * 80)
        print(f"{C}通道状态数值 ({H}x{W})")
        print("=" * 80)
        
        np.set_printoptions(suppress=True, precision=3, linewidth=200)
        
        for ch in range(min(C, len(channel_names))):
            print(f"\n【通道 {ch}】{channel_names[ch]}")
            print(state[ch])
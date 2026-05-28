"""选手端 AI / 搜索用辅助函数（从 Environment 抽离，供策略与搜索使用）。"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from env import Board, Environment, Piece, Cell, Player
from utils import ActionSet, Point, SpellContext, Area


# ---------------------------------------------------------------------------
# ★ 安全的手动拷贝函数 —— 替代 copy.deepcopy，杜绝 0xc0000005 崩溃
#    copy.deepcopy 在 Piece 的复杂对象图上（spell_list、SpellContext 的
#    caster/target 循环引用等）极易触发内存访问违规。
# ---------------------------------------------------------------------------

def _copy_piece(p: Piece) -> Piece:
    """手动深拷贝 Piece，仅拷贝游戏状态属性。Spell 对象不可变，共享引用。"""
    n = Piece()
    n.if_log = 0
    n.health = p.health
    n.max_health = p.max_health
    n.physical_resist = p.physical_resist
    n.magic_resist = p.magic_resist
    n.physical_damage = p.physical_damage
    n.magic_damage = p.magic_damage
    n.action_points = p.action_points
    n.max_action_points = p.max_action_points
    n.spell_slots = p.spell_slots
    n.max_spell_slots = p.max_spell_slots
    n.movement = p.movement
    n.max_movement = p.max_movement
    n.id = p.id
    n.type = p.type
    n.strength = p.strength
    n.dexterity = p.dexterity
    n.intelligence = p.intelligence
    n.position = Point(p.position.x, p.position.y)
    n.height = p.height
    n.attack_range = p.attack_range
    n.spell_list = list(p.spell_list)  # Spell 不可变，浅拷贝即可
    n.death_round = p.death_round
    n.team = p.team
    n.queue_index = p.queue_index
    n.is_alive = p.is_alive
    n.is_in_turn = p.is_in_turn
    n.is_dying = p.is_dying
    n.spell_range = p.spell_range
    n.weapon_type = p.weapon_type
    return n


def _copy_cell(c: Cell) -> Cell:
    """手动拷贝 Cell。"""
    return Cell(state=c.state, player_id=c.player_id, piece_id=c.piece_id)


def _copy_spell_context(sc) -> "SpellContext":
    """手动拷贝 SpellContext —— caster/target 置 None，由调用方重连。"""
    n = SpellContext()
    n.spell = sc.spell  # Spell 不可变，共享引用
    n.spell_power = sc.spell_power
    n.target_type = sc.target_type
    n.target = None   # 稍后重连
    n.caster = None   # 稍后重连
    if sc.target_area is not None:
        n.target_area = Area(sc.target_area.x, sc.target_area.y, sc.target_area.radius)
    n.spell_range = sc.spell_range
    n.effect_type = sc.effect_type
    n.damage_type = sc.damage_type
    n.damage_value = sc.damage_value
    n.heal_value = sc.heal_value
    n.effect_value = sc.effect_value
    n.is_delay_spell = sc.is_delay_spell
    n.base_lifespan = sc.base_lifespan
    n.spell_lifespan = sc.spell_lifespan
    n.is_damage_spell = sc.is_damage_spell
    n.is_area_effect = sc.is_area_effect
    n.is_locking_spell = sc.is_locking_spell
    n.spell_cost = sc.spell_cost
    n.action_cost = sc.action_cost
    n.is_hit = sc.is_hit
    n.is_critical = sc.is_critical
    return n


def _reconnect_spell_contexts(action_queue: list, delayed_spells: list):
    """重连 SpellContext 中的 caster/target 到新拷贝的 Piece 对象。"""
    id_to_piece = {p.id: p for p in action_queue}
    for sc in delayed_spells:
        if sc.caster_orig_id is not None:
            sc.caster = id_to_piece.get(sc.caster_orig_id)
        if sc.target_orig_id is not None:
            sc.target = id_to_piece.get(sc.target_orig_id)


def _disconnect_spell_contexts(delayed_spells: list):
    """断开 SpellContext 中的 caster/target，保存原始 id 供重连。"""
    for sc in delayed_spells:
        sc.caster_orig_id = sc.caster.id if sc.caster is not None else None
        sc.target_orig_id = sc.target.id if sc.target is not None else None
        sc.caster = None
        sc.target = None


def get_state_score(env: Environment) -> float:
    if env.current_piece is None:
        return 0.0

    current_team = env.current_piece.team
    score = 0.0

    for piece in env.action_queue:
        if not piece.is_alive:
            continue

        piece_score = 0.0
        piece_score += piece.health / piece.max_health * 10
        piece_score += piece.height * 0.5
        piece_score += piece.action_points * 2
        piece_score += piece.spell_slots * 1.5
        piece_score += (piece.physical_damage + piece.magic_damage) * 0.3
        piece_score += (piece.physical_resist + piece.magic_resist) * 0.2

        if piece.team == current_team:
            score += piece_score
        else:
            score -= piece_score

    return score


def get_legal_moves(env: Environment, piece: Optional[Piece] = None) -> List[Point]:
    if piece is None:
        piece = env.current_piece

    if piece is None or not piece.is_alive:
        return []

    legal_moves: List[Point] = []
    mask = env.board.valid_target(piece, piece.movement)

    for x in range(env.board.width):
        for y in range(env.board.height):
            if mask[x][y] != -1:
                legal_moves.append(Point(x, y))

    return legal_moves


def get_attackable_targets(env: Environment, piece: Optional[Piece] = None) -> List[Piece]:
    if piece is None:
        piece = env.current_piece

    if piece is None or not piece.is_alive:
        return []

    targets: List[Piece] = []
    for target in env.action_queue:
        if (
            target.is_alive
            and target.team != piece.team
            and env.is_in_attack_range(piece, target)
        ):
            targets.append(target)

    return targets


def simulate_move(env: Environment, piece: Piece, target: Point) -> bool:
    if not piece.is_alive:
        return False

    mask = env.board.valid_target(piece, piece.movement)
    if mask[target.x][target.y] == -1:
        return False

    return True


def simulate_attack(env: Environment, attacker: Piece, target: Piece) -> float:
    if not attacker.is_alive or not target.is_alive:
        return 0.0

    if not env.is_in_attack_range(attacker, target):
        return 0.0

    # 与 env.execute_attack 一致：攻击必定命中。
    # - 法杖（weapon_type==4）：真实伤害固定 4
    # - 其他武器：基础伤害 physical_damage + strength，再由 physical_resist 抵消
    if getattr(attacker, "weapon_type", 0) == 4:
        return 4.0
    raw_damage = attacker.physical_damage + attacker.strength
    return float(max(0, raw_damage - target.physical_resist))


def step_with_action(env: Environment, action: ActionSet) -> None:
    """执行一步完整行动：重置 AP、旋转队列并执行 action。

    修复说明：
    - 旋转队列后正确更新 current_piece，防止下一循环用错误的 current_piece 判断队伍。
    - 增加 current_piece 为 None 的防御性检查和死棋子过滤。
    - ★ 使用 Python list 操作替代 np.append/np.delete，避免 object 数组内存损坏。
    """

    # ★ 防御：过滤已死亡棋子，使用 Python list
    alive_queue = [p for p in env.action_queue if p.is_alive]
    if not alive_queue:
        env.current_piece = None
        env.is_game_over = True
        return
    env.action_queue = alive_queue  # 直接使用 list

    env.round_number += 1

    # 重置所有存活棋子的行动点
    for piece in env.action_queue:
        if piece.is_alive:
            piece.set_action_points(piece.max_action_points)

    # 确定当前应行动的棋子
    env.current_piece = env.action_queue[0]

    # ★ 延时法术处理（list pop 替代 np.delete）
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

    # ★ 旋转队列（list 切片替代 np.append）
    aq = env.action_queue if isinstance(env.action_queue, list) else list(env.action_queue)
    acting_piece = env.current_piece
    aq = aq[1:] + [acting_piece]
    env.action_queue = aq  # 保持为 list

    # 执行行动
    if action and acting_piece is not None and acting_piece.is_alive:
        env.execute_player_action(action)

    # ★ 关键修复：旋转后正确更新 current_piece 为队列新的头部
    env.current_piece = env.action_queue[0] if len(env.action_queue) > 0 else None

    # 检查游戏结束
    env.is_game_over = (
        not any(p.is_alive for p in env.player1.pieces)
        or not any(p.is_alive for p in env.player2.pieces)
    )

    # ★ 死亡棋子追踪：使用 list
    env.last_round_dead_pieces = list(env.new_dead_this_round) if hasattr(env.new_dead_this_round, '__iter__') else []
    env.new_dead_this_round = []


def fork_environment(env: Environment) -> Environment:
    """深拷贝环境。★ 全部手动拷贝，杜绝 copy.deepcopy 导致的 0xc0000005 崩溃。"""
    new_env = Environment(local_mode=(env.mode == 0), if_log=0)

    new_env.mode = env.mode
    new_env.round_number = env.round_number
    new_env.is_game_over = env.is_game_over

    if env.board:
        new_env.board = Board(if_log=new_env.if_log)
        new_env.board.width = env.board.width
        new_env.board.height = env.board.height
        new_env.board.boarder = env.board.boarder
        # ★ 手动拷贝 grid
        new_env.board.grid = [[_copy_cell(cell) for cell in row] for row in env.board.grid]
        new_env.board.height_map = np.copy(env.board.height_map)

    # ★ 手动拷贝 Piece 对象（避免 deepcopy 崩溃）
    new_env.action_queue = [_copy_piece(p) for p in env.action_queue]

    # 建立旧→新 Piece 的 id 映射，用于重连 current_piece
    old_to_new = {old_p.id: new_p for old_p, new_p in zip(env.action_queue, new_env.action_queue)}

    # ★ 重连 current_piece
    if env.current_piece is not None:
        new_env.current_piece = old_to_new.get(env.current_piece.id)

    # ★ 拷贝延时法术——手动拷贝，caster/target 在拷贝中自动置 None
    new_env.delayed_spells = [_copy_spell_context(sc) for sc in env.delayed_spells]
    _reconnect_spell_contexts(new_env.action_queue, new_env.delayed_spells)  # 重连新 env 的引用

    # 死亡追踪
    new_env.new_dead_this_round = [old_to_new[p.id] for p in env.new_dead_this_round if p.id in old_to_new]
    new_env.last_round_dead_pieces = [old_to_new[p.id] for p in env.last_round_dead_pieces if p.id in old_to_new]

    # ★ 玩家分组（从 action_queue 重建，避免 deepcopy Player）
    if new_env.player1 is None:
        new_env.player1 = Player()
    if new_env.player2 is None:
        new_env.player2 = Player()
    new_env.player1.id = 1
    new_env.player2.id = 2
    new_env.player1.pieces = [p for p in new_env.action_queue if p.team == 1]
    new_env.player2.pieces = [p for p in new_env.action_queue if p.team == 2]
    new_env.player1.piece_num = len(new_env.player1.pieces)
    new_env.player2.piece_num = len(new_env.player2.pieces)
    new_env.player1.feature_total = env.player1.feature_total
    new_env.player2.feature_total = env.player2.feature_total

    return new_env

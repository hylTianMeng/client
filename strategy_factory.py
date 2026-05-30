from typing import Callable, List, Tuple, Set
import math
from env import *
from utils import *
from strategy_utils import (
    fork_environment,
    get_attackable_targets,
    get_legal_moves,
    get_state_score,
    step_with_action,
)

# 控制 MCTS 是否输出调试日志，设为 False 可关闭所有 [MCTS] 输出
MCTS_VERBOSE: bool = False


def _init_front_y_range(board: "Board", player_id: int, depth: int = 5) -> range:
    """靠近中线的若干行（按 boarder 相对定位，兼容 case1/case2）。"""
    bdr = board.boarder
    if player_id == 1:
        y_hi = bdr - 1
        y_lo = max(1, bdr - depth)
        return range(y_hi, y_lo - 1, -1)
    y_lo = bdr + 1
    y_hi = min(board.height - 1, bdr + depth)
    return range(y_lo, y_hi + 1)


def _allocate_init_positions(
    board: "Board",
    player_id: int,
    piece_cnt: int,
    preferred_order: List[Tuple[int, int]],
) -> List[Point]:
    """Pick distinct walkable cells on the player's side; prefer scan order then full fallback."""
    occupied: Set[Tuple[int, int]] = set()
    out: List[Point] = []
    bdr = board.boarder

    def cell_ok(x: int, y: int) -> bool:
        if (x, y) in occupied:
            return False
        if not board.is_within_bounds(Point(x, y)):
            return False
        if board.grid[x][y].state != 1:
            return False
        if player_id == 1:
            return y < bdr
        return y > bdr

    for _ in range(piece_cnt):
        pos = None
        for x, y in preferred_order:
            if cell_ok(x, y):
                pos = Point(x, y)
                break
        if pos is None:
            for y in range(board.height):
                for x in range(board.width):
                    if cell_ok(x, y):
                        pos = Point(x, y)
                        break
                if pos is not None:
                    break
        if pos is None:
            raise RuntimeError("init placement: no free cell in player's half")
        out.append(pos)
        occupied.add((pos.x, pos.y))
    return out


class StrategyFactory:
    """策略工厂类 - 提供不同的游戏策略"""
    
    @staticmethod
    def calculate_distance(p1: Point, p2: Point) -> float:
        """计算两点之间的距离"""
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    @staticmethod
    def get_aggressive_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """获取攻击型初始化策略"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            if pid == 1:
                order = [
                    (x, y)
                    for y in _init_front_y_range(board, 1)
                    for x in range(2, board.width - 2)
                ]
            else:
                order = [
                    (x, y)
                    for y in _init_front_y_range(board, 2)
                    for x in range(board.width - 3, 2, -1)
                ]
            positions = _allocate_init_positions(
                board, pid, init_message.piece_cnt, order
            )
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 20
                arg.dexterity = 8
                arg.intelligence = 2
                arg.equip = Point(2, 3)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args

        return strategy

    @staticmethod
    def get_defensive_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """获取防御型初始化策略"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            if pid == 1:
                order = [
                    (x, y)
                    for y in range(3, board.boarder)
                    for x in range(3, board.width - 3)
                ]
            else:
                order = [
                    (x, y)
                    for y in range(board.height - 1, board.boarder, -1)
                    for x in range(board.width - 4, 3, -1)
                ]
            positions = _allocate_init_positions(
                board, pid, init_message.piece_cnt, order
            )
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 5
                arg.dexterity = 15
                arg.intelligence = 10
                arg.equip = Point(3, 1)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args

        return strategy

    @staticmethod
    def get_aggressive_action_strategy() -> Callable[[Environment], ActionSet]:
        """获取攻击型行动策略 - 主动接近并攻击敌人"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            current_piece = env.current_piece
            
            # 寻找最近的敌人
            target_enemy = None
            nearest_distance = float('inf')
            
            for piece in env.action_queue:
                if piece.team != current_piece.team and piece.is_alive:
                    distance = StrategyFactory.calculate_distance(
                        Point(current_piece.position.x, current_piece.position.y),
                        Point(piece.position.x, piece.position.y)
                    )
                    if distance < nearest_distance:
                        nearest_distance = distance
                        target_enemy = piece
            
            # 没有敌人，不执行任何动作
            if target_enemy is None:
                action.move = False
                action.attack = False
                action.spell = False
                return action
            
            # 移动决策 - 向目标敌人移动
            # 获取所有合法移动位置
            legal_moves = get_legal_moves(env)
            if legal_moves:
                # 找到最接近敌人的合法移动位置
                best_move = None
                min_distance = float('inf')
                for move in legal_moves:
                    distance = StrategyFactory.calculate_distance(move, target_enemy.position)
                    if distance < min_distance:
                        min_distance = distance
                        best_move = move
                
                if best_move:
                    action.move = True
                    action.move_target = best_move
                else:
                    action.move = False
            else:
                action.move = False
            
            # 若已在有效攻击范围内（含高地射程加成），则攻击
            if env.is_in_attack_range(current_piece, target_enemy):
                action.attack = True
                action.attack_context = AttackContext()
                action.attack_context.attacker = current_piece
                action.attack_context.target = target_enemy
                action.attack_context.attackPosition = current_piece.position
            else:
                action.attack = False
            
            # 暂不使用法术
            action.spell = False
            
            return action
        
        return strategy

    @staticmethod
    def get_defensive_action_strategy() -> Callable[[Environment], ActionSet]:
        """获取防御型行动策略 - 保持距离，使用远程攻击"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            current_piece = env.current_piece
            
            # 寻找最近的敌人
            target_enemy = None
            nearest_distance = float('inf')
            
            for piece in env.action_queue:
                if piece.team != current_piece.team and piece.is_alive:
                    distance = StrategyFactory.calculate_distance(
                        Point(current_piece.position.x, current_piece.position.y),
                        Point(piece.position.x, piece.position.y)
                    )
                    if distance < nearest_distance:
                        nearest_distance = distance
                        target_enemy = piece
            
            # 没有敌人，不执行任何动作
            if target_enemy is None:
                action.move = False
                action.attack = False
                action.spell = False
                return action
            
            # 移动决策 - 保持在攻击范围内，但不要太近
            # 获取所有合法移动位置
            legal_moves = get_legal_moves(env)
            if legal_moves:
                # 理想距离是攻击范围的70%
                ideal_distance = current_piece.attack_range * 0.7
                
                # 找到最接近理想距离的合法移动位置
                best_move = None
                min_distance_diff = float('inf')
                
                for move in legal_moves:
                    distance_to_enemy = StrategyFactory.calculate_distance(move, target_enemy.position)
                    distance_diff = abs(distance_to_enemy - ideal_distance)
                    
                    if distance_diff < min_distance_diff:
                        # 如果太近了，只考虑能让我们远离的移动
                        if nearest_distance < ideal_distance - 2:
                            if distance_to_enemy > nearest_distance:
                                min_distance_diff = distance_diff
                                best_move = move
                        # 如果太远了，只考虑能让我们靠近的移动
                        elif nearest_distance > ideal_distance + 2:
                            if distance_to_enemy < nearest_distance:
                                min_distance_diff = distance_diff
                                best_move = move
                        # 如果在理想范围附近，选择最接近理想距离的位置
                        else:
                            min_distance_diff = distance_diff
                            best_move = move
                
                if best_move:
                    action.move = True
                    action.move_target = best_move
                else:
                    action.move = False
            else:
                action.move = False
            
            # 如果在攻击范围内，则攻击
            if env.is_in_attack_range(current_piece, target_enemy):
                action.attack = True
                action.attack_context = AttackContext()
                action.attack_context.attacker = current_piece
                action.attack_context.target = target_enemy
                action.attack_context.attackPosition = current_piece.position
            else:
                action.attack = False
            
            # 暂不使用法术
            action.spell = False
            
            return action
        
        return strategy

    # ==================================================================
    #  随机策略选择器（预训练时每局随机换阵容/风格）
    # ==================================================================

    @staticmethod
    def get_random_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """每局从所有阵容中随机选一个（全队统一，每局独立）。"""
        import random
        _POOL = [
            StrategyFactory.get_swordsman3_init_strategy(),
            StrategyFactory.get_assassin3_init_strategy(),
            StrategyFactory.get_mage3_v2_init_strategy(),
            StrategyFactory.get_balanced_team_init_strategy(),
            StrategyFactory.get_tank_healer_dps_init_strategy(),
            StrategyFactory.get_assassin_archer_mage_init_strategy(),
            StrategyFactory.get_battlemage3_init_strategy(),
            StrategyFactory.get_archer3_heavy_init_strategy(),
            StrategyFactory.get_archer22_heavy_init_strategy(),
        ]
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            return random.choice(_POOL)(init_message)
        return strategy

    @staticmethod
    def get_random_mixed_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """每颗棋子独立随机选择职业，阵容多样性最大化。"""
        import random
        _PIECE_POOLS = [
            (18, 6, 6, 1, 2),     # 剑士 长剑+中甲
            (11, 16, 3, 2, 1),    # 刺客 短剑+轻甲
            (0, 6, 24, 4, 1),     # 法师 法杖+轻甲
            (22, 4, 4, 1, 3),     # 重坦 长剑+重甲
            (10, 12, 8, 3, 1),    # 弓手 弓+轻甲
            (8, 8, 14, 1, 1),     # 战法 长剑+轻甲
        ]
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args = []
            for pos in positions:
                s, d, i, wx, wy = random.choice(_PIECE_POOLS)
                arg = PieceArg()
                arg.strength = s; arg.dexterity = d; arg.intelligence = i
                arg.equip = Point(wx, wy); arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_random_action_strategy() -> Callable[[Environment], ActionSet]:
        """每一步从10种风格中随机选一个。"""
        import random
        _POOL = [
            StrategyFactory.get_aggressive_spells_action_strategy(),
            StrategyFactory.get_defensive_spells_action_strategy(),
            StrategyFactory.get_kite_spells_action_strategy(),
            StrategyFactory.get_sniper_spells_action_strategy(),
            StrategyFactory.get_zone_control_action_strategy(),
            StrategyFactory.get_healer_support_action_strategy(),
        ]
        def strategy(env: Environment) -> ActionSet:
            return random.choice(_POOL)(env)
        return strategy

    @staticmethod
    def get_per_game_random_action_strategy() -> Callable[[Environment], ActionSet]:
        """开局时随机选定一种行动风格，整局统一使用。"""
        import random
        _POOL = [
            StrategyFactory.get_aggressive_spells_action_strategy,
            StrategyFactory.get_defensive_spells_action_strategy,
            StrategyFactory.get_kite_spells_action_strategy,
            StrategyFactory.get_sniper_spells_action_strategy,
            StrategyFactory.get_zone_control_action_strategy,
            StrategyFactory.get_healer_support_action_strategy,
        ]
        return random.choice(_POOL)()

    # ==================================================================
    #  ★ 新规则启发式行动策略（用于预训练数据生成）
    # ==================================================================

    @staticmethod
    def _nearest_enemy(env: Environment):
        """返回 (最近敌人, 距离)。"""
        p = env.current_piece
        if p is None:
            return None, float('inf')
        best, best_d = None, float('inf')
        for q in env.action_queue:
            if q.is_alive and q.team != p.team:
                d = StrategyFactory.calculate_distance(p.position, q.position)
                if d < best_d:
                    best_d = d
                    best = q
        return best, best_d

    @staticmethod
    def _safe_zone_center(env: Environment) -> Point:
        """返回安全区中心点。"""
        if hasattr(env.board, 'zone_cx'):
            return Point(int(env.board.zone_cx), int(env.board.zone_cy))
        return Point(0, 0)

    @staticmethod
    def _is_outside_zone(env: Environment, pos: Point) -> bool:
        """检查位置是否在安全区外。"""
        if not hasattr(env.board, 'zone_has_shrink') or not env.board.zone_has_shrink:
            return False
        radius = env._current_zone_radius() if hasattr(env, '_current_zone_radius') else float('inf')
        if radius == float('inf'):
            return False
        return not env.board.is_in_safe_zone(pos.x, pos.y, radius)

    # ==================================================================
    #  法术辅助函数
    # ==================================================================

    @staticmethod
    def _try_offensive_spell(env: Environment, action, target_enemy):
        """尝试对敌人释放伤害法术（Arrow Hit / Fireball）。成功则设置 action.spell。"""
        piece = env.current_piece
        if piece is None or piece.spell_slots <= 0 or piece.action_points <= 0:
            return
        spells = env.get_available_spells(piece)
        for s in spells:
            if str(getattr(s, 'effect_type', '')) not in ('SpellEffectType.DAMAGE', 'DAMAGE'):
                continue
            dist = StrategyFactory.calculate_distance(piece.position, target_enemy.position)
            if dist <= getattr(s, 'range', 0):
                action.spell = True
                ctx = SpellContext()
                ctx.caster = piece; ctx.spell = s; ctx.target = target_enemy
                if getattr(s, 'is_area_effect', False):
                    ctx.target_area = Area(target_enemy.position.x, target_enemy.position.y,
                                            int(getattr(s, 'area_radius', 1)))
                elif getattr(s, 'is_locking_spell', False):
                    ctx.target_area = Area(target_enemy.position.x, target_enemy.position.y, 1)
                action.spell_context = ctx
                return

    @staticmethod
    def _try_heal_spell(env: Environment, action):
        """尝试治疗最残血的友方棋子。成功则设置 action.spell。"""
        piece = env.current_piece
        if piece is None or piece.spell_slots <= 0 or piece.action_points <= 0:
            return
        spells = env.get_available_spells(piece)
        heal = None
        for s in spells:
            if str(getattr(s, 'effect_type', '')) in ('SpellEffectType.HEAL', 'HEAL'):
                heal = s; break
        if heal is None:
            return
        best, ratio = None, 1.0
        for p in env.action_queue:
            if p.is_alive and p.team == piece.team and p.id != piece.id:
                r = p.health / max(p.max_health, 1)
                if r < ratio: ratio = r; best = p
        if best and ratio < 0.8:
            dist = StrategyFactory.calculate_distance(piece.position, best.position)
            if dist <= getattr(heal, 'range', 4):
                action.spell = True
                ctx = SpellContext()
                ctx.caster = piece; ctx.spell = heal; ctx.target = best
                if getattr(heal, 'is_area_effect', False):
                    ctx.target_area = Area(best.position.x, best.position.y,
                                            int(getattr(heal, 'area_radius', 1)))
                action.spell_context = ctx

    @staticmethod
    def _try_teleport_spell(env: Environment, action):
        """低血量或圈外时用传送逃到安全位置。"""
        piece = env.current_piece
        if piece is None or piece.spell_slots <= 0 or piece.action_points <= 0:
            return
        spells = env.get_available_spells(piece)
        tele = None
        for s in spells:
            if str(getattr(s, 'effect_type', '')) in ('SpellEffectType.MOVE', 'MOVE'):
                tele = s; break
        if tele is None:
            return
        # 触发条件：血量 < 30% 或在圈外
        hp_ratio = piece.health / max(piece.max_health, 1)
        outside = StrategyFactory._is_outside_zone(env, piece.position)
        if hp_ratio >= 0.3 and not outside:
            return
        # 安全目标：向安全区中心移动
        center = StrategyFactory._safe_zone_center(env)
        # 在安全区内找一个离中心近、离敌人远的位置
        best_target = None
        best_score = float('-inf')
        moves = get_legal_moves(env)
        for m in moves[:50]:  # 限制搜索
            if StrategyFactory._is_outside_zone(env, m):
                continue
            d_center = StrategyFactory.calculate_distance(m, center)
            score = -d_center
            if score > best_score:
                best_score = score
                best_target = m
        if best_target and best_target != piece.position:
            action.spell = True
            ctx = SpellContext()
            ctx.caster = piece; ctx.spell = tele
            ctx.target_area = Area(best_target.x, best_target.y, 100)
            action.spell_context = ctx
            return

    @staticmethod
    def _find_highest_ground(env: Environment, moves: list) -> Point:
        """从可移动位置中找最高点。"""
        if not moves:
            return None
        best = max(moves, key=lambda m: env.board.height_map[m.x][m.y])
        return best

    @staticmethod
    def get_kite_action_strategy() -> Callable[[Environment], ActionSet]:
        """风筝策略：远程职业专用，攻击后向安全区或远离敌人方向移动，保持最大射程优势。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action

            enemy, dist = StrategyFactory._nearest_enemy(env)
            if enemy is None:
                return action

            # 攻击决策：在有效射程内则攻击
            if env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp
                ctx.target = enemy
                action.attack_context = ctx

            # 移动决策：向远离敌人的方向或安全区移动
            moves = get_legal_moves(env)
            if moves:
                # 优先选择远离最近敌人的位置
                best_move = None
                best_score = float('-inf')
                center = StrategyFactory._safe_zone_center(env)
                for m in moves:
                    d_to_enemy = StrategyFactory.calculate_distance(m, enemy.position)
                    d_to_center = StrategyFactory.calculate_distance(m, center)
                    zone_bonus = 50.0 if not StrategyFactory._is_outside_zone(env, m) else -50.0
                    # 越远越好，越靠近安全区越好，高地加分
                    height_bonus = env.board.height_map[m.x][m.y] * 3.0
                    score = d_to_enemy + 20.0 - d_to_center * 0.5 + zone_bonus + height_bonus
                    if score > best_score:
                        best_score = score
                        best_move = m
                if best_move:
                    action.move = True
                    action.move_target = best_move

            action.spell = False
            return action
        return strategy

    @staticmethod
    def get_sniper_action_strategy() -> Callable[[Environment], ActionSet]:
        """狙击策略：优先抢占高地获得+2射程+3伤害的优势，再攻击。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action

            enemy, dist = StrategyFactory._nearest_enemy(env)
            if enemy is None:
                return action

            moves = get_legal_moves(env)
            current_h = env.board.height_map[cp.position.x][cp.position.y]

            if moves:
                # 找最高可到达位置
                high_ground = StrategyFactory._find_highest_ground(env, moves)
                high_h = env.board.height_map[high_ground.x][high_ground.y] if high_ground else 0

                # 如果高地更高且能攻击到敌人，优先上高地
                if high_ground and high_h > current_h:
                    # 检查从高地能否攻击到敌人
                    dh = high_h - enemy.height if hasattr(enemy, 'height') else 0
                    eff_range = cp.attack_range + 2 * max(0, dh)
                    dist_from_high = StrategyFactory.calculate_distance(high_ground, enemy.position)
                    if dist_from_high <= eff_range or dist_from_high <= cp.attack_range + 4:
                        action.move = True
                        action.move_target = high_ground

                if not action.move:
                    # 否则找最近的能攻击到的位置
                    for m in moves:
                        if env.is_in_attack_range(cp, enemy):
                            action.move = False
                            break
                        d = StrategyFactory.calculate_distance(m, enemy.position)
                        if d <= cp.attack_range + 2:
                            action.move = True
                            action.move_target = m
                            break

            # 如果在有效射程内则攻击
            if env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp
                ctx.target = enemy
                action.attack_context = ctx

            action.spell = False
            return action
        return strategy

    @staticmethod
    def get_zone_control_action_strategy() -> Callable[[Environment], ActionSet]:
        """缩圈控制策略：优先向安全区中心移动，圈外时快速回缩，避免圈外伤害。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action

            enemy, dist = StrategyFactory._nearest_enemy(env)
            center = StrategyFactory._safe_zone_center(env)
            outside = StrategyFactory._is_outside_zone(env, cp.position)

            moves = get_legal_moves(env)
            if moves:
                best_move = None
                best_score = float('-inf')
                for m in moves:
                    d_to_enemy = StrategyFactory.calculate_distance(m, enemy.position) if enemy else 999
                    d_to_center = StrategyFactory.calculate_distance(m, center)
                    in_zone = not StrategyFactory._is_outside_zone(env, m)
                    # 圈外时：安全区优先于一切
                    zone_score = 1000 if in_zone else -1000
                    if outside:
                        # 圈外时拼命回缩
                        score = zone_score - d_to_center * 10 + d_to_enemy * 0.1
                    else:
                        # 圈内时正常交战，但注意不要跑出圈
                        score = zone_score - d_to_enemy - d_to_center * 0.3
                    if score > best_score:
                        best_score = score
                        best_move = m
                if best_move:
                    action.move = True
                    action.move_target = best_move

            # 在射程内则攻击
            if enemy and env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp
                ctx.target = enemy
                action.attack_context = ctx

            action.spell = False
            return action
        return strategy

    @staticmethod
    def get_healer_support_action_strategy() -> Callable[[Environment], ActionSet]:
        """治疗支援策略：法师专用，优先治疗低血量友方，使用火球/箭击远程支援。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action

            # 找受伤最重的友方
            most_wounded = None
            lowest_hp_ratio = 1.0
            for p in env.action_queue:
                if p.is_alive and p.team == cp.team and p.id != cp.id:
                    ratio = p.health / max(p.max_health, 1)
                    if ratio < lowest_hp_ratio:
                        lowest_hp_ratio = ratio
                        most_wounded = p

            # 尝试使用治疗法术
            spells = env.get_available_spells(cp)
            heal_spell = None
            for s in spells:
                if hasattr(s, 'effect_type') and str(s.effect_type) in ('SpellEffectType.HEAL', 'HEAL'):
                    heal_spell = s
                    break

            # 如果有受伤的友方且有治疗法术
            if most_wounded and heal_spell and lowest_hp_ratio < 0.8:
                dist_to_ally = StrategyFactory.calculate_distance(cp.position, most_wounded.position)
                if dist_to_ally <= heal_spell.range:
                    action.spell = True
                    ctx = SpellContext()
                    ctx.caster = cp
                    ctx.spell = heal_spell
                    ctx.target = most_wounded
                    if heal_spell.is_area_effect:
                        ctx.target_area = Area(most_wounded.position.x, most_wounded.position.y,
                                                int(heal_spell.area_radius))
                    action.spell_context = ctx
                    # 施法后向友方靠近
                    moves = get_legal_moves(env)
                    if moves:
                        best = min(moves, key=lambda m: StrategyFactory.calculate_distance(m, most_wounded.position))
                        if best:
                            action.move = True
                            action.move_target = best
                else:
                    # 太远，先靠近
                    moves = get_legal_moves(env)
                    if moves:
                        best = min(moves, key=lambda m: StrategyFactory.calculate_distance(m, most_wounded.position))
                        if best:
                            action.move = True
                            action.move_target = best

            # 没施法则正常攻击
            if not action.spell:
                enemy, dist = StrategyFactory._nearest_enemy(env)
                if enemy:
                    # 先找火球类范围伤害法术
                    for s in spells:
                        if hasattr(s, 'is_area_effect') and s.is_area_effect:
                            dist_to_enemy = StrategyFactory.calculate_distance(cp.position, enemy.position)
                            if dist_to_enemy <= s.range:
                                action.spell = True
                                ctx = SpellContext()
                                ctx.caster = cp
                                ctx.spell = s
                                ctx.target_area = Area(enemy.position.x, enemy.position.y,
                                                        int(s.area_radius))
                                action.spell_context = ctx
                                break

                    if not action.spell and env.is_in_attack_range(cp, enemy):
                        action.attack = True
                        ctx = AttackContext()
                        ctx.attacker = cp
                        ctx.target = enemy
                        action.attack_context = ctx

                    if not action.move:
                        moves = get_legal_moves(env)
                        if moves:
                            best = min(moves, key=lambda m: StrategyFactory.calculate_distance(m, enemy.position))
                            if best:
                                action.move = True
                                action.move_target = best

            return action
        return strategy

    # ==================================================================
    #  ★ 带法术的变体策略（用于预训练数据生成，让模型学会施法）
    # ==================================================================

    @staticmethod
    def get_aggressive_spells_action_strategy() -> Callable[[Environment], ActionSet]:
        """aggressive + 法术：冲向敌人 + 伤害法术/治疗。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action
            enemy, dist = StrategyFactory._nearest_enemy(env)
            if enemy is None:
                return action

            moves = get_legal_moves(env)
            if moves:
                best = min(moves, key=lambda m: StrategyFactory.calculate_distance(m, enemy.position))
                if best:
                    action.move = True
                    action.move_target = best

            if env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp; ctx.target = enemy
                action.attack_context = ctx

            action.spell = False
            StrategyFactory._try_teleport_spell(env, action)
            if not action.spell:
                StrategyFactory._try_heal_spell(env, action)
            if not action.spell:
                StrategyFactory._try_offensive_spell(env, action, enemy)
            return action
        return strategy

    @staticmethod
    def get_defensive_spells_action_strategy() -> Callable[[Environment], ActionSet]:
        """defensive + 法术：保持距离 + 伤害法术/治疗。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action
            enemy, dist = StrategyFactory._nearest_enemy(env)
            if enemy is None:
                return action

            ideal = cp.attack_range * 0.7
            moves = get_legal_moves(env)
            if moves:
                best, best_diff = None, float('inf')
                for m in moves:
                    d = StrategyFactory.calculate_distance(m, enemy.position)
                    diff = abs(d - ideal)
                    if diff < best_diff:
                        best_diff = diff
                        best = m
                if best:
                    action.move = True
                    action.move_target = best

            if env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp; ctx.target = enemy
                action.attack_context = ctx

            action.spell = False
            StrategyFactory._try_heal_spell(env, action)
            if not action.spell:
                StrategyFactory._try_offensive_spell(env, action, enemy)
            return action
        return strategy

    @staticmethod
    def get_kite_spells_action_strategy() -> Callable[[Environment], ActionSet]:
        """kite + 法术：风筝 + 远程伤害法术。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action
            enemy, dist = StrategyFactory._nearest_enemy(env)
            if enemy is None:
                return action

            if env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp; ctx.target = enemy
                action.attack_context = ctx

            moves = get_legal_moves(env)
            if moves:
                center = StrategyFactory._safe_zone_center(env)
                best, best_score = None, float('-inf')
                for m in moves:
                    de = StrategyFactory.calculate_distance(m, enemy.position)
                    dc = StrategyFactory.calculate_distance(m, center)
                    zs = 50 if not StrategyFactory._is_outside_zone(env, m) else -50
                    hb = env.board.height_map[m.x][m.y] * 3
                    score = de + 20 - dc * 0.5 + zs + hb
                    if score > best_score:
                        best_score = score
                        best = m
                if best:
                    action.move = True
                    action.move_target = best

            action.spell = False
            StrategyFactory._try_teleport_spell(env, action)
            if not action.spell:
                StrategyFactory._try_heal_spell(env, action)
            if not action.spell:
                StrategyFactory._try_offensive_spell(env, action, enemy)
            return action
        return strategy

    @staticmethod
    def get_sniper_spells_action_strategy() -> Callable[[Environment], ActionSet]:
        """sniper + 法术：抢占高地 + 远程法术。"""
        def strategy(env: Environment) -> ActionSet:
            action = ActionSet()
            cp = env.current_piece
            if cp is None:
                return action
            enemy, dist = StrategyFactory._nearest_enemy(env)
            if enemy is None:
                return action

            moves = get_legal_moves(env)
            cur_h = env.board.height_map[cp.position.x][cp.position.y]
            if moves:
                high = StrategyFactory._find_highest_ground(env, moves)
                hh = env.board.height_map[high.x][high.y] if high else 0
                if high and hh > cur_h:
                    dh = hh - (enemy.height if hasattr(enemy, 'height') else 0)
                    er = cp.attack_range + 2 * max(0, dh)
                    if StrategyFactory.calculate_distance(high, enemy.position) <= er + 2:
                        action.move = True
                        action.move_target = high
                if not action.move:
                    for m in moves:
                        if StrategyFactory.calculate_distance(m, enemy.position) <= cp.attack_range + 2:
                            action.move = True
                            action.move_target = m
                            break

            if env.is_in_attack_range(cp, enemy):
                action.attack = True
                ctx = AttackContext()
                ctx.attacker = cp; ctx.target = enemy
                action.attack_context = ctx

            action.spell = False
            StrategyFactory._try_teleport_spell(env, action)
            if not action.spell:
                StrategyFactory._try_heal_spell(env, action)
            if not action.spell:
                StrategyFactory._try_offensive_spell(env, action, enemy)
            return action
        return strategy

    # ------------------------------------------------------------------
    #  自定义初始化策略
    # ------------------------------------------------------------------

    @staticmethod
    def get_archer3_heavy_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3 弓箭手，加点 STR=29 DEX=1 INT=0，重甲（archer29）。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 29
                arg.dexterity = 1
                arg.intelligence = 0
                arg.equip = Point(3, 3)  # 弓+重甲
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_archer22_heavy_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3 弓箭手，加点 STR=22 DEX=4 INT=4，重甲（archer22）。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 22
                arg.dexterity = 4
                arg.intelligence = 4
                arg.equip = Point(3, 3)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_two_archers_one_mage_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """2 弓箭手（STR=22 DEX=4 INT=4 重甲）+ 1 法师（STR=4 DEX=4 INT=22 轻甲）。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args: List[PieceArg] = []
            for idx, pos in enumerate(positions):
                arg = PieceArg()
                if idx < 2:
                    arg.strength = 22
                    arg.dexterity = 4
                    arg.intelligence = 4
                    arg.equip = Point(3, 3)
                else:
                    arg.strength = 4
                    arg.dexterity = 4
                    arg.intelligence = 22
                    arg.equip = Point(4, 1)  # 法杖+轻甲
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_one_archer_two_mages_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """1 弓箭手（STR=29 DEX=1 INT=0 重甲）+ 2 法师（STR=4 DEX=4 INT=22 轻甲）。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args: List[PieceArg] = []
            for idx, pos in enumerate(positions):
                arg = PieceArg()
                if idx == 0:
                    arg.strength = 29
                    arg.dexterity = 1
                    arg.intelligence = 0
                    arg.equip = Point(3, 3)
                else:
                    arg.strength = 4
                    arg.dexterity = 4
                    arg.intelligence = 22
                    arg.equip = Point(4, 1)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_mage3_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3 法师，加点 STR=4 DEX=4 INT=22，轻甲（mage3）。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args: List[PieceArg] = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 4
                arg.dexterity = 4
                arg.intelligence = 22
                arg.equip = Point(4, 1)
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    # ==================================================================
    #  ★ 新规则初始化策略
    # ==================================================================

    @staticmethod
    def get_swordsman3_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3×剑士：长剑+中甲，力量18主打前排。移动=6/2+6+0=9，物伤=12+18=30，抗18。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid, depth=4)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 18; arg.dexterity = 6; arg.intelligence = 6
                arg.equip = Point(1, 2)  # 长剑+中甲
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_assassin3_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3×刺客：短剑+轻甲，高敏捷高机动。移动=16/2+6+3=17，物伤=16+11=27，抗10，无法术。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid, depth=5)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 11; arg.dexterity = 16; arg.intelligence = 3
                arg.equip = Point(2, 1)  # 短剑+轻甲
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_mage3_v2_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3×法师：法杖+轻甲，智力24=5法术位。火球=8+12=20伤，移动=6/2+6+3=12。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid, depth=3)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 0; arg.dexterity = 6; arg.intelligence = 24
                arg.equip = Point(4, 1)  # 法杖+轻甲
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_balanced_team_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """均衡队：剑士(前排)+弓手(后排)+法师(法术)。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            configs = [
                (18, 6, 6, Point(1, 2)),   # 剑士 长剑+中甲
                (10, 12, 8, Point(3, 1)),  # 弓手 弓+轻甲
                (0, 6, 24, Point(4, 1)),   # 法师 法杖+轻甲
            ]
            piece_args = []
            for pos, (s, d, i, eq) in zip(positions, configs):
                arg = PieceArg()
                arg.strength = s; arg.dexterity = d; arg.intelligence = i
                arg.equip = eq; arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_tank_healer_dps_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """重装+治疗+输出：重甲坦(长剑)+治疗法师(法杖)+弓手。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            configs = [
                (22, 4, 4, Point(1, 3)),   # 坦克 长剑+重甲
                (0, 8, 22, Point(4, 1)),   # 治疗法师 法杖+轻甲
                (10, 12, 8, Point(3, 1)),  # 弓手 弓+轻甲
            ]
            piece_args = []
            for pos, (s, d, i, eq) in zip(positions, configs):
                arg = PieceArg()
                arg.strength = s; arg.dexterity = d; arg.intelligence = i
                arg.equip = eq; arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_assassin_archer_mage_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """刺客+弓手+法师：高机动混合队。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            configs = [
                (11, 16, 3, Point(2, 1)),  # 刺客 短剑+轻甲
                (10, 12, 8, Point(3, 1)),  # 弓手 弓+轻甲
                (0, 6, 24, Point(4, 1)),   # 法师 法杖+轻甲
            ]
            piece_args = []
            for pos, (s, d, i, eq) in zip(positions, configs):
                arg = PieceArg()
                arg.strength = s; arg.dexterity = d; arg.intelligence = i
                arg.equip = eq; arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    @staticmethod
    def get_battlemage3_init_strategy() -> Callable[['InitGameMessage'], List[PieceArg]]:
        """3×战斗法师：长剑+轻甲，智力14=2法术位，力量8。移动=8/2+6+3=13。"""
        def strategy(init_message: 'InitGameMessage') -> List[PieceArg]:
            board = init_message.board
            pid = init_message.id
            order = [(x, y) for y in _init_front_y_range(board, pid)
                     for x in range(board.width)]
            positions = _allocate_init_positions(board, pid, init_message.piece_cnt, order)
            piece_args = []
            for pos in positions:
                arg = PieceArg()
                arg.strength = 8; arg.dexterity = 8; arg.intelligence = 14
                arg.equip = Point(1, 1)  # 长剑+轻甲
                arg.pos = pos
                piece_args.append(arg)
            return piece_args
        return strategy

    # ------------------------------------------------------------------
    #  初始化策略查询
    # ------------------------------------------------------------------

    @staticmethod
    def get_init_strategy_by_name(name: str) -> Callable[['InitGameMessage'], List[PieceArg]]:
        mapping = {
            "aggressive": StrategyFactory.get_aggressive_init_strategy(),
            "defensive": StrategyFactory.get_defensive_init_strategy(),
            "archer29": StrategyFactory.get_archer3_heavy_init_strategy(),
            "archer22": StrategyFactory.get_archer22_heavy_init_strategy(),
            "2archer1mage": StrategyFactory.get_two_archers_one_mage_init_strategy(),
            "1archer2mage": StrategyFactory.get_one_archer_two_mages_init_strategy(),
            "mage3": StrategyFactory.get_mage3_init_strategy(),
            "swordsman3": StrategyFactory.get_swordsman3_init_strategy(),
            "assassin3": StrategyFactory.get_assassin3_init_strategy(),
            "mage3_v2": StrategyFactory.get_mage3_v2_init_strategy(),
            "balanced": StrategyFactory.get_balanced_team_init_strategy(),
            "tank_healer_dps": StrategyFactory.get_tank_healer_dps_init_strategy(),
            "assassin_archer_mage": StrategyFactory.get_assassin_archer_mage_init_strategy(),
            "battlemage3": StrategyFactory.get_battlemage3_init_strategy(),
            "random": StrategyFactory.get_random_init_strategy(),
            "random_mixed": StrategyFactory.get_random_mixed_init_strategy(),
        }
        if name not in mapping:
            raise ValueError(f"Unknown init strategy: {name}")
        return mapping[name]

    # ------------------------------------------------------------------
    #  PUCT / MCTS 行动策略（本地训练用）
    # ------------------------------------------------------------------

    @staticmethod
    def get_puct_action_strategy(
        model=None,
        processor=None,
        device="cpu",
        simulations: int = 16,
        temperature: float = 0.0,
    ) -> Callable[[Environment], ActionSet]:
        """使用模型和 PersistentMCTS 生成行动策略。"""
        if model is None or processor is None:
            raise ValueError("Model and processor are required for PUCT action strategy")
        import torch
        from mcts import PersistentMCTS
        persistent = PersistentMCTS(
            model, processor, torch.device(device), simulations=simulations
        )

        def strategy(env: Environment) -> ActionSet:
            action = persistent.select_action(env, temperature=temperature)
            strategy._last_visit_dists = persistent.get_visit_distributions()
            return action

        strategy._persistent_mcts = persistent
        strategy._temperature = temperature
        strategy._last_visit_dists = None
        return strategy

    @staticmethod
    def get_action_strategy_by_name(
        name: str,
        model=None,
        processor=None,
        device="cpu",
        simulations: int = 16,
    ) -> Callable[[Environment], ActionSet]:
        if name == "aggressive":
            return StrategyFactory.get_aggressive_action_strategy()
        if name == "defensive":
            return StrategyFactory.get_defensive_action_strategy()
        if name == "random":
            return StrategyFactory.get_random_action_strategy()
        if name == "alpha_beta":
            return StrategyFactory.get_alpha_beta_action_strategy()
        if name == "kite":
            return StrategyFactory.get_kite_action_strategy()
        if name == "sniper":
            return StrategyFactory.get_sniper_action_strategy()
        if name == "zone_control":
            return StrategyFactory.get_zone_control_action_strategy()
        if name == "healer":
            return StrategyFactory.get_healer_support_action_strategy()
        if name == "aggressive_spells":
            return StrategyFactory.get_aggressive_spells_action_strategy()
        if name == "defensive_spells":
            return StrategyFactory.get_defensive_spells_action_strategy()
        if name == "kite_spells":
            return StrategyFactory.get_kite_spells_action_strategy()
        if name == "sniper_spells":
            return StrategyFactory.get_sniper_spells_action_strategy()
        if name == "random_per_game":
            return StrategyFactory.get_per_game_random_action_strategy()
        if name == "puct":
            return StrategyFactory.get_puct_action_strategy(
                model=model, processor=processor, device=device, simulations=simulations,
            )
        raise ValueError(f"Unknown action strategy: {name}")

    @staticmethod
    def get_alpha_beta_action_strategy(max_depth: int = 3) -> Callable[[Environment], ActionSet]:
        """获取基于AlphaBeta剪枝的行动策略
        
        Args:
            max_depth: 最大搜索深度
            
        Returns:
            Callable[[Environment], ActionSet]: 策略函数
        """
        def alpha_beta(env: Environment, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[ActionSet]]:
            if depth == 0 or env.is_game_over:
                return get_state_score(env), None
                
            current_piece = env.current_piece
            if maximizing:
                max_eval = float('-inf')
                best_action = None
                
                # 获取所有可能的行动
                legal_moves = get_legal_moves(env)
                attackable_targets = get_attackable_targets(env)
                
                # 获取当前棋子可用的法术
                spells = env.get_available_spells(current_piece)
                
                # 尝试每个可能的行动组合
                for move in [None] + legal_moves:
                    # 如果已经没有行动点，跳过移动
                    if move is not None and current_piece.action_points <= 0:
                        continue
                        
                    for target in [None] + attackable_targets:
                        # 如果已经没有行动点，跳过攻击
                        if target is not None and current_piece.action_points <= 0:
                            continue
                            
                        for spell in [None] + spells:
                            # 如果已经没有行动点或法术位，跳过法术
                            if spell is not None and (current_piece.action_points <= 0 or current_piece.spell_slots <= 0):
                                continue
                                
                            action = ActionSet()
                            next_env = fork_environment(env)
                            remaining_points = current_piece.action_points
                            
                            # 设置移动
                            if move is not None and remaining_points > 0:
                                action.move = True
                                action.move_target = move
                                remaining_points -= 1
                            else:
                                action.move = False
                            
                            # 设置攻击
                            if target is not None and remaining_points > 0:
                                action.attack = True
                                action.attack_context = AttackContext()
                                action.attack_context.attacker = current_piece
                                action.attack_context.target = target
                                remaining_points -= 1
                            else:
                                action.attack = False
                            
                            # 设置法术
                            if spell is not None and remaining_points > 0 and current_piece.spell_slots > 0:
                                action.spell = True
                                action.spell_context = SpellContext()
                                action.spell_context.caster = current_piece
                                action.spell_context.target = target if target else current_piece  # 如果没有目标就施放在自己身上
                                action.spell_context.spell = spell
                                action.spell_context.target_area = Area(
                                    current_piece.position.x,
                                    current_piece.position.y,
                                    2  # 默认范围
                                )
                                remaining_points -= 1
                            else:
                                action.spell = False
                            
                            # 模拟行动
                            next_env.execute_player_action(action)
                            
                            eval, _ = alpha_beta(next_env, depth - 1, alpha, beta, False)
                            if eval > max_eval:
                                max_eval = eval
                                best_action = action
                                
                            alpha = max(alpha, eval)
                            if beta <= alpha:
                                break
                        if beta <= alpha:
                            break
                    if beta <= alpha:
                        break
                            
                return max_eval, best_action
            else:
                min_eval = float('inf')
                best_action = None
                
                # 获取所有可能的行动
                legal_moves = get_legal_moves(env)
                attackable_targets = get_attackable_targets(env)
                
                # 创建基础法术列表
                spells = []
                if current_piece.spell_slots > 0:
                    spells.extend([
                        Spell("Damage", "Damage", 10, False),
                        Spell("Heal", "Heal", 8, False),
                        Spell("Buff", "Buff", 5, False),
                        Spell("Debuff", "Debuff", 3, False)
                    ])
                
                # 尝试每个可能的行动组合
                for move in [None] + legal_moves:
                    # 如果已经没有行动点，跳过移动
                    if move is not None and current_piece.action_points <= 0:
                        continue
                        
                    for target in [None] + attackable_targets:
                        # 如果已经没有行动点，跳过攻击
                        if target is not None and current_piece.action_points <= 0:
                            continue
                            
                        for spell in [None] + spells:
                            # 如果已经没有行动点或法术位，跳过法术
                            if spell is not None and (current_piece.action_points <= 0 or current_piece.spell_slots <= 0):
                                continue
                                
                            action = ActionSet()
                            next_env = fork_environment(env)
                            remaining_points = current_piece.action_points
                            
                            # 设置移动
                            if move is not None and remaining_points > 0:
                                action.move = True
                                action.move_target = move
                                remaining_points -= 1
                            else:
                                action.move = False
                            
                            # 设置攻击
                            if target is not None and remaining_points > 0:
                                action.attack = True
                                action.attack_context = AttackContext()
                                action.attack_context.attacker = current_piece
                                action.attack_context.target = target
                                remaining_points -= 1
                            else:
                                action.attack = False
                            
                            # 设置法术
                            if spell is not None and remaining_points > 0 and current_piece.spell_slots > 0:
                                # 获取法术可选目标
                                spell_targets = env.get_spell_targets(spell, current_piece)
                                if not spell_targets and not spell.is_area_effect:
                                    continue
                                    
                                action.spell = True
                                action.spell_context = SpellContext()
                                action.spell_context.caster = current_piece
                                action.spell_context.spell = spell
                                
                                # 设置目标和范围
                                if spell.is_area_effect:
                                    # 范围法术以当前位置为中心
                                    action.spell_context.target = None
                                    action.spell_context.target_area = Area(
                                        current_piece.position.x,
                                        current_piece.position.y,
                                        spell.area_radius
                                    )
                                else:
                                    # 单体法术选择最佳目标
                                    best_target = None
                                    if spell.effect_type in [SpellEffectType.DAMAGE, SpellEffectType.DEBUFF]:
                                        # 选择生命值最低的敌人
                                        best_target = min(spell_targets, key=lambda p: p.health)
                                    elif spell.effect_type in [SpellEffectType.HEAL, SpellEffectType.BUFF]:
                                        # 选择生命值损失最多的友军
                                        best_target = min(spell_targets, key=lambda p: p.health / p.max_health)
                                    elif spell.effect_type == SpellEffectType.MOVE:
                                        best_target = current_piece
                                        
                                    action.spell_context.target = best_target
                                    action.spell_context.target_area = Area(
                                        best_target.position.x,
                                        best_target.position.y,
                                        0
                                    )
                                    
                                remaining_points -= 1
                            else:
                                action.spell = False
                            
                            # 模拟行动
                            next_env.execute_player_action(action)
                            
                            eval, _ = alpha_beta(next_env, depth - 1, alpha, beta, True)
                            if eval < min_eval:
                                min_eval = eval
                                best_action = action
                                
                            beta = min(beta, eval)
                            if beta <= alpha:
                                break
                        if beta <= alpha:
                            break
                    if beta <= alpha:
                        break
                            
                return min_eval, best_action
        
        def strategy(env: Environment) -> ActionSet:
            _, best_action = alpha_beta(env, max_depth, float('-inf'), float('inf'), True)
            return best_action if best_action is not None else ActionSet()
            
        return strategy
        
  
    def get_mcts_action_strategy(simulation_count: int = 10) -> Callable[[Environment], ActionSet]:
        """获取基于MCTS的行动策略
        
        Args:
            simulation_count: 每个决策点的模拟次数
            
        Returns:
            Callable[[Environment], ActionSet]: 策略函数
        """
        class MCTSNode:
            def __init__(self, env: Environment, parent=None, action: Optional[ActionSet] = None):
                self.env = env
                self.parent = parent
                self.action = action
                self.children = []
                self.visits = 0
                self.value = 0.0
                
            def expand(self):
                """扩展当前节点"""
                current_piece = self.env.current_piece
                legal_moves = get_legal_moves(self.env)
                attackable_targets = get_attackable_targets(self.env)
                
                # 获取当前棋子可用的法术
                spells = self.env.get_available_spells(current_piece)
                
                # print(f"[MCTS] 开始生成动作组合:")
                # print(f"[MCTS] - 可移动位置: {len(legal_moves)}")
                # print(f"[MCTS] - 可攻击目标: {len(attackable_targets)}")
                # print(f"[MCTS] - 可用法术: {len(spells)}")
                # print(f"[MCTS] - 当前行动点: {current_piece.action_points}")

                # 生成所有可能的行动组合
                for move in [None] + legal_moves:
                    # 如果已经没有行动点，跳过移动
                    if move is not None and current_piece.action_points <= 0:
                        if MCTS_VERBOSE: print("[MCTS] 跳过移动：没有足够的行动点")
                        continue
                        
                    for target in [None] + attackable_targets:
                        # 如果已经没有行动点，跳过攻击
                        if target is not None and current_piece.action_points <= 0:
                            if MCTS_VERBOSE: print("[MCTS] 跳过攻击：没有足够的行动点")
                            continue
                            
                        for spell in [None] + spells:
                            # 如果已经没有行动点或法术位，跳过法术
                            if spell is not None and (current_piece.action_points <= 0 or current_piece.spell_slots <= 0):
                                if MCTS_VERBOSE: print("[MCTS] 跳过法术：没有足够的资源")
                                continue
                                
                            action = ActionSet()
                            next_env = fork_environment(self.env)
                            remaining_points = current_piece.action_points
                            has_action = False  # 标记是否有任何动作
                            
                            # 设置移动
                            if move is not None and remaining_points > 0:
                                action.move = True
                                action.move_target = move
                                remaining_points -= 1
                                has_action = True
                                if MCTS_VERBOSE: print(f"[MCTS] 添加移动到 ({move.x}, {move.y})")
                            else:
                                action.move = False
                            
                            # 设置攻击
                            if target is not None and remaining_points > 0:
                                action.attack = True
                                action.attack_context = AttackContext()
                                action.attack_context.attacker = current_piece
                                action.attack_context.target = target
                                remaining_points -= 1
                                has_action = True
                                if MCTS_VERBOSE: print(f"[MCTS] 添加攻击目标 {target.id}")
                            else:
                                action.attack = False
                            
                            # 设置法术
                            if spell is not None and remaining_points > 0 and current_piece.spell_slots > 0:
                                # 获取法术可选目标
                                spell_targets = self.env.get_spell_targets(spell, current_piece)
                                if not spell_targets and not spell.is_area_effect:
                                    if MCTS_VERBOSE: print("[MCTS] 跳过法术：没有有效目标")
                                    continue
                                    
                                has_action = True
                                    
                                action.spell = True
                                action.spell_context = SpellContext()
                                action.spell_context.caster = current_piece
                                action.spell_context.spell = spell
                                if MCTS_VERBOSE: print(f"[MCTS] 添加法术 {spell.name}")
                                
                                # 设置目标和范围
                                if spell.is_area_effect:
                                    # 范围法术以当前位置为中心
                                    action.spell_context.target = None
                                    action.spell_context.target_area = Area(
                                        current_piece.position.x,
                                        current_piece.position.y,
                                        spell.area_radius
                                    )
                                else:
                                    # 单体法术选择最佳目标
                                    best_target = None
                                    if spell.effect_type in [SpellEffectType.DAMAGE, SpellEffectType.DEBUFF]:
                                        # 选择生命值最低的敌人
                                        best_target = min(spell_targets, key=lambda p: p.health)
                                    elif spell.effect_type in [SpellEffectType.HEAL, SpellEffectType.BUFF]:
                                        # 选择生命值损失最多的友军
                                        best_target = min(spell_targets, key=lambda p: p.health / p.max_health)
                                    elif spell.effect_type == SpellEffectType.MOVE:
                                        best_target = current_piece
                                        
                                    action.spell_context.target = best_target
                                    action.spell_context.target_area = Area(
                                        best_target.position.x,
                                        best_target.position.y,
                                        spell.area_radius
                                    )
                                    
                                remaining_points -= 1
                            else:
                                action.spell = False
                            
                            # 如果有行动点但没有执行任何动作，跳过这个组合
                            if current_piece.action_points > 0 and not has_action:
                                if MCTS_VERBOSE: print("[MCTS] 跳过：有行动点但未执行任何动作")
                                continue
                                
                            # 创建子节点并执行完整的步进
                            if MCTS_VERBOSE: print(f"[MCTS] 尝试动作: {action}")
                            step_with_action(next_env, action)
                            child = MCTSNode(next_env, self, action)
                            self.children.append(child)
                            if MCTS_VERBOSE: print(f"[MCTS] 成功添加子节点，当前共有 {len(self.children)} 个子节点")
                        
            def select(self) -> 'MCTSNode':
                """选择最有希望的子节点"""
                if not self.children:
                    return self
                    
                # UCB1公式选择节点
                def ucb1(node: MCTSNode) -> float:
                    if node.visits == 0:
                        return float('inf')
                    return node.value / node.visits + math.sqrt(2 * math.log(self.visits) / node.visits)
                    
                return max(self.children, key=ucb1)
                
            def simulate(self) -> float:
                """模拟到游戏结束或达到最大步数
                
                Returns:
                    float: 1.0 表示当前行动方胜利，-1.0 表示对手胜利，
                          如果达到最大步数，则根据双方棋子血量总和判断胜负
                """
                sim_env = fork_environment(self.env)
                max_steps = 50  # 最大模拟步数
                initial_team = sim_env.current_piece.team  # 记录当前行动方
                
                while not sim_env.is_game_over and max_steps > 0:
                    # 随机选择行动
                    legal_moves = get_legal_moves(sim_env)
                    attackable_targets = get_attackable_targets(sim_env)
                    
                    action = ActionSet()
                    
                    # 随机移动
                    if legal_moves and random.random() < 0.7:
                        action.move = True
                        action.move_target = random.choice(legal_moves)
                    else:
                        action.move = False
                        
                    # 随机攻击
                    if attackable_targets and random.random() < 0.8:
                        action.attack = True
                        action.attack_context = AttackContext()
                        action.attack_context.attacker = sim_env.current_piece
                        action.attack_context.target = random.choice(attackable_targets)
                    else:
                        action.attack = False
                        
                    action.spell = False
                    
                    # 执行模拟动作
                    step_with_action(sim_env, action)
                    max_steps -= 1
                
                # 如果游戏已经结束，直接根据胜负返回结果
                if sim_env.is_game_over:
                    team1_alive = any(p.is_alive for p in sim_env.player1.pieces)
                    team2_alive = any(p.is_alive for p in sim_env.player2.pieces)
                    if team1_alive and not team2_alive:
                        return 1.0 if initial_team == 1 else -1.0
                    elif team2_alive and not team1_alive:
                        return 1.0 if initial_team == 2 else -1.0
                    else:
                        return 0.0  # 平局
                
                # 如果没有执行任何动作，给予惩罚
                if not (action.move or action.attack or action.spell):
                    return -0.5  # 不行动的惩罚值
                
                # 如果达到最大步数，根据双方棋子血量总和判断
                team1_health = sum(p.health for p in sim_env.player1.pieces if p.is_alive)
                team2_health = sum(p.health for p in sim_env.player2.pieces if p.is_alive)
                
                if team1_health > team2_health:
                    return 1.0 if initial_team == 1 else -1.0
                elif team2_health > team1_health:
                    return 1.0 if initial_team == 2 else -1.0
                else:
                    return 0.0  # 血量相等，平局
                
            def backpropagate(self, value: float):
                """反向传播模拟结果"""
                node = self
                while node is not None:
                    node.visits += 1
                    node.value += value
                    node = node.parent
                    value = -value  # 对抗游戏中，父节点的收益是子节点的相反数
        
        def strategy(env: Environment) -> ActionSet:
            root = MCTSNode(env)
            
            # 运行MCTS
            for _ in range(simulation_count):
                node = root
                
                # 选择
                while node.children:
                    node = node.select()
                    
                # 扩展
                if node.visits > 0:
                    node.expand()
                    if node.children:
                        node = random.choice(node.children)
                        
                # 模拟
                value = node.simulate()
                
                # 反向传播
                node.backpropagate(value)
                
            # 选择访问次数最多的子节点对应的行动
            if not root.children:
                if MCTS_VERBOSE:
                    print("\n[MCTS] 警告: 没有生成任何子节点!")
                    print(f"[MCTS] 当前棋子: ID={env.current_piece.id if env.current_piece else None}")
                    print(f"[MCTS] 可移动位置数量: {len(get_legal_moves(env))}")
                    print(f"[MCTS] 可攻击目标数量: {len(get_attackable_targets(env))}")
                    print(f"[MCTS] 可用法术数量: {len(env.get_available_spells())}")
                    print(f"[MCTS] 当前行动点: {env.current_piece.action_points if env.current_piece else 0}")
                    print(f"[MCTS] 当前法术位: {env.current_piece.spell_slots if env.current_piece else 0}")
                return ActionSet()
                
            if MCTS_VERBOSE:
                print(f"\n[MCTS] 找到 {len(root.children)} 个可能的动作")
            best_child = max(root.children, key=lambda c: c.visits)
            if MCTS_VERBOSE:
                print(f"[MCTS] 选择最佳动作: 访问次数={best_child.visits}, 评分={best_child.value}")
                print(f"[MCTS] 动作详情:\n{best_child.action}")
            return best_child.action
        
        return strategy
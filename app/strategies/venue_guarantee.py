# -*- coding: utf-8 -*-
"""
app/strategies/venue_guarantee.py

会場別スロット保証コーディネーター (VenueGuaranteeCoordinator)
- 1日・各会場 1R以上の【実勝負】を確実に保証
- 1日・各会場 3R以上の【仮想】を確実に保証
"""

import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from app.core.database import get_supabase_client
from app.core.timezone import now_jst_naive, today_jst_str

# メモリキャッシュ: target_date -> {"real_guarantees": set(race_ids), "virtual_guarantees": set(race_ids)}
_venue_guarantee_cache: Dict[str, Dict[str, Set[str]]] = {}


def invalidate_venue_guarantee_cache(target_date: Optional[str] = None):
    """指定日付（または全日付）の保証キャッシュを破棄"""
    global _venue_guarantee_cache
    if target_date:
        _venue_guarantee_cache.pop(target_date, None)
    else:
        _venue_guarantee_cache.clear()


def resolve_venue_guarantees_for_date(
    target_date: str,
    supabase_client: Optional[Any] = None
) -> Tuple[Set[str], Set[str]]:
    """
    当日全レースから各会場ごとに：
    1. 最低1Rの【実勝負】スロット（各場最高安定レース）
    2. 最低3Rの【仮想】スロット（各場上位期待値レース）
    を特定して race_id のセットを返す。
    """
    if target_date in _venue_guarantee_cache:
        c = _venue_guarantee_cache[target_date]
        return c["real_guarantees"], c["virtual_guarantees"]

    supabase = supabase_client or get_supabase_client()
    if not target_date or not supabase:
        return set(), set()

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = res.data or []
        if not rows:
            return set(), set()

        # 会場コード (race_id[4:6]) ごとにレースを集約
        races_by_venue: Dict[str, List[dict]] = {}
        for r in rows:
            rid = str(r.get("race_id") or "")
            if len(rid) < 12:
                continue
            v_code = rid[4:6]
            dj = r.get("data_json") or {}
            r_name = str(dj.get("race_name") or "")
            c_det = str((dj.get("race_details") or {}).get("course_details") or "")

            # 障害戦・新馬戦は保証枠からも除外 (不確実性排除)
            if "障" in r_name or "障害" in c_det or "新馬" in r_name or "メイクデビュー" in r_name:
                continue

            horses = dj.get("horses") or []
            if len(horses) < 6:
                continue

            h_rec = dj.get("hybrid_recommendation") or {}

            # スコアリング
            s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
            w0 = s_win[0] if s_win else {}
            w0_top3 = float(w0.get("prob_top3", 0.0))
            w0_prob = float(w0.get("prob_win", 0.0))
            w0_odds = float(w0.get("odds") or 50.0)
            conf_score = float(h_rec.get("confidence_score") or 0.0)
            win_edge = float(h_rec.get("edge_ratio") or (w0_prob * w0_odds))

            item = {
                "race_id": rid,
                "venue_code": v_code,
                "h_rec": h_rec,
                "conf_score": conf_score,
                "win_edge": win_edge,
                "w0_top3": w0_top3,
                "stability_score": round(w0_top3 * 2.0 + conf_score + win_edge, 4),
                "bet_category": h_rec.get("bet_category", "skip"),
                "strategy_type": h_rec.get("strategy_type", "skip"),
            }

            if v_code not in races_by_venue:
                races_by_venue[v_code] = []
            races_by_venue[v_code].append(item)

        real_guarantees: Set[str] = set()
        virtual_guarantees: Set[str] = set()

        for v_code, r_items in races_by_venue.items():
            # 既存の自然合致チェック
            has_super = any(item["bet_category"] == "super" or item["strategy_type"] == "super_bet" for item in r_items)
            has_real = any(item["bet_category"] == "real" or item["strategy_type"] == "real_bet" for item in r_items)
            existing_virt_count = sum(1 for item in r_items if item["bet_category"] == "virtual" or item["strategy_type"] == "virtual_bet")

            # 1. 実勝負保証: この会場で【勝負R】も【実勝負】も0件の場合、安定度最上位1Rを保証昇格
            if not has_super and not has_real:
                # 安定度順にソート
                sorted_by_stab = sorted(r_items, key=lambda x: x["stability_score"], reverse=True)
                if sorted_by_stab:
                    promoted_real_rid = sorted_by_stab[0]["race_id"]
                    real_guarantees.add(promoted_real_rid)

            # 2. 仮想保証: この会場で【仮想】が3R未満の場合、残りのレースから上位を補充
            avail_for_virt = [item for item in r_items if item["race_id"] not in real_guarantees and item["bet_category"] not in ("super", "real")]
            avail_for_virt.sort(key=lambda x: (x["win_edge"] + x["conf_score"]), reverse=True)

            needed_virt = max(0, 3 - existing_virt_count)
            for item in avail_for_virt[:needed_virt]:
                virtual_guarantees.add(item["race_id"])

        _venue_guarantee_cache[target_date] = {
            "real_guarantees": real_guarantees,
            "virtual_guarantees": virtual_guarantees
        }
        return real_guarantees, virtual_guarantees

    except Exception as e:
        print(f"[VenueGuarantee] スロット保証解決エラー ({target_date}): {e}")
        return set(), set()


def get_race_guarantee_status(race_id: str, target_date: str) -> Tuple[bool, bool]:
    """指定レースが実勝負保証枠か、仮想保証枠かを判定 (is_real_guarantee, is_virtual_guarantee)"""
    if not race_id or not target_date:
        return False, False

    real_set, virt_set = resolve_venue_guarantees_for_date(target_date)
    return (race_id in real_set), (race_id in virt_set)

# -*- coding: utf-8 -*-
"""
app/strategies/venue_guarantee.py

会場別・日次スロット保証コーディネーター (VenueGuaranteeCoordinator)
- 実勝負保証: 当日全体で【勝負R】も【自然実勝負】も0件の日のみ、当日全場から最も安定度が高い「1Rだけ」を保証昇格 (少額300円)
- 仮想保証: 1日・各会場 3R以上の【仮想】を確実に保証
"""

import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from app.core.database import get_supabase_client
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
    当日全レースから：
    1. 【実勝負保証】当日全体で勝負Rも自然実勝負も0件の場合のみ、当日最高安定レース1R
    2. 【仮想保証】各会場ごとに最低3Rの仮想スロット
    を特定して race_id のセットを返す。
    """
    if target_date in _venue_guarantee_cache:
        c = _venue_guarantee_cache[target_date]
        return c["real_guarantees"], c["virtual_guarantees"]

    supabase = supabase_client or get_supabase_client()
    if not target_date or not supabase:
        return set(), set()

    try:
        scheduled_res = supabase.table("keiba_scheduled_races")\
            .select("race_id")\
            .eq("race_date", target_date)\
            .execute()
        scheduled_ids = {
            str(row.get("race_id") or "")
            for row in (scheduled_res.data or [])
            if row.get("race_id")
        }
        if not scheduled_ids:
            return set(), set()

        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = [
            row for row in (res.data or [])
            if str(row.get("race_id") or "") in scheduled_ids
        ]
        predicted_ids = {str(row.get("race_id") or "") for row in rows}
        if predicted_ids != scheduled_ids:
            return set(), set()

        all_day_races: List[dict] = []
        races_by_venue: Dict[str, List[dict]] = {}

        for r in rows:
            rid = str(r.get("race_id") or "")
            if len(rid) < 12:
                continue
            v_code = rid[4:6]
            dj = r.get("data_json") or {}
            r_name = str(dj.get("race_name") or "")
            c_det = str((dj.get("race_details") or {}).get("course_details") or "")

            # 障害戦は除外
            if "障" in r_name or "障害" in c_det:
                continue

            is_shinba = ("新馬" in r_name) or ("メイクデビュー" in r_name)
            horses = dj.get("horses") or []
            if len(horses) < 6:
                continue

            h_rec = dj.get("hybrid_recommendation") or {}

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
                "is_shinba": is_shinba,
                "stability_score": round(w0_top3 * 1.5 + conf_score + win_edge, 4),
                "bet_category": h_rec.get("bet_category", "skip"),
                "strategy_type": h_rec.get("strategy_type", "skip"),
            }

            all_day_races.append(item)
            races_by_venue.setdefault(v_code, []).append(item)

        real_guarantees: Set[str] = set()
        virtual_guarantees: Set[str] = set()

        # 当日全体での勝負R・自然実勝負の有無を判定
        has_any_super = any(item["bet_category"] == "super" or item["strategy_type"] == "super_bet" for item in all_day_races)
        has_any_real = any(item["bet_category"] == "real" or item["strategy_type"] == "real_bet" for item in all_day_races)

        # 1. 実勝負保証: 当日全体で勝負Rも自然実勝負も0件の場合のみ、当日最高安定レース1Rを保証昇格 (新馬戦は実馬券除外)
        if not has_any_super and not has_any_real and all_day_races:
            eligible_for_real_guarantee = [item for item in all_day_races if not item["is_shinba"]]
            sorted_by_stab = sorted(eligible_for_real_guarantee, key=lambda x: x["stability_score"], reverse=True)
            if sorted_by_stab:
                real_guarantees.add(sorted_by_stab[0]["race_id"])

        # 2. 仮想保証: 各会場ごとに最低3Rを確保
        for v_code, r_items in races_by_venue.items():
            existing_participation_count = sum(
                1 for item in r_items
                if item["race_id"] in real_guarantees
                or item["bet_category"] in ("super", "real", "virtual")
                or item["strategy_type"] in ("super_bet", "real_bet", "virtual_bet")
            )
            avail_for_virt = [
                item for item in r_items
                if item["race_id"] not in real_guarantees
                and item["bet_category"] not in ("super", "real", "virtual")
                and item["strategy_type"] not in ("super_bet", "real_bet", "virtual_bet")
            ]
            avail_for_virt.sort(key=lambda x: (x["win_edge"] + x["conf_score"]), reverse=True)

            needed_virt = max(0, 3 - existing_participation_count)
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

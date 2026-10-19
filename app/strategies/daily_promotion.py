# -*- coding: utf-8 -*-
"""
app/strategies/daily_promotion.py

本日厳選No.1: デイリー昇格枠 (自然合致0件時専用フォールバック)
"""

from typing import Dict, Any, List, Optional, Callable
from app.core.database import get_supabase_client
from app.core.timezone import now_jst_naive, today_jst_str, parse_post_datetime
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class DailyPromotionStrategy(BaseBettingStrategy):
    """
    【本日厳選No.1】デイリー昇格枠
    - 自然合致レース (B/A/C/A-2) が当日0件の時のみ選定
    - 当日全レースの中で「確信度 × エッジ」最高スコアのレースを実戦昇格
    - 馬場・オッズ適性に応じた単勝＋複勝の配分 (計1,000円)
    """

    @property
    def strategy_type(self) -> str:
        return "daily_top"

    @property
    def priority(self) -> int:
        return 99  # フォールバック枠

    def evaluate_eligibility(
        self,
        race_info: Dict[str, Any],
        honmei_horse: Dict[str, Any],
        horses: List[Dict[str, Any]],
        confidence_score: float,
        win_edge: float,
        prob_gachi: float,
        surface_type: int,
        is_summer: bool,
    ) -> bool:
        # StrategyArbiter によって明示的に指定された場合のみ True
        return bool(race_info.get("is_daily_top_race", False))

    def generate_tickets(
        self,
        race_info: Dict[str, Any],
        honmei_horse: Dict[str, Any],
        horses: List[Dict[str, Any]],
        confidence_score: float,
        win_edge: float,
        bias_boost_mult: float,
        bias_tag: str,
        suffix: str,
    ) -> StrategyDecision:
        w0_u = int(honmei_horse.get("umaban", 0))
        w0_name = str(honmei_horse.get("horse_name", f"馬{w0_u}"))
        w0_odds = float(honmei_horse.get("odds") or 50.0)
        surface_type = int(race_info.get("surface_type", 0))
        prob_gachi = float(race_info.get("prob_gachi", 0.0))

        if surface_type == 1:
            d_tan, d_fuku = 700, 300
            strategy_name = f"⭐ 本日厳選No.1: ダート適性単複 (単{d_tan}円＋複{d_fuku}円)"
            real_bet_badge = f"⭐【本日厳選No.1】ダート適性単複{bias_tag}{suffix}"
        elif w0_odds >= 5.0 or prob_gachi <= 0.15:
            d_tan, d_fuku = 500, 500
            strategy_name = f"⭐ 本日厳選No.1: 芝妙味適性単複 (単{d_tan}円＋複{d_fuku}円)"
            real_bet_badge = f"⭐【本日厳選No.1】芝妙味単複{bias_tag}{suffix}"
        else:
            d_tan, d_fuku = 700, 300
            strategy_name = f"⭐ 本日厳選No.1: 芝本命適性単複 (単{d_tan}円＋複{d_fuku}円)"
            real_bet_badge = f"⭐【本日厳選No.1】芝本命単複{bias_tag}{suffix}"

        tickets: List[BetTicket] = []

        # 単勝
        tickets.append(BetTicket(
            ticket_type="単勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=d_tan,
            odds=w0_odds,
            expected_payout=int(round(d_tan * w0_odds)),
            is_real_odds=True,
            memo=f"単勝(デイリー厳選) {w0_u}番 ({w0_name}){bias_tag} {d_tan:,}円"
        ))

        # 複勝
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        tickets.append(BetTicket(
            ticket_type="複勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=d_fuku,
            odds=est_fuku_odds,
            expected_payout=int(round(d_fuku * est_fuku_odds)),
            is_real_odds=False,
            memo=f"複勝(デイリー厳選) {w0_u}番 ({w0_name}){bias_tag} {d_fuku:,}円"
        ))

        total_amount = sum(t.amount for t in tickets)
        summary = f"{strategy_name} [単勝{d_tan:,}円 ＋ 複勝{d_fuku:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            assigned_tier="Daily_No1",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            real_bet_level="standard",
            tickets=tickets,
            total_amount=total_amount,
            confidence_score=confidence_score,
            edge_ratio=round(win_edge, 2),
            synth_odds=round(w0_odds, 1),
            summary=summary,
            reason="本日厳選No.1昇格枠採択 (自然合致0件時リカバリ)"
        )

# =========================================================================
# ★ デイリー昇格調停オーケストレーション (Coordinator)
# =========================================================================

NATURAL_REAL_STRATEGIES = ("strategy_b", "strategy_a", "strategy_c", "strategy_a_hole")

_daily_top_race_id_cache: Dict[str, str] = {}
_cache_invalidation_handler: Optional[Callable[[str], None]] = None


def set_daily_promotion_dependencies(cache_invalidation_handler: Optional[Callable[[str], None]] = None):
    """外部キャッシュ無効化ハンドラーの登録"""
    global _cache_invalidation_handler
    if cache_invalidation_handler is not None:
        _cache_invalidation_handler = cache_invalidation_handler


def invalidate_daily_top_cache(target_date: str):
    """指定日付のデイリー厳選キャッシュを破棄"""
    _daily_top_race_id_cache.pop(target_date, None)
    if _cache_invalidation_handler:
        try:
            _cache_invalidation_handler(target_date)
        except Exception:
            pass


def get_or_select_daily_top_race_id(target_date: str, supabase_client: Optional[Any] = None) -> Optional[str]:
    """
    当日(target_date)全レースの中で、自然条件（戦略B/A/C/A-2）を満たすレースが0レースのときのみ、
    『確信度 × エッジ適性スコア』が最も高い1レースの race_id を特定して返す。
    自然条件を満たすレースが1つ以上存在する場合は None を返す（全合致レースをそのまま実馬券とする）。
    """
    supabase = supabase_client or get_supabase_client()
    if not target_date or not supabase:
        return None

    now_t = now_jst_naive()
    is_today = (target_date == today_jst_str())

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = res.data or []
        if not rows:
            return None

        # 自然条件（戦略B / A / C / A-2）を満たすレースが1つでもあれば、昇格枠は不要
        existing_daily_top_rid = None
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            h_rec = dj.get("hybrid_recommendation") or {}
            if h_rec.get("strategy_type") in NATURAL_REAL_STRATEGIES:
                _daily_top_race_id_cache.pop(target_date, None)
                return None
            if h_rec.get("strategy_type") == "daily_top" and h_rec.get("is_real_bet_target"):
                existing_daily_top_rid = rid

        # 既に daily_top に昇格済みのレースがあれば、再推論・発走後もそのレースを維持する
        if existing_daily_top_rid:
            _daily_top_race_id_cache[target_date] = existing_daily_top_rid
            return existing_daily_top_rid

        candidates = []
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            r_name = str(dj.get("race_name") or "")
            c_det = str((dj.get("race_details") or {}).get("course_details") or "")
            if "障" in r_name or "障害" in c_det or "新馬" in r_name or "メイクデビュー" in r_name:
                continue
            horses = dj.get("horses") or []
            if len(horses) < 8:
                continue

            # 当日の場合は未発走（発走時刻前）のレースを優先対象とする
            is_unrun = True
            if is_today:
                p_time_str = (dj.get("race_details") or {}).get("post_time", "")
                post_dt = parse_post_datetime(p_time_str, target_date)
                if post_dt and now_t >= post_dt:
                    is_unrun = False

            h_rec = dj.get("hybrid_recommendation") or {}
            conf = float(h_rec.get("confidence_score") or 0.0)
            edge = float(h_rec.get("edge_ratio") or 1.0)
            aptitude_score = conf * min(max(edge, 0.8), 2.5)

            candidates.append({
                "race_id": rid,
                "is_unrun": is_unrun,
                "confidence_score": conf,
                "edge_ratio": edge,
                "aptitude_score": aptitude_score
            })

        if not candidates:
            return None

        unrun_candidates = [c for c in candidates if c["is_unrun"]]
        pool = unrun_candidates if unrun_candidates else candidates
        pool.sort(key=lambda x: (x["aptitude_score"], x["confidence_score"]), reverse=True)
        chosen_rid = pool[0]["race_id"]

        _daily_top_race_id_cache[target_date] = chosen_rid
        return chosen_rid
    except Exception as e:
        print(f"[get_or_select_daily_top_race_id] エラー: {e}")
        return None


def ensure_daily_single_real_bet(target_date: str, supabase_client: Optional[Any] = None) -> Optional[str]:
    """
    【実馬券厳選ルール】
    1. 戦略B（勝負馬券）および戦略A・C・A-2（実馬券1〜3）を満たすレースはすべて実馬券（is_real_bet_target = True）とする。
    2. 上記を満たすレースが0レースのときのみ、当日全レースの中で『確信度 × エッジ適性スコア』が最も高い1レースを実馬券に昇格する。
    """
    supabase = supabase_client or get_supabase_client()
    if not target_date or not supabase:
        return None

    now_t = now_jst_naive()
    is_today = (target_date == today_jst_str())

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = res.data or []
        if not rows:
            return None

        # 当日公式スケジュールに登録されているレースIDのみを厳密に対象とする (過去テストデータの混入を物理遮断)
        if target_date == "2026-10-18":
            rows = [r for r in rows if str(r.get("race_id") or "")[4:6] in {"05", "08", "04"}]
        else:
            sched_res = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", target_date).execute()
            valid_scheduled_rids = {str(sr["race_id"]) for sr in (sched_res.data or []) if sr.get("race_id")}
            if valid_scheduled_rids:
                rows = [r for r in rows if str(r.get("race_id") or "") in valid_scheduled_rids]

        natural_real_rids = []
        existing_daily_top_rid = None
        # Step 1: 過去に「参考推奨」へ格下げされていた自然合致レース（戦略B/A/C/A-2）があれば全て実馬券に復元
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            h_rec = dj.get("hybrid_recommendation") or {}
            st_type = h_rec.get("strategy_type")
            if st_type in NATURAL_REAL_STRATEGIES:
                natural_real_rids.append(rid)
                if not h_rec.get("is_real_bet_target") or "参考推奨" in str(h_rec.get("real_bet_badge") or ""):
                    h_rec["is_participate"] = True
                    h_rec["is_real_bet_target"] = True
                    h_rec["real_bet_level"] = "high" if st_type == "strategy_b" else "standard"
                    badge_map = {
                        "strategy_b": "🔥【超勝負R】単勝＋馬単レバレッジ",
                        "strategy_a": "⭐【芝厳選】単複ケリー傾斜",
                        "strategy_c": "🛡️【ダート本命】手堅い単複防衛",
                        "strategy_a_hole": "⚡【芝波乱妙味】単複好配当狙い"
                    }
                    h_rec["real_bet_badge"] = badge_map.get(st_type, "⭐【実馬券推奨】")
                    dj["hybrid_recommendation"] = h_rec
                    supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", rid).execute()
            elif st_type == "daily_top" and h_rec.get("is_real_bet_target"):
                existing_daily_top_rid = rid

        # Step 2: 自然合致レースが1レース以上存在する場合は、不要な daily_top 昇格枠を解除して終了
        if natural_real_rids:
            for r in rows:
                rid = str(r.get("race_id") or "")
                dj = r.get("data_json") or {}
                h_rec = dj.get("hybrid_recommendation") or {}
                if h_rec.get("strategy_type") == "daily_top":
                    h_rec["is_participate"] = False
                    h_rec["is_real_bet_target"] = False
                    h_rec["real_bet_level"] = "none"
                    h_rec["real_bet_badge"] = None
                    h_rec["strategy_type"] = "skip"
                    h_rec["strategy_name"] = "見送り推奨 (資金保全)"
                    h_rec["tansho_bets"] = []
                    h_rec["fukusho_bets"] = []
                    h_rec["tansho_amount"] = 0
                    h_rec["fukusho_amount"] = 0
                    h_rec["total_amount"] = 0
                    dj["hybrid_recommendation"] = h_rec
                    rec_obj = dj.get("recommendation") or {}
                    rec_obj["is_pass"] = True
                    rec_obj["grade"] = "D"
                    rec_obj["confidence_label"] = "見送り推奨"
                    rec_obj["tansho_amount"] = 0
                    rec_obj["fukusho_amount"] = 0
                    rec_obj["total_amount"] = 0
                    rec_obj["summary"] = "見送り推奨 (資金保全)"
                    dj["recommendation"] = rec_obj
                    supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", rid).execute()
            invalidate_daily_top_cache(target_date)
            return natural_real_rids[0]

        # Step 3: 自然合致が0レースのときのみ、確信度 × エッジ適性スコア 最高位の1レースを実馬券に昇格
        candidates = []
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            r_name = str(dj.get("race_name") or "")
            c_det = str((dj.get("race_details") or {}).get("course_details") or "")
            if "障" in r_name or "障害" in c_det or "新馬" in r_name or "メイクデビュー" in r_name:
                continue
            horses = dj.get("horses") or []
            if len(horses) < 8:
                continue

            is_unrun = True
            if is_today:
                p_time_str = (dj.get("race_details") or {}).get("post_time", "")
                post_dt = parse_post_datetime(p_time_str, target_date)
                if post_dt and now_t >= post_dt:
                    is_unrun = False

            h_rec = dj.get("hybrid_recommendation") or {}
            conf = float(h_rec.get("confidence_score") or 0.0)
            edge = float(h_rec.get("edge_ratio") or 1.0)
            score = conf * min(max(edge, 0.8), 2.5)
            candidates.append({
                "race_id": rid,
                "data_json": dj,
                "is_unrun": is_unrun,
                "confidence_score": conf,
                "edge_ratio": edge,
                "score": score
            })

        if not candidates:
            return None

        # 既に daily_top に設定済みのレースがあれば、勝手に他レースへ移らないよう維持
        existing_cand = next((c for c in candidates if c["race_id"] == existing_daily_top_rid), None) if existing_daily_top_rid else None
        if existing_cand:
            top_race = existing_cand
        else:
            unrun_candidates = [c for c in candidates if c["is_unrun"]]
            pool = unrun_candidates if unrun_candidates else candidates
            pool.sort(key=lambda x: (x["score"], x["confidence_score"]), reverse=True)
            top_race = pool[0]

        top_rid = top_race["race_id"]
        top_dj = top_race["data_json"]

        # 本命馬（◎ または prob_win 最大馬）を特定
        horses = top_dj.get("horses") or []
        honmei_horse = next((h for h in horses if h.get("mark") == "◎"), None)
        if not honmei_horse:
            honmei_horse = max(horses, key=lambda x: x.get("prob_win", 0.0)) if horses else {}
        w0_u = int(honmei_horse.get("umaban") or 1)
        w0_name = str(honmei_horse.get("horse_name") or "")
        w0_odds = float(honmei_horse.get("odds") or 3.0)
        c_det = str((top_dj.get("race_details") or {}).get("course_details") or "")
        is_dirt = ("ダ" in c_det) or ("ダート" in c_det)

        h_rec = top_dj.get("hybrid_recommendation") or {}
        h_rec["is_participate"] = True
        h_rec["is_real_bet_target"] = True
        h_rec["real_bet_level"] = "standard"
        h_rec["strategy_type"] = "daily_top"
        h_rec["reason"] = "戦略Daily_No1採択"
        h_rec["confidence_score"] = top_race["confidence_score"]
        h_rec["edge_ratio"] = top_race["edge_ratio"]

        tansho_bets = []
        fukusho_bets = []
        bet_amount = 300
        h_rec["strategy_name"] = f"🔰【実勝負(保証)】日次1R参加・少額防衛 (複勝{bet_amount:,}円)"
        h_rec["real_bet_badge"] = "🔰【実勝負(保証)】複勝 (0R時昇格)"

        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        fukusho_bets.append({
            "ticket_type": "複勝",
            "combination": [w0_u],
            "combination_str": f"{w0_u}",
            "amount": bet_amount,
            "odds": est_fuku_odds,
            "expected_payout": int(round(bet_amount * est_fuku_odds)),
            "is_real_odds": False,
            "memo": f"複勝(デイリー厳選) {w0_u}番 ({w0_name}) {bet_amount:,}円"
        })

        h_rec["tansho_bets"] = []
        h_rec["fukusho_bets"] = fukusho_bets
        h_rec["tansho_amount"] = 0
        h_rec["fukusho_amount"] = bet_amount
        h_rec["umatan_amount"] = 0
        h_rec["total_amount"] = bet_amount
        h_rec["summary"] = f"{h_rec['strategy_name']}：複勝{bet_amount:,}円"
        h_rec["is_virtual_bet_target"] = True
        h_rec["virtual_bets"] = tansho_bets + fukusho_bets
        h_rec["virtual_total_amount"] = d_tan + d_fuku
        h_rec["virtual_strategy_name"] = h_rec["strategy_name"]
        h_rec["virtual_badge"] = h_rec["real_bet_badge"]
        top_dj["hybrid_recommendation"] = h_rec

        top_rec = top_dj.get("recommendation") or {}
        top_rec["target_umaban"] = w0_u
        top_rec["horse_name"] = w0_name
        top_rec["is_pass"] = False
        top_rec["grade"] = "A"
        top_rec["confidence_label"] = "厳選勝負"
        top_rec["tansho_amount"] = d_tan
        top_rec["fukusho_amount"] = d_fuku
        top_rec["total_amount"] = d_tan + d_fuku
        top_rec["summary"] = h_rec["summary"]
        top_dj["recommendation"] = top_rec

        supabase.table("keiba_predictions_cache").update({"data_json": top_dj}).eq("race_id", top_rid).execute()
        try:
            supabase.table("keiba_scheduled_races").update({"is_predicted": 1}).eq("race_id", top_rid).eq("is_predicted", 0).execute()
        except Exception:
            pass
        print(f"[DailySingleRealBet] {target_date}: 自然合致0Rのため厳選No.1レースとして {top_rid} (◎{w0_u}番 {w0_name}) を昇格しました。")

        # 他の daily_top 昇格レースがあれば解除
        for r in rows:
            r_rid = str(r.get("race_id") or "")
            if r_rid != top_rid:
                r_dj = r.get("data_json") or {}
                r_h = r_dj.get("hybrid_recommendation") or {}
                if r_h.get("strategy_type") == "daily_top" and r_h.get("is_real_bet_target"):
                    r_h["is_participate"] = False
                    r_h["is_real_bet_target"] = False
                    r_h["real_bet_level"] = "none"
                    r_h["real_bet_badge"] = None
                    r_h["strategy_type"] = "skip"
                    r_h["tansho_bets"] = []
                    r_h["fukusho_bets"] = []
                    r_h["tansho_amount"] = 0
                    r_h["fukusho_amount"] = 0
                    r_h["total_amount"] = 0
                    r_dj["hybrid_recommendation"] = r_h
                    supabase.table("keiba_predictions_cache").update({"data_json": r_dj}).eq("race_id", r_rid).execute()

        _daily_top_race_id_cache[target_date] = top_rid
        if _cache_invalidation_handler:
            try:
                _cache_invalidation_handler(target_date)
            except Exception:
                pass
        return top_rid
    except Exception as e:
        print(f"[ensure_daily_single_real_bet] エラー: {e}")
        return None


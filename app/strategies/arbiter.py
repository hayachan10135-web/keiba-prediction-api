# -*- coding: utf-8 -*-
"""
app/strategies/arbiter.py

戦略調停オーケストレーター (StrategyArbiter)
カスケード評価、スリッページ停止判定、トリガミ間引き、仮想観戦枠の生成を統合
"""

import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.config import settings
from app.core.timezone import now_jst_naive, today_jst_str
from app.strategies.base import BaseBettingStrategy
from app.strategies.strategy_super import StrategySuper
from app.strategies.strategy_real import StrategyReal
from app.strategies.strategy_virtual import StrategyVirtual
from app.strategies.strategy_b import StrategyB
from app.strategies.strategy_a import StrategyA
from app.strategies.strategy_c import StrategyC
from app.strategies.strategy_a_hole import StrategyAHole
from app.strategies.daily_promotion import DailyPromotionStrategy
from app.strategies.trigami_pruner import TrigamiPruner
from app.strategies.models import StrategyDecision, BetTicket


class StrategyArbiter:
    """
    全買い目戦略を統括する調停オーケストレーター (3段階レース区分: 勝負R / 実勝負 / 仮想 + 見送り)
    """

    def __init__(self):
        # 3段階区分戦略を優先度順に登録
        self.strategies: List[BaseBettingStrategy] = [
            StrategySuper(),           # 1. ⭐【勝負R】 (最優先: 芝オッズ妙味帯・馬単3点)
            StrategyReal(),            # 2. 🎯【実勝負】 (各場1R保証枠 または 高安定本命・単複)
            StrategyVirtual(),         # 3. 🎮【仮想】   (各場3R保証枠 または 観戦枠・3連複6点)
        ]

    def evaluate_race(
        self,
        race_name: str,
        race_details: Optional[dict],
        horses: List[dict],
        surface_type: int,
        prob_gachi: float,
        prob_himo: float,
        opponents: List[tuple],
        u0: int,
        honmei_horse: Optional[dict],
        race_date_str: str,
        force_final_odds: bool = False,
        real_trio_odds: Optional[dict] = None,
        track_bias_data: Optional[dict] = None,
        is_daily_top_race: bool = False,
        is_real_guarantee: bool = False,
        is_virtual_guarantee: bool = False,
    ) -> Dict[str, Any]:
        """
        compute_hybrid_rec_from_inputs と完全互換のハイブリッド推奨辞書を生成
        (3段階レース区分: 勝負R / 実勝負 / 仮想 + 見送り)
        """
        course_details_str = (race_details or {}).get("course_details", "") if isinstance(race_details, dict) else ""
        if "障害" in race_name or "障" in course_details_str:
            surface_type = 2
        is_shinba = ("新馬" in race_name) or ("メイクデビュー" in race_name)

        has_enough_horses = len(horses) >= 6
        is_valid_race_type = (surface_type != 2) and (not is_shinba) and has_enough_horses

        # AI勝率順ソート (Model B)
        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        w0 = s_win[0] if s_win else {}
        w1 = s_win[1] if len(s_win) > 1 else w0

        # AI複勝率順ソート (Model B)
        s_top3 = sorted(horses, key=lambda x: x.get("prob_top3", 0.0), reverse=True)
        t0 = s_top3[0] if s_top3 else {}
        t1 = s_top3[1] if len(s_top3) > 1 else t0
        t2 = s_top3[2] if len(s_top3) > 2 else t1

        w0_prob = float(w0.get("prob_win", 0.0))
        w1_prob = float(w1.get("prob_win", 0.0))
        w0_odds = float(w0.get("odds") or 50.0)
        w0_u = int(w0.get("umaban", 0))
        w0_name = str(w0.get("horse_name", f"馬{w0_u}"))

        t0_prob = float(t0.get("prob_top3", 0.0))
        t1_prob = float(t1.get("prob_top3", 0.0))
        t2_prob = float(t2.get("prob_top3", 0.0))

        win_margin = w0_prob - w1_prob
        win_edge = w0_prob * w0_odds
        top3_margin_0_1 = t0_prob - t1_prob
        top3_margin_1_2 = t1_prob - t2_prob

        confidence_score = round(
            win_margin * 2.0 +
            top3_margin_0_1 +
            top3_margin_1_2 +
            float(np.clip(win_edge - 1.0, -0.5, 0.5)),
            4
        )
        edge_ratio = round(win_edge, 2)

        # トラックバイアス判定 (表示用タグのみ維持し、賭け金ブーストは一旦停止)
        w0_bias = str(w0.get("bias_match") or "")
        is_bias_favorable = any(fav in w0_bias for fav in ("絶好バイアス", "有利", "合致"))
        is_bias_adverse = "逆風" in w0_bias
        bias_boost_mult = 1.0
        bias_tag = " [⚡バイアス追い風]" if is_bias_favorable else (" [⚠️逆風]" if is_bias_adverse else "")

        # 確定オッズ・発走前判定 (Stage 5)
        is_final_odds = force_final_odds
        if not is_final_odds and race_details:
            post_time_str = race_details.get("post_time", "")
            import re
            m_time = re.search(r"(\d{1,2}):(\d{2})", post_time_str)
            if m_time:
                try:
                    hh, mm = int(m_time.group(1)), int(m_time.group(2))
                    now_dt = now_jst_naive()
                    r_date_str = race_date_str or today_jst_str()
                    d_parts = [int(p) for p in r_date_str.split("-")]
                    post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hh, mm)
                    if now_dt >= post_dt - datetime.timedelta(minutes=7):
                        is_final_odds = True
                except Exception:
                    pass

        suffix = " (確定)" if is_final_odds else " (暫定)"

        # 季節判定 (夏競馬: 6〜9月)
        try:
            month_num = int(race_date_str[5:7]) if (race_date_str and len(race_date_str) >= 7 and race_date_str[5:7].isdigit()) else now_jst_naive().month
        except Exception:
            month_num = now_jst_naive().month
        is_summer = 1 if 6 <= month_num <= 9 else 0

        race_info = {
            "race_name": race_name,
            "surface_type": surface_type,
            "prob_gachi": prob_gachi,
            "is_daily_top_race": is_daily_top_race,
            "is_real_guarantee": is_real_guarantee,
            "is_virtual_guarantee": is_virtual_guarantee,
        }

        decision: Optional[StrategyDecision] = None

        if is_valid_race_type:
            # 優先度順に戦略を評価 (勝負R -> 実勝負 -> 仮想)
            for strat in self.strategies:
                if strat.evaluate_eligibility(
                    race_info=race_info,
                    honmei_horse=w0,
                    horses=horses,
                    confidence_score=confidence_score,
                    win_edge=win_edge,
                    prob_gachi=prob_gachi,
                    surface_type=surface_type,
                    is_summer=bool(is_summer),
                ):
                    decision = strat.generate_tickets(
                        race_info=race_info,
                        honmei_horse=w0,
                        horses=horses,
                        confidence_score=confidence_score,
                        win_edge=win_edge,
                        bias_boost_mult=bias_boost_mult,
                        bias_tag=bias_tag,
                        suffix=suffix,
                    )
                    break

        if not decision or not decision.is_participate:
            # 見送り判定
            if not has_enough_horses:
                s_reason = f"出走頭数不足 ({len(horses)}頭 < 6頭のため見送り)"
            elif surface_type == 2:
                s_reason = "障害戦除外 (不確実性高のため見送り)"
            elif is_shinba:
                s_reason = "新馬戦除外 (過去走データなしのため見送り)"
            else:
                s_reason = "3区分選定基準不一致 (資金保全のため見送り)"

            decision = StrategyDecision(
                is_participate=False,
                is_real_bet_target=False,
                is_virtual_bet=False,
                bet_category="skip",
                assigned_tier=None,
                strategy_type="skip",
                strategy_name="見送り推奨 (資金保全)",
                real_bet_badge=None,
                badge_text="見送り推奨",
                real_bet_level="none",
                skip_reason=s_reason,
                tickets=[],
                total_amount=0,
                confidence_score=confidence_score,
                edge_ratio=edge_ratio,
                synth_odds=round(w0_odds, 1),
                summary=f"見送り推奨 ({s_reason})",
                reason=s_reason
            )

        # 券種別リストの分解
        tansho_bets = [t.to_dict() for t in decision.tickets if t.ticket_type == "単勝"]
        fukusho_bets = [t.to_dict() for t in decision.tickets if t.ticket_type == "複勝"]
        umatan_bets = [t.to_dict() for t in decision.tickets if t.ticket_type == "馬単"]
        umaren_bets = [t.to_dict() for t in decision.tickets if t.ticket_type == "馬連"]
        trio_bets = [t.to_dict() for t in decision.tickets if t.ticket_type == "3連複"]
        sanrentan_bets = [t.to_dict() for t in decision.tickets if t.ticket_type == "3連単"]

        tan_amt = sum(b["amount"] for b in tansho_bets)
        fuku_amt = sum(b["amount"] for b in fukusho_bets)
        utan_amt = sum(b["amount"] for b in umatan_bets)
        total_amt = decision.total_amount

        # 仮想馬券 (総合台帳用) の生成
        virtual_bets = []
        virtual_total_amount = 0
        virtual_strategy_name = None
        virtual_badge = None

        if decision.bet_category in ("super", "real"):
            # 実投票対象 (勝負R または 実勝負) -> 総合台帳にもそのまま反映
            virtual_bets = [t.to_dict() for t in decision.tickets]
            virtual_total_amount = total_amt
            virtual_strategy_name = decision.strategy_name
            virtual_badge = decision.real_bet_badge
        elif decision.bet_category == "virtual":
            # 仮想観戦枠 -> 3連複チケットを仮想馬券として登録
            virtual_bets = trio_bets
            virtual_total_amount = total_amt
            virtual_strategy_name = decision.strategy_name
            virtual_badge = decision.real_bet_badge

        all_active_bets = [t.to_dict() for t in decision.tickets]

        return {
            "is_participate": decision.is_participate,
            "strategy_name": decision.strategy_name,
            "strategy_type": decision.strategy_type,
            "bet_category": decision.bet_category,
            "badge_text": decision.badge_text,
            "total_amount": total_amt,
            "summary": decision.summary,
            "tansho_bets": tansho_bets,
            "fukusho_bets": fukusho_bets,
            "umatan_bets": umatan_bets,
            "umaren_bets": umaren_bets,
            "trio_bets": trio_bets,
            "sanrentan_bets": sanrentan_bets,
            "tansho_amount": tan_amt,
            "fukusho_amount": fuku_amt,
            "umatan_amount": utan_amt,
            "hypothetical_umaren_bets": [],
            "trio_bet": trio_bets[0] if trio_bets else None,
            "reason": decision.reason,
            "is_real_odds_applied": any(b.get("is_real_odds", False) for b in all_active_bets),
            "total_expected_payout": sum(b.get("expected_payout", 0) for b in all_active_bets),
            "prob_gachi": prob_gachi,
            "prob_himo": prob_himo,
            "edge_ratio": edge_ratio,
            "synth_odds": round(w0_odds, 1),
            "confidence_score": confidence_score,
            "is_real_bet_target": decision.is_real_bet_target,
            "is_virtual_bet": decision.is_virtual_bet,
            "real_bet_badge": decision.real_bet_badge,
            "real_bet_level": decision.real_bet_level,
            "is_final_odds": is_final_odds,
            # 仮想馬券フィールド
            "is_virtual_bet_target": (decision.bet_category in ("super", "real", "virtual")),
            "virtual_bet_slot": decision.bet_category if decision.bet_category in ("super", "real", "virtual") else None,
            "virtual_strategy_name": virtual_strategy_name,
            "virtual_badge": virtual_badge,
            "virtual_bets": virtual_bets,
            "virtual_total_amount": virtual_total_amount
        }


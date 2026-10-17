# -*- coding: utf-8 -*-
"""
app/strategies/trigami_pruner.py

トリガミ検知 ＆ 券種動的間引きエンジン
要件 FR-STR-08: 各券種の下限オッズで 単独的中払戻金 < 総投資額 となる目のみを除外
"""

from typing import List, Tuple
from app.strategies.models import BetTicket


class TrigamiPruner:
    """
    トリガミとなる券種を事前に間引くフィルタ
    """

    @classmethod
    def prune(cls, tickets: List[BetTicket], total_amount: int) -> Tuple[List[BetTicket], List[str]]:
        """
        チケットリストを精査し、単独的中で投資総額を下回るトリガミチケットを除外する。
        戻り値: (間引き後の有効チケットリスト, 間引かれた理由のログリスト)
        """
        if not tickets or total_amount <= 0:
            return tickets, []

        survived_tickets: List[BetTicket] = []
        pruned_logs: List[str] = []

        for t in tickets:
            # 単独的中時の払戻額 = amount * odds
            # ※ odds が下限オッズ (単勝オッズや複勝下限オッズ)
            payout = int(t.amount * t.odds)
            if payout < total_amount:
                # トリガミ認定 -> 間引き
                reason = f"【トリガミ間引き】{t.ticket_type}{t.combination_str} (オッズ{t.odds}倍 × {t.amount:,}円 = {payout:,}円 < 投資総額{total_amount:,}円)"
                pruned_logs.append(reason)
            else:
                survived_tickets.append(t)

        return survived_tickets, pruned_logs


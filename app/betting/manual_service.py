# -*- coding: utf-8 -*-
"""
app/betting/manual_service.py

新要件 FR-BET-07: マニュアル（手動・実機テスト）投票サービス
★【厳格な集計除外規定】: is_manual_bet=True で記録し、AI収支集計から物理的に除外する
"""

import time
import sys
from typing import Dict, Any, List, Optional
from app.core.database import get_supabase_client
from app.betting.factory import BettingClientFactory


def execute_manual_bet(
    race_id: str,
    venue_name: str,
    race_no: int,
    items: List[Dict[str, Any]],
    mode: Optional[str] = "ipat",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    手動または実機テスト用の馬券発注を実行し、keiba_real_bets に is_manual_bet=True で永続化する。
    AIの収支計算（日次サマリー・累計サマリー・回収率）からは完全に除外されます。
    """
    if not items:
        return {"status": "skipped", "error": "買い目が指定されていません"}

    total_amount = sum(int(item.get("amount", 0)) for item in items)
    if total_amount < 100:
        return {"status": "skipped", "error": "投資総額が100円未満です"}

    client = BettingClientFactory.get_client(mode)
    print(f"[ManualBetService] [REQ] マニュアル実機投票リクエスト: {venue_name}{race_no}R ({race_id}) {len(items)}件 計{total_amount:,}円 (モード: {client.platform_name}, dry_run={dry_run})")

    # 1. クライアント経由で発注執行
    bet_res = client.execute_bets(venue_name, race_no, items, dry_run=dry_run)

    # 2. 結果のDB永続化 (is_manual_bet=True を明示)
    supabase = get_supabase_client()
    if supabase and bet_res.get("status") in ("success", "dry_run_success"):
        try:
            receipt_no = bet_res.get("receipt_no", "")
            record = {
                "race_id": race_id,
                "platform": client.platform_name,
                "receipt_no": receipt_no,
                "total_amount": total_amount,
                "items": items,
                "status": bet_res.get("status", "success"),
                "is_manual_bet": True,  # ★ AI収支集計から完全除外するフラグ
                "bet_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            supabase.table("keiba_real_bets").insert(record).execute()
            print(f"[ManualBetService] [SAVED] マニュアル投票台帳に記録完了 (is_manual_bet=True, 受付番号: {receipt_no})")
        except Exception as e:
            print(f"[ManualBetService] [WARN] 台帳永続化エラー (発注自体は完了): {e}", file=sys.stderr)

    # 3. 手動実機テスト投票完了プッシュ通知の配信
    if not dry_run and bet_res.get("status") == "success":
        try:
            from main import send_fcm_notification
            receipt_no = bet_res.get("receipt_no", "")
            send_fcm_notification(
                title=f"🎫【{client.platform_name}実機投票完了】{venue_name}{race_no}R",
                body=f"受付番号:{receipt_no} / {len(items)}点 計{total_amount:,}円 投票完了！\n残高より正常に引き落とされました。",
                data={
                    "type": "MANUAL_BET_SUCCESS",
                    "race_id": race_id,
                    "receipt_no": str(receipt_no),
                    "amount": str(total_amount),
                    "mode": mode or "ipat"
                }
            )
        except Exception as fe:
            print(f"[ManualBetService] [WARN] FCM通知エラー: {fe}", file=sys.stderr)

    return bet_res


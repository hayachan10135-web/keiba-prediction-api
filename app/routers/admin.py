# -*- coding: utf-8 -*-
"""
app/routers/admin.py

管理者向けメンテナンス・診断・データ修復・ヘルスチェック API ルーター
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query

from app.core.database import get_supabase_client
from app.core.timezone import now_jst, today_jst_str, JST

router = APIRouter(tags=["Admin"])

DICT_PATH = "stats_dict.json"


@router.get("/admin/test-umaca-connection")
def endpoint_test_umaca_connection():
    """Renderコンテナ上からJRA（UMACAスマートおよび照会サービス）へのネットワーク疎通テスト"""
    urls = {
        "voting_site": "https://www.ipat.jra.go.jp/sp/umaca/",
        "inquiry_site": "https://www.nvinq.jra.go.jp/jrasp/SMP000400",
        "jra_top": "https://www.jra.go.jp/"
    }
    results = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
    }

    for name, url in urls.items():
        try:
            start_t = time.time()
            resp = requests.get(url, headers=headers, timeout=10)
            elapsed = round(time.time() - start_t, 2)
            html_snippet = resp.text[:300].replace("\n", " ").strip()
            is_blocked = resp.status_code in (403, 401, 503) or "アクセスが制限されています" in resp.text
            results[name] = {
                "status_code": resp.status_code,
                "elapsed_sec": elapsed,
                "is_blocked": is_blocked,
                "content_bytes": len(resp.content),
                "snippet": html_snippet[:120]
            }
        except Exception as e:
            results[name] = {
                "status_code": None,
                "error": str(e)
            }

    can_reach_all = all(r.get("status_code") == 200 for r in results.values())
    return {
        "status": "ok",
        "can_reach_all": can_reach_all,
        "render_ip_check": results,
        "summary": "【接続成功】RenderからJRAへの通信はブロックされていません。クラウド化が可能です。" if can_reach_all else "【接続制限あり】一部の通信がブロックされています。"
    }


def fix_cancelled_races_2026() -> dict:
    """2026年の中止レース（2/7 東京8-12R、2/8 小倉4R等）を正規の開催中止（返還・投資0/払戻0/不参戦）に是正"""
    supabase = get_supabase_client()
    if not supabase:
        print("[FixCancelled] Supabase未接続")
        return {"status": "skipped", "message": "Supabase not connected"}

    target_race_ids = [
        "202605010308",  # 2026-02-07 東京8R (積雪中止)
        "202605010309",  # 2026-02-07 東京9R (積雪中止)
        "202605010310",  # 2026-02-07 東京10R (積雪中止)
        "202605010311",  # 2026-02-07 東京11R (積雪中止)
        "202605010312",  # 2026-02-07 東京12R (積雪中止)
        "202610010604",  # 2026-02-08 小倉4R (中止)
    ]

    updated_count = 0
    try:
        for r_id in target_race_ids:
            row_data = {
                "strategy_type": "cancelled",
                "target_umaban": None,
                "horse_name": None,
                "grade": "C",
                "is_pass": 1,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": 0,
                "actual_order": None,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": 0,
                "is_hit": 0,
                "hit_type": "開催中止",
                "hypothetical_bet": 0,
                "hypothetical_payout": 0,
                "is_real_bet_target": False,
                "real_bet_tier": "none",
                "real_bet_badge": None,
                "confidence_score": 0.0,
                "edge_ratio": 0.0,
                "bet_details": {
                    "reason": "悪天候によるレース開催中止・全額返還",
                    "real_bet_meta": {
                        "is_real_bet_target": False,
                        "real_bet_badge": None,
                        "real_bet_level": "none",
                        "mark_orders": [],
                        "top3_orders": []
                    }
                },
                "payout_details": None,
                "tansho_bet": 0,
                "fukusho_bet": 0,
                "tansho_payout": 0,
                "fukusho_payout": 0
            }
            supabase.table("keiba_race_settlements").update(row_data).eq("race_id", r_id).execute()
            updated_count += 1
            print(f"[FixCancelled] {r_id} を開催中止（全額返還）に更新しました。")

        msg = f"中止レース是正完了: {updated_count}件を全額返還（投資0/払戻0）に修正しました。"
        print(f"[FixCancelled] {msg}")
        return {"status": "ok", "message": msg, "updated_count": updated_count}
    except Exception as e:
        err_msg = f"[FixCancelled Error] {e}"
        print(err_msg)
        return {"status": "error", "message": err_msg}


@router.get("/admin/fix-cancelled-races")
@router.post("/admin/fix-cancelled-races")
def endpoint_fix_cancelled_races():
    return fix_cancelled_races_2026()


@router.get("/admin/stats-dict-info")
def endpoint_stats_dict_info():
    """統計辞書（stats_dict.json）のファイル存在状態・容量およびSupabase接続ステータス情報"""
    supabase = get_supabase_client()
    file_exists = os.path.exists(DICT_PATH)
    file_size_mb = round(os.path.getsize(DICT_PATH) / (1024 * 1024), 2) if file_exists else 0.0

    return {
        "status": "ok",
        "file_exists": file_exists,
        "file_size_mb": file_size_mb,
        "supabase_connected": supabase is not None,
        "server_time_jst": now_jst().strftime("%Y-%m-%d %H:%M:%S")
    }

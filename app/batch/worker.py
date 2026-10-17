# -*- coding: utf-8 -*-
"""
app/batch/worker.py

一括処理ワーカー（1開催期間内の未予想・未検証レースの高速差分実行パイプライン）
"""

import datetime
from typing import Optional, Callable, Dict, Any, List

from app.core.database import get_supabase_client
from app.routers.batch import batch_status
from app.settlement import invalidate_summary_caches

# 外部からDIされる依存関数
_get_current_kaisai_period_fn: Optional[Callable] = None
_discover_period_kaisai_dates_fn: Optional[Callable] = None
_fetch_venues_for_date_raw_fn: Optional[Callable] = None
_predict_race_core_fn: Optional[Callable] = None
_verify_race_result_fn: Optional[Callable] = None
_now_jst_naive_fn: Optional[Callable] = None
_today_jst_str_fn: Optional[Callable] = None


def set_batch_worker_dependencies(
    get_current_kaisai_period: Callable,
    discover_period_kaisai_dates: Callable,
    fetch_venues_for_date_raw: Callable,
    predict_race_core: Callable,
    verify_race_result: Callable,
    now_jst_naive: Callable,
    today_jst_str: Callable,
):
    """バッチワーカーが必要とする外部関数をDI登録"""
    global _get_current_kaisai_period_fn, _discover_period_kaisai_dates_fn
    global _fetch_venues_for_date_raw_fn, _predict_race_core_fn
    global _verify_race_result_fn, _now_jst_naive_fn, _today_jst_str_fn

    _get_current_kaisai_period_fn = get_current_kaisai_period
    _discover_period_kaisai_dates_fn = discover_period_kaisai_dates
    _fetch_venues_for_date_raw_fn = fetch_venues_for_date_raw
    _predict_race_core_fn = predict_race_core
    _verify_race_result_fn = verify_race_result
    _now_jst_naive_fn = now_jst_naive
    _today_jst_str_fn = today_jst_str


def run_batch_worker(
    target_date: Optional[str] = None,
    mode: str = "all",
    force_predict: bool = False,
    is_19h_settle: bool = False
):
    """
    指定日または現開催期間内の全レースを対象に、未予想・未確定レースを差分判定して並列実行
    mode: "all" (予想+検証), "predict" (予想のみ), "settle" (検証のみ)
    """
    global batch_status
    supabase = get_supabase_client()

    batch_status["is_running"] = True
    batch_status["current"] = 0
    batch_status["predicted"] = 0
    batch_status["settled"] = 0
    batch_status["message"] = "開催期間内の対象レースを走査中..."

    try:
        now_dt = _now_jst_naive_fn() if _now_jst_naive_fn else datetime.datetime.now()
        today_str = _today_jst_str_fn() if _today_jst_str_fn else now_dt.strftime("%Y-%m-%d")
        p_info = _get_current_kaisai_period_fn(now_dt) if _get_current_kaisai_period_fn else {}

        period_dates = _discover_period_kaisai_dates_fn(now_dt) if _discover_period_kaisai_dates_fn else []

        # target_date が明示されている場合は指定日のみを対象とする
        if target_date:
            clean_d = target_date.replace("-", "")
            fmt_d = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}" if len(clean_d) == 8 else target_date
            target_dates = [fmt_d]
        elif mode == "all" and not is_19h_settle:
            target_dates = period_dates if period_dates else []
        else:
            target_dates = period_dates

        if not target_dates:
            batch_status["total"] = 0
            batch_status["message"] = "今開催期間の対象レースが見つかりませんでした。"
            return

        # 対象開催日の全レースを収集
        all_period_races = []
        if _fetch_venues_for_date_raw_fn:
            for d_str in target_dates:
                venues = _fetch_venues_for_date_raw_fn(d_str)
                for v in (venues or []):
                    for r in v.get("races", []):
                        all_period_races.append((d_str, r))

        if not all_period_races:
            batch_status["total"] = 0
            batch_status["message"] = "開催期間内に処理可能なレースが見つかりませんでした。"
            return

        # Supabaseからすでに予想済み・検証済みのrace_idを一括取得して差分判定を高速化
        predicted_ids = set()
        settled_ids = set()
        if supabase:
            try:
                p_res = supabase.table("keiba_predictions_cache").select("race_id").in_("race_date", target_dates).execute()
                for row in (p_res.data or []):
                    if row.get("race_id"):
                        predicted_ids.add(row["race_id"])

                r_res = supabase.table("keiba_race_results_cache").select("race_id").in_("race_date", target_dates).execute()
                for row in (r_res.data or []):
                    if row.get("race_id"):
                        settled_ids.add(row["race_id"])
            except Exception as se:
                print(f"[Batch] 差分キャッシュ照会エラー（フォールバック実行）: {se}")

        work_items = []
        for d_str, race_item in all_period_races:
            r_id = race_item["race_id"]
            need_predict = False
            need_settle = False

            if mode in ("all", "predict"):
                if force_predict or (r_id not in predicted_ids):
                    need_predict = True

            if mode in ("all", "settle"):
                if r_id not in settled_ids:
                    need_settle = True

            if need_predict or need_settle:
                work_items.append((d_str, race_item, need_predict, need_settle))

        batch_status["total"] = len(work_items)
        if not work_items:
            batch_status["message"] = "全対象レースの処理が完了しています（新規対象なし）。"
            return

        for idx, (d_str, race_item, do_predict, do_settle) in enumerate(work_items, 1):
            r_id = race_item["race_id"]
            batch_status["current"] = idx
            batch_status["message"] = f"処理中 ({idx}/{len(work_items)}件: 予想{batch_status['predicted']}件 / 検証{batch_status['settled']}件)"

            if do_predict and _predict_race_core_fn:
                try:
                    _predict_race_core_fn(r_id, force_refresh=force_predict, is_auto_final=False)
                    batch_status["predicted"] += 1
                    predicted_ids.add(r_id)
                except Exception as pe:
                    print(f"[Batch] 予想スキップ/エラー ({r_id}): {pe}")

            if do_settle and _verify_race_result_fn:
                try:
                    _verify_race_result_fn(r_id, is_19h_settle=is_19h_settle)
                    batch_status["settled"] += 1
                    settled_ids.add(r_id)
                except Exception:
                    pass

        batch_status["message"] = f"完了: 対象{len(work_items)}R処理 (新規予想{batch_status['predicted']}件 / 新規検証{batch_status['settled']}件)"

    except Exception as e:
        batch_status["message"] = f"エラー終了: {e}"
        print(f"[Batch] ワーカー全体エラー: {e}")
    finally:
        batch_status["is_running"] = False
        invalidate_summary_caches()

# -*- coding: utf-8 -*-
"""
app/routers/batch.py

一括処理（1開催期間内の未予想・未検証レースの高速差分実行・再計算）API ルーター
"""

import threading
from typing import Optional, Callable, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["Batch"])

# --- レスポンスモデル ---

class BatchStartResponse(BaseModel):
    status: str
    message: str


class BatchStatusResponse(BaseModel):
    is_running: bool
    total: int
    current: int
    predicted: int
    settled: int
    message: str


batch_status: Dict[str, Any] = {
    "is_running": False,
    "total": 0,
    "current": 0,
    "predicted": 0,
    "settled": 0,
    "message": "待機中"
}

# 外部から注入されるワーカー関数コールバック
_batch_worker_fn: Optional[Callable] = None
_recalculate_period_fn: Optional[Callable] = None


def set_batch_workers(worker_fn: Callable, recalculate_fn: Optional[Callable] = None):
    """バッチ実行ワーカー関数をルーターへ登録（DI: 依存性注入）"""
    global _batch_worker_fn, _recalculate_period_fn
    _batch_worker_fn = worker_fn
    if recalculate_fn:
        _recalculate_period_fn = recalculate_fn


def start_batch_thread(target_date: Optional[str] = None, mode: str = "all", force_predict: bool = False) -> BatchStartResponse:
    global batch_status, _batch_worker_fn
    if batch_status["is_running"]:
        return BatchStartResponse(
            status="already_running",
            message=f"現在処理中です ({batch_status['current']}/{batch_status['total']}レース進行中)"
        )

    if not _batch_worker_fn:
        raise HTTPException(status_code=500, detail="バッチワーカーが未初期化です。")

    thread = threading.Thread(
        target=_batch_worker_fn,
        args=(target_date, mode, force_predict),
        daemon=True
    )
    thread.start()

    return BatchStartResponse(
        status="started",
        message=f"バックグラウンドで今開催期間の未処理レース一括処理を開始しました (モード: {mode})。"
    )


@router.post("/batch/today", response_model=BatchStartResponse)
def run_today_batch(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD、未指定時は開催期間全体)"),
    mode: str = Query("all", description="処理モード: all (予想＆照合), predict (予想のみ), settle (照合のみ)"),
    force: bool = Query(False, description="強制再予想フラグ")
):
    return start_batch_thread(date, mode=mode, force_predict=force)


@router.post("/batch/predict", response_model=BatchStartResponse)
def run_predict_batch(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)"),
    force: bool = Query(False, description="強制再予想フラグ")
):
    return start_batch_thread(date, mode="predict", force_predict=force)


@router.get("/batch/status", response_model=BatchStatusResponse)
def get_batch_status():
    return BatchStatusResponse(**batch_status)


@router.post("/batch/recalculate-period")
def recalculate_period_records_endpoint():
    """今開催期間のキャッシュ済み予想・確定成績に最新の買い目・実戦推奨ロジックを再適用して一括更新"""
    global _recalculate_period_fn
    if not _recalculate_period_fn:
        raise HTTPException(status_code=500, detail="再計算ワーカーが未初期化です。")
    return _recalculate_period_fn()


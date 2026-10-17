# -*- coding: utf-8 -*-
"""
app/routers/settlement.py

レース着順・払戻金結果照合および検証 API ルーター
"""

from typing import Optional, Dict, Any, List, Callable
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["Settlement"])


class VerificationResult(BaseModel):
    race_id: str
    race_name: str
    is_pass: bool
    grade: str
    target_umaban: Optional[int]
    horse_name: Optional[str]
    actual_order: Optional[int]
    is_hit: bool
    total_bet: int
    total_payout: int
    profit: int
    recovery_rate: float
    from_cache: bool
    message: str
    hypothetical_bet: Optional[int] = None
    hypothetical_payout: Optional[int] = None
    pass_benefit: Optional[int] = None
    strategy_type: Optional[str] = "hybrid"
    hit_type: Optional[str] = None
    bet_breakdown: Optional[Dict[str, int]] = {}
    payout_breakdown: Optional[Dict[str, int]] = {}
    bet_details: Optional[Any] = None
    payout_details: Optional[Any] = None
    umaren_bet: Optional[int] = 0
    trio_bet: Optional[int] = 0
    tansho_bet: Optional[int] = 0
    fukusho_bet: Optional[int] = 0
    umatan_bet: Optional[int] = 0
    umaren_payout: Optional[int] = 0
    trio_payout: Optional[int] = 0
    tansho_payout: Optional[int] = 0
    fukusho_payout: Optional[int] = 0
    umatan_payout: Optional[int] = 0
    mark_orders: Optional[List[dict]] = None
    top3_orders: Optional[List[dict]] = None
    target_popularity: Optional[int] = None


# 外部から注入される照合ハンドラー関数
_verify_race_handler: Optional[Callable] = None


def set_settlement_handler(handler_fn: Callable):
    """結果照合ハンドラー関数をルーターへ登録（DI: 依存性注入）"""
    global _verify_race_handler
    _verify_race_handler = handler_fn


@router.get("/result/{race_id}", response_model=VerificationResult)
def verify_race_result_endpoint(
    race_id: str,
    bypass_cache: bool = Query(False, description="キャッシュをバイパスして最新結果を取得"),
    is_19h_settle: bool = Query(False, description="19時の一括確定モード")
):
    """指定レースの結果（着順・確定払戻金）を検証・照合して成績台帳を更新"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    if not _verify_race_handler:
        raise HTTPException(status_code=500, detail="照合ハンドラーが未初期化です。")

    return _verify_race_handler(race_id, bypass_cache=bypass_cache, is_19h_settle=is_19h_settle)


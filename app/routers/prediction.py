# -*- coding: utf-8 -*-
"""
app/routers/prediction.py

レース情報一覧・事前予想・リアルタイム予想 API ルーター
"""

from typing import Optional, Dict, Any, List, Callable
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["Prediction"])

# --- Pydantic スキーマ ---

class HorsePrediction(BaseModel):
    umaban: int
    wakuban: int
    horse_name: str
    jockey: str
    trainer: Optional[str] = None
    kinryo: float
    odds: Optional[float] = None
    popularity: Optional[int] = None
    score: Optional[float] = 0.0
    prob_win: Optional[float] = None
    prob_top3: Optional[float] = None
    mark: str
    is_blinker: Optional[bool] = False
    is_first_blinker: Optional[bool] = False
    horse_weight: Optional[int] = None
    weight_diff: Optional[int] = None
    sex_age: Optional[str] = None
    base_odds: Optional[float] = None
    odds_history: Optional[Dict[str, float]] = None
    odds_change_pct: Optional[float] = None
    odds_trend: Optional[str] = None
    odds_alert: Optional[str] = None
    bias_match: Optional[str] = None


class TicketBet(BaseModel):
    ticket_type: str
    combination: List[int]
    combination_str: str
    amount: int
    odds: Optional[float] = None
    expected_payout: Optional[int] = None
    is_real_odds: Optional[bool] = False
    memo: Optional[str] = None


class HybridRecommendation(BaseModel):
    is_participate: bool
    strategy_name: str = "波乱度メタモデル最適化"
    strategy_type: Optional[str] = "daiharan"
    bet_category: Optional[str] = "skip"
    badge_text: Optional[str] = None
    total_amount: int
    summary: str
    tansho_bets: Optional[List[TicketBet]] = []
    fukusho_bets: Optional[List[TicketBet]] = []
    umatan_bets: Optional[List[TicketBet]] = []
    wide_bets: Optional[List[TicketBet]] = []
    tansho_amount: Optional[int] = 0
    fukusho_amount: Optional[int] = 0
    umatan_amount: Optional[int] = 0
    wide_amount: Optional[int] = 0
    umaren_amount: Optional[int] = 0
    trio_amount: Optional[int] = 0
    sanrentan_amount: Optional[int] = 0
    tickets: Optional[List[TicketBet]] = []
    umaren_bets: List[TicketBet] = []
    sanrentan_bets: Optional[List[TicketBet]] = []
    hypothetical_umaren_bets: Optional[List[TicketBet]] = []
    trio_bets: Optional[List[TicketBet]] = []
    trio_bet: Optional[TicketBet] = None
    reason: Optional[str] = None
    is_real_odds_applied: Optional[bool] = False
    total_expected_payout: Optional[int] = None
    prob_gachi: Optional[float] = None
    prob_himo: Optional[float] = None
    edge_ratio: Optional[float] = None
    confidence_score: Optional[float] = None
    is_real_bet_target: Optional[bool] = False
    real_bet_badge: Optional[str] = None
    real_bet_level: Optional[str] = None
    is_final_odds: Optional[bool] = False
    is_virtual_bet: Optional[bool] = False
    # 仮想馬券フィールド
    is_virtual_bet_target: Optional[bool] = False
    virtual_bet_slot: Optional[int] = None
    virtual_strategy_name: Optional[str] = None
    virtual_badge: Optional[str] = None
    virtual_bets: Optional[List[TicketBet]] = []
    virtual_total_amount: Optional[int] = 0


class TanFukuRecommendation(BaseModel):
    target_umaban: Optional[int]
    horse_name: Optional[str]
    grade: str
    confidence_label: str
    is_pass: bool
    tansho_amount: int
    fukusho_amount: int
    total_amount: int
    summary: str
    bet_category: Optional[str] = "skip"
    badge_text: Optional[str] = None
    is_virtual_bet: Optional[bool] = False


class RaceDetails(BaseModel):
    race_num: str
    post_time: str
    course_details: str


class TrackBiasInfo(BaseModel):
    venue_name: str
    surface_name: str
    sample_races: int
    bias_summary: str
    waku_bias: str = "flat"
    running_style_bias: str = "flat"
    inside_top3_ratio: float = 0.0
    outside_top3_ratio: float = 0.0
    detail_text: str = ""


class PredictResponse(BaseModel):
    race_id: str
    race_name: str
    race_details: RaceDetails
    horses: List[HorsePrediction]
    recommendation: TanFukuRecommendation
    hybrid_recommendation: Optional[HybridRecommendation] = None
    track_bias: Optional[TrackBiasInfo] = None
    last_odds_stage: Optional[str] = None


# --- 外部から注入される推論・レース一覧ハンドラー関数 ---
_today_races_handler: Optional[Callable] = None
_all_predictions_handler: Optional[Callable] = None
_predict_race_handler: Optional[Callable] = None


def set_prediction_handlers(today_fn: Callable, all_fn: Callable, predict_fn: Callable):
    """推論ハンドラー関数をルーターへ登録（DI: 依存性注入）"""
    global _today_races_handler, _all_predictions_handler, _predict_race_handler
    _today_races_handler = today_fn
    _all_predictions_handler = all_fn
    _predict_race_handler = predict_fn


@router.get("/races/today")
def get_today_races_endpoint(
    date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)"),
    force_refresh: bool = Query(False, description="手動でレース一覧・発走時刻を再取得する場合True")
):
    """当日のレース一覧・会場別一覧を取得"""
    if not _today_races_handler:
        raise HTTPException(status_code=500, detail="レース一覧ハンドラーが未初期化です。")
    return _today_races_handler(date=date, force_refresh=force_refresh)


@router.get("/predict/all")
def get_all_predictions_endpoint(date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)")):
    """指定日付の全レースのキャッシュ済み予想を一括返却（アプリの先読み・ゼロレイテンシ用）"""
    if not _all_predictions_handler:
        raise HTTPException(status_code=500, detail="全予想ハンドラーが未初期化です。")
    return _all_predictions_handler(date=date)


@router.get("/predict/{race_id}", response_model=PredictResponse)
def predict_race_endpoint(race_id: str, force_refresh: bool = Query(False, description="手動で予想を再実行する場合True")):
    """指定レースの予想結果を取得（リアルタイムオッズ・バイアス反映）"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")
    if not _predict_race_handler:
        raise HTTPException(status_code=500, detail="推論ハンドラーが未初期化です。")
    res = _predict_race_handler(race_id=race_id, force_refresh=force_refresh)
    if isinstance(res, dict):
        hr = res.get("hybrid_recommendation")
        if hr and isinstance(hr, dict):
            if hr.get("strategy_type") == "skip" or (hr.get("total_amount") or 0) == 0 or res.get("recommendation", {}).get("is_pass"):
                hr["is_real_bet_target"] = False
                if not hr.get("is_virtual_bet_target") and not hr.get("is_virtual_bet") and hr.get("bet_category") != "virtual":
                    hr["real_bet_badge"] = None
                    hr["real_bet_level"] = "none"
    return res


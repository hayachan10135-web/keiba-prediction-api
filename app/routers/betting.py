# -*- coding: utf-8 -*-
"""
app/routers/betting.py

自動投票設定・残高照会・手動（マニュアル）投票・接続テスト API ルーター
"""

import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, StrictInt

from app.betting.factory import BettingClientFactory
from app.betting.umaca_client import UMACAClient
from app.betting.manual_service import execute_manual_bet
from app.betting.settings import (
    load_betting_settings_from_supabase,
    save_betting_settings_to_supabase,
    calculate_remaining_real_bet_amount,
    get_today_total_betted,
)

router = APIRouter(tags=["Betting"])


class BettingModeItem(BaseModel):
    id: str
    name: str
    enabled: bool
    description: str


class BettingSettingsResponse(BaseModel):
    betting_mode: str
    auto_bet_real_only: bool
    available_modes: List[BettingModeItem]
    ipat_configured: bool
    umaca_configured: Optional[bool] = False
    last_balance: Optional[int] = None
    balance_status: Optional[str] = "ok"
    balance_message: Optional[str] = None
    remaining_bet_amount: int = 0
    remaining_real_races_count: int = 0
    balance_shortfall: int = 0
    daily_investment_limit: Optional[int] = 30000
    today_total_betted: int = 0
    updated_at: Optional[str] = None


class UpdateBettingModeRequest(BaseModel):
    betting_mode: Optional[str] = None
    daily_investment_limit: Optional[int] = None


class ManualBetItem(BaseModel):
    ticket_type: str
    combination: List[StrictInt]
    amount: StrictInt


class ManualBetRequest(BaseModel):
    race_id: str
    venue_name: str
    race_no: StrictInt
    items: List[ManualBetItem]
    mode: Optional[str] = None
    dry_run: bool = False


@router.get("/settings/betting-mode", response_model=BettingSettingsResponse)
def get_betting_settings_endpoint():
    """現在の自動投票設定を取得 (Fail-Safe 耐障害性強化版)"""
    try:
        settings = load_betting_settings_from_supabase()
        cur_mode = settings.get("betting_mode", "off")
        has_ipat_env = bool(os.getenv("IPAT_USER_NUM") and os.getenv("IPAT_PIN") and os.getenv("IPAT_PARS"))
        has_umaca_env = UMACAClient().is_configured()

        modes = [
            BettingModeItem(
                id="off",
                name="自動投票なし (OFF)",
                enabled=True,
                description="予想・買い目表示および通知のみ行い、自動投票は実行しません。"
            ),
            BettingModeItem(
                id="ipat",
                name="即PAT (IPAT)",
                enabled=True,
                description="発走5分前に「実馬券・勝負馬券」に該当した場合、即PATへ自動で買い目を投入します（事前手動入金残高内）。"
            ),
            BettingModeItem(
                id="umaca",
                name="UMACAスマート",
                enabled=True,
                description="発走5分前に「実馬券・勝負馬券」に該当した場合、UMACAスマートへ自動で買い目を投入します（投票用残高内）。"
            )
        ]

        try:
            rem_amt, rem_count, _ = calculate_remaining_real_bet_amount()
        except Exception as ce:
            print(f"[get_betting_settings] 投票予定額算出エラー: {ce}")
            rem_amt, rem_count = 0, 0

        last_bal = settings.get("last_balance") or 0
        shortfall = max(0, rem_amt - last_bal) if (cur_mode in ("ipat", "umaca") and last_bal is not None) else 0

        today_betted = get_today_total_betted()
        daily_limit = settings.get("daily_investment_limit", 30000)

        return BettingSettingsResponse(
            betting_mode=cur_mode,
            auto_bet_real_only=bool(settings.get("auto_bet_real_only", True)),
            available_modes=modes,
            ipat_configured=(has_umaca_env if cur_mode == "umaca" else has_ipat_env),
            umaca_configured=has_umaca_env,
            last_balance=settings.get("last_balance"),
            balance_status=settings.get("balance_status", "ok" if settings.get("last_balance") is not None else "unavailable"),
            balance_message=settings.get("balance_message"),
            remaining_bet_amount=rem_amt,
            remaining_real_races_count=rem_count,
            balance_shortfall=shortfall,
            daily_investment_limit=daily_limit,
            today_total_betted=today_betted,
            updated_at=settings.get("updated_at")
        )
    except Exception as e:
        print(f"[get_betting_settings] 設定取得エラー: {e}")
        return BettingSettingsResponse(
            betting_mode="off",
            auto_bet_real_only=True,
            available_modes=[
                BettingModeItem(id="off", name="自動投票なし (OFF)", enabled=True, description="予想・買い目表示および通知のみ行います。"),
                BettingModeItem(id="ipat", name="即PAT (IPAT)", enabled=True, description="即PAT連携"),
                BettingModeItem(id="umaca", name="UMACAスマート", enabled=True, description="UMACA連携")
            ],
            ipat_configured=False,
            umaca_configured=False,
            last_balance=None,
            balance_status="unavailable",
            balance_message="設定を一時的に読み込めませんでした",
            remaining_bet_amount=0,
            remaining_real_races_count=0,
            balance_shortfall=0,
            daily_investment_limit=30000,
            today_total_betted=0,
            updated_at=None
        )


@router.post("/settings/betting-mode")
def update_betting_mode_endpoint(req: UpdateBettingModeRequest):
    """自動投票モードおよび1日投資上限を変更 (off / ipat / umaca)"""
    settings = load_betting_settings_from_supabase()

    if req.daily_investment_limit is not None:
        if req.daily_investment_limit < 0:
            raise HTTPException(status_code=400, detail="投資上限額には0以上の数値を指定してください。")
        settings["daily_investment_limit"] = req.daily_investment_limit

    if req.betting_mode is not None:
        new_mode = req.betting_mode.lower().strip()
        if new_mode not in ("off", "ipat", "umaca"):
            raise HTTPException(status_code=400, detail=f"無効な投票モードです: {new_mode}")
        settings["betting_mode"] = new_mode

        # モード切替時に選択したプラットフォームの最新残高を自動照会
        if new_mode in ("ipat", "umaca"):
            try:
                client = BettingClientFactory.get_client(new_mode)
                l_res = client.login()
                if l_res.get("success"):
                    settings["last_balance"] = l_res.get("balance", 0)
                    settings["balance_status"] = "ok"
                    settings["balance_message"] = None
                else:
                    # 取得できなかった場合は前回の古い残高をクリアして明示
                    settings["last_balance"] = None
                    settings["balance_status"] = "unavailable"
                    err_text = l_res.get("error", "残高を取得できませんでした")
                    if "時間外" in err_text:
                        settings["balance_message"] = "ただいま投票受付時間外です"
                    else:
                        settings["balance_message"] = err_text
            except Exception as e:
                settings["last_balance"] = None
                settings["balance_status"] = "unavailable"
                settings["balance_message"] = f"照会例外: {e}"
                print(f"[BettingModeSwitch] 残高自動照会エラー ({new_mode}): {e}")
        else:
            # off の場合
            settings["last_balance"] = None
            settings["balance_status"] = "off"
            settings["balance_message"] = "自動投票OFF"

    if not save_betting_settings_to_supabase(settings):
        raise HTTPException(status_code=503, detail="投票設定をDBへ保存できませんでした。自動投票設定を変更していません。")
    mode_names = {
        "off": "自動投票なし (OFF)",
        "ipat": "即PAT (IPAT)",
        "umaca": "UMACAスマート"
    }
    new_mode = settings.get("betting_mode", "off")
    return {
        "status": "ok",
        "betting_mode": new_mode,
        "last_balance": settings.get("last_balance"),
        "balance_status": settings.get("balance_status"),
        "balance_message": settings.get("balance_message"),
        "message": f"自動投票モードを「{mode_names.get(new_mode, new_mode)}」に設定しました。"
    }


@router.post("/settings/test-ipat")
@router.post("/settings/test-umaca")
def test_ipat_endpoint(mode: Optional[str] = Query(None, description="テスト対象 ('ipat' または 'umaca'。省略時は現在の設定モード)")):
    """現在選択中のプラットフォーム（即PAT または UMACAスマート）へのログイン＆残高取得テストを実行"""
    settings = load_betting_settings_from_supabase()
    target_mode = (mode or settings.get("betting_mode") or "ipat").lower().strip()
    if target_mode == "off":
        # OFF選択中に接続テストを押した場合は設定済みのプラットフォームをテスト
        target_mode = "umaca" if UMACAClient().is_configured() else "ipat"

    try:
        client = BettingClientFactory.get_client(target_mode)
        res = client.login()
        label = "UMACAスマート" if target_mode == "umaca" else "即PAT"
        bal_label = "投票用残高" if target_mode == "umaca" else "購入限度額"

        if res.get("success"):
            balance = res.get("balance", 0)
            settings["last_balance"] = balance
            settings["balance_status"] = "ok"
            settings["balance_message"] = None
            save_betting_settings_to_supabase(settings)
            return {
                "success": True,
                "platform": target_mode,
                "balance": balance,
                "message": f"{label}接続成功！ {bal_label}: {balance:,}円"
            }
        else:
            err_msg = res.get("error", f"{label}への接続に失敗しました")
            settings["last_balance"] = None
            settings["balance_status"] = "unavailable"
            settings["balance_message"] = err_msg
            save_betting_settings_to_supabase(settings)
            return {
                "success": False,
                "platform": target_mode,
                "balance": None,
                "error": err_msg
            }
    except Exception as e:
        err_msg = f"接続テスト例外エラー: {e}"
        settings["last_balance"] = None
        settings["balance_status"] = "unavailable"
        settings["balance_message"] = err_msg
        save_betting_settings_to_supabase(settings)
        return {
            "success": False,
            "platform": target_mode,
            "balance": None,
            "error": err_msg
        }


@router.post("/betting/manual")
def manual_bet_endpoint(req: ManualBetRequest):
    """
    新要件 FR-BET-07: マニュアル（手動・実機テスト）投票API
    ★【厳格な集計除外規定】: is_manual_bet=True で記録され、AI収支集計から完全に除外されます。
    """
    settings = load_betting_settings_from_supabase()
    target_mode = (req.mode or settings.get("betting_mode") or "ipat").lower().strip()
    if target_mode not in ("ipat", "umaca"):
        raise HTTPException(status_code=400, detail="投票モードはipatまたはumacaを指定してください。")

    items_dicts = [item.dict() for item in req.items]
    try:
        res = execute_manual_bet(
            race_id=req.race_id,
            venue_name=req.venue_name,
            race_no=req.race_no,
            items=items_dicts,
            mode=target_mode,
            dry_run=req.dry_run
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return res

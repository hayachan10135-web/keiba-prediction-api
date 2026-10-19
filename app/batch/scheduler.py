# -*- coding: utf-8 -*-
"""
app/batch/scheduler.py

リアルタイム・メンテナンススケジューラー（JST完全同期・最適化タイムテーブル版）
① 開催日 朝07:30 : 当日の出走表・発走スケジュール一括登録
② 開催日 朝08:30 : 当日の未予想レース暫定一括予想 ＆ 朝ダイジェスト通知
③ 各レース発走 60m(馬体重) / 30m / 15m / 05m(最終確定) : マルチステージ自動推論 ＆ 自動投票 (即PAT/UMACA)
④ 各レース発走 10分後 : レース結果自動検証（未確定時は3分毎リトライ）＆ 結果通知
⑤ 開催日 17:00 / 19:00 : 当日最終の未確定レース一括照合
⑥ 金曜日 夜20:00 : 1開催期間分の一時キャッシュ破棄（成績台帳は永続保持）
"""

import asyncio
import datetime
import gc
from typing import Optional, Callable, Dict, Any, List, Set

from app.core.database import get_supabase_client
from app.routers.batch import batch_status
from app.betting.factory import BettingClientFactory
from app.betting.settings import (
    load_betting_settings_from_supabase,
    save_betting_settings_to_supabase,
    check_investment_limit,
)

# DIされる外部依存
_now_jst_fn: Optional[Callable] = None
_now_jst_naive_fn: Optional[Callable] = None
_today_jst_str_fn: Optional[Callable] = None
_parse_post_datetime_fn: Optional[Callable] = None
_get_current_kaisai_period_fn: Optional[Callable] = None
_clear_kaisai_period_caches_fn: Optional[Callable] = None
_discover_period_kaisai_dates_fn: Optional[Callable] = None
_fetch_venues_for_date_raw_fn: Optional[Callable] = None
_ensure_daily_single_real_bet_fn: Optional[Callable] = None
_send_fcm_notification_fn: Optional[Callable] = None
_check_and_notify_balance_shortfall_fn: Optional[Callable] = None
_predict_race_core_fn: Optional[Callable] = None
_verify_race_result_fn: Optional[Callable] = None
_has_already_betted_race_fn: Optional[Callable] = None
_run_batch_worker_fn: Optional[Callable] = None

_period_checked_empty_dates_ref: Optional[Set[str]] = None
_auto_betted_races_ref: Optional[Set[str]] = None
_auto_betted_dates_ref: Optional[Set[str]] = None
_venue_code_map_ref: Optional[Dict[str, str]] = None
_last_stage2_check: Dict[str, float] = {}


def set_scheduler_dependencies(
    now_jst: Callable,
    now_jst_naive: Callable,
    today_jst_str: Callable,
    parse_post_datetime: Callable,
    get_current_kaisai_period: Callable,
    clear_kaisai_period_caches: Callable,
    discover_period_kaisai_dates: Callable,
    fetch_venues_for_date_raw: Callable,
    ensure_daily_single_real_bet: Callable,
    send_fcm_notification: Callable,
    check_and_notify_balance_shortfall: Callable,
    predict_race_core: Callable,
    verify_race_result: Callable,
    has_already_betted_race: Callable,
    run_batch_worker: Callable,
    period_checked_empty_dates: Set[str],
    auto_betted_races: Set[str],
    auto_betted_dates: Set[str],
    venue_code_map: Dict[str, str],
):
    """スケジューラーが必要とする外部関数・変数をDI登録"""
    global _now_jst_fn, _now_jst_naive_fn, _today_jst_str_fn, _parse_post_datetime_fn
    global _get_current_kaisai_period_fn, _clear_kaisai_period_caches_fn
    global _discover_period_kaisai_dates_fn, _fetch_venues_for_date_raw_fn
    global _ensure_daily_single_real_bet_fn, _send_fcm_notification_fn
    global _check_and_notify_balance_shortfall_fn, _predict_race_core_fn
    global _verify_race_result_fn, _has_already_betted_race_fn, _run_batch_worker_fn
    global _period_checked_empty_dates_ref, _auto_betted_races_ref
    global _auto_betted_dates_ref, _venue_code_map_ref

    _now_jst_fn = now_jst
    _now_jst_naive_fn = now_jst_naive
    _today_jst_str_fn = today_jst_str
    _parse_post_datetime_fn = parse_post_datetime
    _get_current_kaisai_period_fn = get_current_kaisai_period
    _clear_kaisai_period_caches_fn = clear_kaisai_period_caches
    _discover_period_kaisai_dates_fn = discover_period_kaisai_dates
    _fetch_venues_for_date_raw_fn = fetch_venues_for_date_raw
    _ensure_daily_single_real_bet_fn = ensure_daily_single_real_bet
    _send_fcm_notification_fn = send_fcm_notification
    _check_and_notify_balance_shortfall_fn = check_and_notify_balance_shortfall
    _predict_race_core_fn = predict_race_core
    _verify_race_result_fn = verify_race_result
    _has_already_betted_race_fn = has_already_betted_race
    _run_batch_worker_fn = run_batch_worker
    _period_checked_empty_dates_ref = period_checked_empty_dates
    _auto_betted_races_ref = auto_betted_races
    _auto_betted_dates_ref = auto_betted_dates
    _venue_code_map_ref = venue_code_map


async def sync_daily_schedules_if_needed(today_str: str):
    """当日のJRA開催有無を確認し、開催日であればレーススケジュールをkeiba_scheduled_racesに事前登録"""
    supabase = get_supabase_client()
    if not supabase:
        return
    if _period_checked_empty_dates_ref is not None and today_str in _period_checked_empty_dates_ref:
        return
    try:
        res = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", today_str).execute()
        valid_ids = [r["race_id"] for r in (res.data or []) if len(r["race_id"]) == 12]
        if len(valid_ids) >= 20:
            return

        if _fetch_venues_for_date_raw_fn:
            venues = _fetch_venues_for_date_raw_fn(today_str)
            if venues:
                print(f"[Scheduler] {today_str} (JST) のJRA開催スケジュール ({sum(len(v.get('races', [])) for v in venues)}R) を事前登録しました。")
    except Exception as e:
        print(f"[Scheduler] スケジュール事前登録エラー: {e}")


async def background_maintenance_loop():
    """リアルタイム・常駐メンテナンススケジューラー"""
    print("[Scheduler] リアルタイム・メンテナンススケジューラー (JST同期・1開催期間サイクル版) を開始しました。")
    last_friday_cleaned_period = None
    last_17h_settle_date = None
    last_19h_settle_date = None
    last_830_predict_date = None
    last_daily_backup_date = None
    healed_post_weight_rids = set()

    while True:
        try:
            supabase = get_supabase_client()
            now_dt = _now_jst_naive_fn() if _now_jst_naive_fn else datetime.datetime.now()
            today_str = _today_jst_str_fn() if _today_jst_str_fn else now_dt.strftime("%Y-%m-%d")
            p_info = _get_current_kaisai_period_fn(now_dt) if _get_current_kaisai_period_fn else {}

            # --- ⓪ 深夜 3:00 の Google ドライブ自動日次バックアップ ---
            if now_dt.hour == 3 and last_daily_backup_date != today_str:
                try:
                    from app.maintenance.backup import backup_database_to_drive
                    backup_database_to_drive()
                    last_daily_backup_date = today_str
                except Exception as be:
                    print(f"[Scheduler] 自動日次バックアップエラー: {be}")

            # --- ⑥ 金曜 20:00 以降の古いキャッシュ自動整理 (75日以前の過去分のみ安全破棄) ---
            if now_dt.weekday() == 4 and now_dt.hour >= 20:
                if last_friday_cleaned_period != p_info.get("period_start"):
                    if _clear_kaisai_period_caches_fn:
                        _clear_kaisai_period_caches_fn()
                    last_friday_cleaned_period = p_info.get("period_start")
                    last_830_predict_date = None
                    last_17h_settle_date = None
                    last_19h_settle_date = None

            # --- ① JST朝 7:30 以降、当日のレーススケジュールを事前登録 ---
            if now_dt.hour > 7 or (now_dt.hour == 7 and now_dt.minute >= 30):
                await sync_daily_schedules_if_needed(today_str)

            # --- ② JST朝 8:30 以降、当日の未予想レース事前予想（朝オッズによる暫定判定）を一括実行 ---
            if (now_dt.hour > 8 or (now_dt.hour == 8 and now_dt.minute >= 30)) and last_830_predict_date != today_str:
                period_dates = _discover_period_kaisai_dates_fn(now_dt) if _discover_period_kaisai_dates_fn else []
                if today_str in period_dates:
                    already_predicted_count = 0
                    if supabase:
                        try:
                            chk_p = supabase.table("keiba_predictions_cache").select("race_id").eq("race_date", today_str).execute()
                            already_predicted_count = len(chk_p.data or [])
                        except Exception:
                            pass

                    if already_predicted_count >= 20:
                        print(f"[Scheduler] 当日({today_str})の予想データは既に準備済み({already_predicted_count}R)です。各レースのマルチステージ自動推論(60m/30m/15m/05m)へ移行します。")
                        if _ensure_daily_single_real_bet_fn:
                            await asyncio.to_thread(_ensure_daily_single_real_bet_fn, today_str)
                        last_830_predict_date = today_str
                    elif not batch_status["is_running"]:
                        print(f"[Scheduler] 朝8:30 当日({today_str})全レースの一括事前予想を実行します...")
                        loop = asyncio.get_event_loop()
                        if _run_batch_worker_fn:
                            await loop.run_in_executor(None, lambda: _run_batch_worker_fn(today_str, "predict", False, False))
                        if _ensure_daily_single_real_bet_fn:
                            await asyncio.to_thread(_ensure_daily_single_real_bet_fn, today_str)
                        last_830_predict_date = today_str

                        # 朝の実戦推奨候補を走査してダイジェスト通知
                        try:
                            if supabase:
                                p_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", today_str).execute()
                                real_races = []
                                for row in (p_res.data or []):
                                    h_rec = row.get("data_json", {}).get("hybrid_recommendation") or {}
                                    if h_rec.get("is_real_bet_target"):
                                        r_name = row.get("data_json", {}).get("race_name", row["race_id"])
                                        b_txt = h_rec.get("real_bet_badge", "実戦推奨")
                                        e_ratio = h_rec.get("edge_ratio", 0.0)
                                        real_races.append(f"・{r_name} (エッジ{e_ratio}倍 {b_txt})")

                                if real_races and _send_fcm_notification_fn:
                                    summary_body = "\n".join(real_races[:5])
                                    if len(real_races) > 5:
                                        summary_body += f"\n他{len(real_races)-5}レース"
                                    summary_body += "\n※発走60分前(馬体重)・30分前・15分前・5分前(最終確定)にオッズ推移を追跡し判定します。"
                                    _send_fcm_notification_fn(
                                        title=f"🌅【本日実戦推奨候補】計{len(real_races)}レース該当！",
                                        body=summary_body,
                                        data={"type": "MORNING_SUMMARY", "race_date": today_str}
                                    )
                                    if _check_and_notify_balance_shortfall_fn:
                                        _check_and_notify_balance_shortfall_fn()
                        except Exception as fe:
                            print(f"[Scheduler] 朝通知エラー: {fe}")
                else:
                    # 本日が非開催日（平日等）の場合はチェック済みとしてマーク
                    last_830_predict_date = today_str

            if supabase:
                # --- ③ 各レース【発走 60分前 / 30分前 / 15分前 / 5分前】のマルチステージ自動推論 ＆ 最終確定通知 ---
                try:
                    res_pred = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("race_date", today_str)\
                        .lt("is_predicted", 5)\
                        .execute()
                    unpredicted_races = res_pred.data or []

                    # 1ループあたりに実行する最大推論数をPCスペック(10コア/16GB)に合わせて引き上げ (直前推論の遅延・滞留を完全根絶)
                    predicted_count_this_loop = 0
                    MAX_PREDICTIONS_PER_LOOP = 6

                    for r in unpredicted_races:
                        if predicted_count_this_loop >= MAX_PREDICTIONS_PER_LOOP:
                            break

                        r_id = r["race_id"]
                        # 【最重要ガード】JRA公式の本日会場コード (04: 新潟, 05: 東京, 08: 京都) 以外の不正・過去レースは除外
                        venue_code = r_id[4:6] if len(r_id) == 12 else ""
                        if venue_code not in {"04", "05", "08"}:
                            continue

                        p_dt_str = r.get("post_datetime")
                        cur_stage = int(r.get("is_predicted") or 0)
                        post_dt = _parse_post_datetime_fn(p_dt_str, today_str) if _parse_post_datetime_fn else None
                        if not post_dt:
                            continue

                        mins_to_post = (post_dt - now_dt).total_seconds() / 60.0
                        if mins_to_post <= 0 or mins_to_post > 65.0:
                            continue

                        # Stage 5: 発走 5分前 (0 < mins <= 6.5) -> 最終オッズ推論 ＆ 確定通知
                        if mins_to_post <= 6.5 and cur_stage < 5:
                            predicted_count_this_loop += 1
                            print(f"[Auto-Predict Stage5: 5分前最終確定] {r_id} ({r.get('race_name')}) の最終推論＆買い目確定を実行します...")
                            try:
                                pred_res = await asyncio.to_thread(_predict_race_core_fn, r_id, True, True, "05m") if _predict_race_core_fn else {}
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 5})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Predict Stage5] 確定完了: {r_id}")

                                h_rec = pred_res.get("hybrid_recommendation") or {}
                                b_cat = h_rec.get("bet_category", "skip")
                                is_bet_target = bool(h_rec.get("is_real_bet_target")) and (b_cat in ("super", "real"))
                                if is_bet_target:
                                    badge = h_rec.get("badge_text") or h_rec.get("real_bet_badge", "🔥 実戦推奨(確定)")
                                    edge = h_rec.get("edge_ratio", 0.0)
                                    target_u = pred_res.get("recommendation", {}).get("target_umaban")
                                    if not target_u:
                                        for h in pred_res.get("horses", []):
                                            if h.get("mark") == "◎":
                                                target_u = int(h.get("umaban"))
                                                break
                                    h_name = pred_res.get("recommendation", {}).get("horse_name", "")
                                    r_name = pred_res.get("race_name", r.get("race_name", f"Race {r_id}"))

                                    all_tickets = (
                                        (h_rec.get("tansho_bets") or [])
                                        + (h_rec.get("fukusho_bets") or [])
                                        + (h_rec.get("umatan_bets") or [])
                                        + (h_rec.get("umaren_bets") or [])
                                        + (h_rec.get("trio_bets") or [])
                                        + (h_rec.get("sanrentan_bets") or [])
                                    )
                                    tot_amt = h_rec.get("total_amount", 0)

                                    bets_summary = ", ".join([f"{b.get('ticket_type')} {b.get('combination_str')}({b.get('amount')}円)" for b in all_tickets[:3]])

                                    body_text = f"◎{target_u}番 {h_name} (期待エッジ:{edge}倍)\n【推奨買い目】{bets_summary} 計{tot_amt:,}円"
                                    if _send_fcm_notification_fn:
                                        _send_fcm_notification_fn(
                                            title=f"{badge} {r_name}！",
                                            body=body_text,
                                            data={
                                                "type": "REAL_BET_FINAL",
                                                "race_id": r_id,
                                                "edge_ratio": str(edge),
                                                "badge": badge
                                            }
                                        )

                                    # --- 【自動投票実行判定 (即PAT / UMACAスマート)】 ---
                                    b_settings = load_betting_settings_from_supabase()
                                    cur_bet_mode = b_settings.get("betting_mode", "off")

                                    if cur_bet_mode in ("ipat", "umaca"):
                                        mode_label = "UMACA" if cur_bet_mode == "umaca" else "即PAT"
                                        already_betted = _has_already_betted_race_fn(today_str, r_id) if _has_already_betted_race_fn else False
                                        if already_betted:
                                            print(f"[AutoBet] {r_id} は既に投票済みのためスキップします (二重発注防止)")
                                        else:
                                            # --- 【1日最大投資上限ガード判定 (FR-BET-04 / FR-SAF-01)】 ---
                                            is_limit_exceeded, cur_today_betted, daily_limit = check_investment_limit(tot_amt, today_str)
                                            if is_limit_exceeded:
                                                print(f"[AutoBet Guard] 🛡️ 1日投資上限（{daily_limit:,}円）を超過するため自動投票を安全スキップします (本日累計: {cur_today_betted:,}円 + 今回予定: {tot_amt:,}円 > 上限: {daily_limit:,}円)")
                                                if _send_fcm_notification_fn:
                                                    _send_fcm_notification_fn(
                                                        title=f"🛡️【投資上限ガード】{r_name}",
                                                        body=f"1日の最大投資上限（{daily_limit:,}円）に達したため、自動投票を安全スキップしました。\n（本日累計: {cur_today_betted:,}円 / 今回見送り: {tot_amt:,}円）",
                                                        data={
                                                            "type": "DAILY_LIMIT_EXCEEDED",
                                                            "race_id": r_id,
                                                            "limit": str(daily_limit),
                                                            "current_betted": str(cur_today_betted),
                                                            "attempted_amount": str(tot_amt)
                                                        }
                                                    )
                                            else:
                                                if _auto_betted_races_ref is not None:
                                                    _auto_betted_races_ref.add(r_id)
                                                print(f"[AutoBet] 🎫 {r_id} ({r_name}) の{mode_label}自動投票を開始します...")
                                                try:
                                                    v_code = r_id[4:6] if len(r_id) == 12 else "05"
                                                    v_map = _venue_code_map_ref or {}
                                                    v_name = v_map.get(v_code, "東京")
                                                    r_no = int(r_id[10:12]) if len(r_id) == 12 else 11

                                                    bet_items = []
                                                    for b in all_tickets:
                                                        c_list = b.get("combination")
                                                        if not c_list:
                                                            t_u = b.get("target_umaban")
                                                            c_list = [t_u] if t_u is not None else []
                                                        bet_items.append({
                                                            "ticket_type": b.get("ticket_type", "単勝"),
                                                            "combination": c_list,
                                                            "amount": b.get("amount", 100)
                                                        })

                                                    client = BettingClientFactory.get_client(cur_bet_mode)
                                                    bet_res = await asyncio.to_thread(client.execute_bets, v_name, r_no, bet_items, False)

                                                    if bet_res.get("status") == "success":
                                                        rcpt = bet_res.get("receipt_no", "")
                                                        if _auto_betted_dates_ref is not None:
                                                            _auto_betted_dates_ref.add(today_str)
                                                        if _auto_betted_races_ref is not None:
                                                            _auto_betted_races_ref.add(r_id)
                                                        try:
                                                            cur_s = load_betting_settings_from_supabase()
                                                            b_dict = cur_s.get("betted_races_by_date", {})
                                                            b_list = b_dict.get(today_str, [])
                                                            now_jst_obj = _now_jst_fn() if _now_jst_fn else datetime.datetime.now()
                                                            b_list.append({
                                                                "race_id": r_id,
                                                                "receipt_no": str(rcpt),
                                                                "amount": tot_amt,
                                                                "mode": cur_bet_mode,
                                                                "time": now_jst_obj.strftime("%H:%M:%S")
                                                            })
                                                            b_dict[today_str] = b_list
                                                            cur_s["betted_races_by_date"] = b_dict
                                                            if bet_res.get("balance") is not None:
                                                                cur_s["last_balance"] = bet_res.get("balance")
                                                                cur_s["last_balance_updated_at"] = now_jst_obj.strftime("%Y-%m-%d %H:%M:%S")
                                                            save_betting_settings_to_supabase(cur_s)
                                                        except Exception as me:
                                                            print(f"[AutoBet Meta Save Error] {me}")
                                                        print(f"[AutoBet] 🎉 {mode_label}投票成功！ 受付番号: {rcpt}")
                                                        if _send_fcm_notification_fn:
                                                            _send_fcm_notification_fn(
                                                                title=f"🎫【{mode_label}投票完了】{v_name}{r_no}R",
                                                                body=f"受付番号:{rcpt} / {len(bet_items)}点 計{tot_amt:,}円 投票完了！\n残高より正常に引き落とされました。",
                                                                data={
                                                                    "type": "AUTO_BET_SUCCESS",
                                                                    "race_id": r_id,
                                                                    "receipt_no": str(rcpt),
                                                                    "amount": str(tot_amt),
                                                                    "mode": cur_bet_mode
                                                                }
                                                            )
                                                    else:
                                                        err_msg = bet_res.get("reason") or bet_res.get("error") or "投票処理エラー"
                                                        print(f"[AutoBet] ⚠️ {mode_label}投票失敗: {err_msg}")
                                                        if _send_fcm_notification_fn:
                                                            _send_fcm_notification_fn(
                                                                title=f"⚠️【{mode_label}投票エラー】{v_name}{r_no}R",
                                                                body=f"自動投票に失敗しました: {err_msg}",
                                                                data={
                                                                    "type": "AUTO_BET_FAILED",
                                                                    "race_id": r_id,
                                                                    "error": str(err_msg),
                                                                    "mode": cur_bet_mode
                                                                }
                                                            )
                                                except Exception as ab_err:
                                                    print(f"[AutoBet] 例外エラー: {ab_err}")
                                                    if _send_fcm_notification_fn:
                                                        _send_fcm_notification_fn(
                                                            title=f"⚠️【{mode_label}投票例外】{r_name}",
                                                            body=f"投票通信中に例外が発生しました: {ab_err}",
                                                            data={
                                                                "type": "AUTO_BET_FAILED",
                                                                "race_id": r_id,
                                                                "error": str(ab_err),
                                                                "mode": cur_bet_mode
                                                            }
                                                        )
                            except Exception as pe:
                                print(f"[Auto-Predict Stage5] 最終推論失敗 ({r_id}): {pe}")
                            finally:
                                gc.collect()

                        # Stage 4: 発走 15分前 (6.5 < mins <= 17.0) -> 本馬場入場後オッズ更新
                        elif 6.5 < mins_to_post <= 17.0 and cur_stage < 4:
                            predicted_count_this_loop += 1
                            print(f"[Auto-Predict Stage4: 15分前直前更新] {r_id} ({r.get('race_name')}) の15分前オッズ更新を実行します...")
                            try:
                                if _predict_race_core_fn:
                                    await asyncio.to_thread(_predict_race_core_fn, r_id, True, False, "15m")
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 4})\
                                    .eq("race_id", r_id)\
                                    .execute()
                            except Exception as pe:
                                print(f"[Auto-Predict Stage4] 更新失敗 ({r_id}): {pe}")
                            finally:
                                gc.collect()

                        # Stage 3: 発走 30分前 (17.0 < mins <= 33.0) -> パドック後オッズ更新
                        elif 17.0 < mins_to_post <= 33.0 and cur_stage < 3:
                            predicted_count_this_loop += 1
                            print(f"[Auto-Predict Stage3: 30分前パドック更新] {r_id} ({r.get('race_name')}) の30分前オッズ更新を実行します...")
                            try:
                                if _predict_race_core_fn:
                                    await asyncio.to_thread(_predict_race_core_fn, r_id, True, False, "30m")
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 3})\
                                    .eq("race_id", r_id)\
                                    .execute()
                            except Exception as pe:
                                print(f"[Auto-Predict Stage3] 更新失敗 ({r_id}): {pe}")
                            finally:
                                gc.collect()

                        # Stage 2: 発走 58分前〜33分前 -> 馬体重反映・第1回再計算 (馬体重未発表時の毎ループ再試行を防止し最低5分間隔で実行)
                        elif 33.0 < mins_to_post <= 58.0 and cur_stage < 2:
                            now_ts = now_dt.timestamp()
                            last_chk = _last_stage2_check.get(r_id, 0.0)
                            if now_ts - last_chk < 300.0:  # 5分以内にチェック済みならスキップ
                                continue
                            _last_stage2_check[r_id] = now_ts
                            predicted_count_this_loop += 1
                            try:
                                if _predict_race_core_fn:
                                    pred_s2 = await asyncio.to_thread(_predict_race_core_fn, r_id, True, False, "60m")
                                    has_w = any(h.get("horse_weight") is not None for h in (pred_s2 or {}).get("horses", []))
                                    if has_w:
                                        supabase.table("keiba_scheduled_races")\
                                            .update({"is_predicted": 2})\
                                            .eq("race_id", r_id)\
                                            .execute()
                                        print(f"[Auto-Predict Stage2] {r_id}: 馬体重反映完了")
                            except Exception as we:
                                print(f"[Auto-Predict Stage2] 馬体重更新チェック失敗 ({r_id}): {we}")
                            finally:
                                gc.collect()
                except Exception as e:
                    print(f"[Scheduler] 予想ループエラー: {e}")

                # --- ③-B 発走後レースで馬体重が未取得のまま残っている場合の自動バックグラウンド補完 ---
                try:
                    res_all_sched = supabase.table("keiba_scheduled_races").select("race_id, post_datetime").eq("race_date", today_str).execute()
                    past_rids_today = []
                    for sr in (res_all_sched.data or []):
                        sr_id = sr.get("race_id", "")
                        if sr_id in healed_post_weight_rids or sr_id[4:6] not in {"04", "05", "08"}:
                            continue
                        p_dt = _parse_post_datetime_fn(sr.get("post_datetime"), today_str) if _parse_post_datetime_fn else None
                        if p_dt and now_dt >= p_dt:
                            past_rids_today.append(sr_id)
                    if past_rids_today:
                        p_chk = supabase.table("keiba_predictions_cache").select("race_id, data_json").in_("race_id", past_rids_today).execute()
                        healed_cnt = 0
                        for prow in (p_chk.data or []):
                            prid = prow.get("race_id")
                            pdj = prow.get("data_json") or {}
                            phorses = pdj.get("horses", [])
                            if phorses and any(h.get("horse_weight") is not None for h in phorses):
                                healed_post_weight_rids.add(prid)
                            elif phorses:
                                if healed_cnt < 2 and _predict_race_core_fn:
                                    await asyncio.to_thread(_predict_race_core_fn, prid, False, False, None)
                                    healed_cnt += 1
                                    healed_post_weight_rids.add(prid)
                                    gc.collect()
                except Exception as he_loop:
                    print(f"[Scheduler] 発走後馬体重補完チェック警告: {he_loop}")

                # 予想更新時に即PAT残高不足の恐れがあればアラート通知
                if _check_and_notify_balance_shortfall_fn:
                    _check_and_notify_balance_shortfall_fn()

                # --- ④ 各レース【発走 10分後】の個別自動照合（未確定時は3分おきにリトライ） ---
                try:
                    res_settle = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("race_date", today_str)\
                        .eq("is_settled", 0)\
                        .execute()
                    unsettled_races = res_settle.data or []

                    settled_count_this_loop = 0
                    MAX_SETTLEMENTS_PER_LOOP = 6

                    for r in unsettled_races:
                        if settled_count_this_loop >= MAX_SETTLEMENTS_PER_LOOP:
                            break

                        r_id = r["race_id"]
                        # 【最重要ガード】JRA公式の本日会場コード (04: 新潟, 05: 東京, 08: 京都) 以外の不正レースは除外
                        venue_code = r_id[4:6] if len(r_id) == 12 else ""
                        if venue_code not in {"04", "05", "08"}:
                            continue

                        p_dt_str = r.get("post_datetime")
                        retries = r.get("retry_count", 0) or 0
                        post_dt = _parse_post_datetime_fn(p_dt_str, today_str) if _parse_post_datetime_fn else None
                        if not post_dt:
                            continue

                        # 初回は発走10分後、以降は3分間隔でリトライ
                        trigger_settle_time = post_dt + datetime.timedelta(minutes=10 + 3 * retries)
                        if now_dt >= trigger_settle_time:
                            settled_count_this_loop += 1
                            try:
                                if _verify_race_result_fn:
                                    await asyncio.to_thread(_verify_race_result_fn, r_id, True, False)
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_settled": 1})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Settle] 結果照合完了: {r_id}")
                            except Exception as se_err:
                                print(f"[Auto-Settle Retry] {r_id} (retry={retries}): {se_err}")
                                if retries >= 15:
                                    supabase.table("keiba_scheduled_races")\
                                        .update({"is_settled": 1})\
                                        .eq("race_id", r_id)\
                                        .execute()
                                else:
                                    supabase.table("keiba_scheduled_races")\
                                        .update({"retry_count": retries + 1})\
                                        .eq("race_id", r_id)\
                                        .execute()
                            finally:
                                gc.collect()
                except Exception as e:
                    print(f"[Scheduler] 照合ループエラー: {e}")

            # --- ⑤ JST 17:00 および 19:00 の当日最終未確定一括照合 ---
            period_dates = _discover_period_kaisai_dates_fn(now_dt) if _discover_period_kaisai_dates_fn else []
            if now_dt.hour == 17 and last_17h_settle_date != today_str:
                if today_str in period_dates:
                    if not batch_status["is_running"]:
                        print(f"[Scheduler] 17:00 当日レースの最終結果照合を実行します...")
                        loop = asyncio.get_event_loop()
                        if _run_batch_worker_fn:
                            await loop.run_in_executor(None, lambda: _run_batch_worker_fn(today_str, "settle", False, True))
                        last_17h_settle_date = today_str
                else:
                    last_17h_settle_date = today_str

            if now_dt.hour == 19 and last_19h_settle_date != today_str:
                if today_str in period_dates:
                    if not batch_status["is_running"]:
                        print(f"[Scheduler] 19:00 当日レースの予備最終結果照合を実行します...")
                        loop = asyncio.get_event_loop()
                        if _run_batch_worker_fn:
                            await loop.run_in_executor(None, lambda: _run_batch_worker_fn(today_str, "settle", False, True))
                        last_19h_settle_date = today_str
                else:
                    last_19h_settle_date = today_str

        except Exception as e:
            print(f"[Scheduler] 全体ループエラー: {e}")

        await asyncio.sleep(60)

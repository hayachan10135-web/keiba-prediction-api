# -*- coding: utf-8 -*-
"""
app/maintenance/healer.py

データ修復・自己回復・スキーマバージョン管理・過去開催一括同期エンジン
"""

import re
import datetime
from typing import Optional, Callable, Dict, Any, List, Set

from app.core.database import get_supabase_client
from app.settlement import (
    invalidate_summary_caches,
    resolve_race_date_for_id,
    process_settlement_calculation,
    extract_top3_orders,
    extract_mark_orders,
    sync_orders_marks,
    save_cached_result,
)
from app.scrapers import fetch_shutuba_table, fetch_venues_for_date_raw, fetch_netkeiba_race_result

CURRENT_DB_VERSION = "v5_umaren_3point_strategy"

# 外部DI
_today_jst_str_fn: Optional[Callable] = None
_now_jst_naive_fn: Optional[Callable] = None
_parse_post_datetime_fn: Optional[Callable] = None
_get_today_races_fn: Optional[Callable] = None
_predict_race_core_fn: Optional[Callable] = None
_verify_race_result_fn: Optional[Callable] = None
_ensure_daily_single_real_bet_fn: Optional[Callable] = None
_infer_odds_stage_from_minutes_fn: Optional[Callable] = None
_get_cached_prediction_fn: Optional[Callable] = None
_get_cached_result_fn: Optional[Callable] = None
_all_predictions_cache_ref: Optional[Dict[str, Any]] = None
_today_races_cache_ref: Optional[Dict[str, Any]] = None
_last_weight_check_ts_ref: Optional[Dict[str, float]] = None


def set_healer_dependencies(
    today_jst_str: Callable,
    now_jst_naive: Callable,
    parse_post_datetime: Callable,
    get_today_races: Callable,
    predict_race_core: Callable,
    verify_race_result: Callable,
    ensure_daily_single_real_bet: Callable,
    infer_odds_stage_from_minutes: Callable,
    get_cached_prediction: Callable,
    get_cached_result: Callable,
    all_predictions_cache: Dict[str, Any],
    today_races_cache: Dict[str, Any],
    last_weight_check_ts: Dict[str, float],
):
    """データ修復エンジンが必要とする外部依存を登録"""
    global _today_jst_str_fn, _now_jst_naive_fn, _parse_post_datetime_fn
    global _get_today_races_fn, _predict_race_core_fn, _verify_race_result_fn
    global _ensure_daily_single_real_bet_fn, _infer_odds_stage_from_minutes_fn
    global _get_cached_prediction_fn, _get_cached_result_fn
    global _all_predictions_cache_ref, _today_races_cache_ref, _last_weight_check_ts_ref

    _today_jst_str_fn = today_jst_str
    _now_jst_naive_fn = now_jst_naive
    _parse_post_datetime_fn = parse_post_datetime
    _get_today_races_fn = get_today_races
    _predict_race_core_fn = predict_race_core
    _verify_race_result_fn = verify_race_result
    _ensure_daily_single_real_bet_fn = ensure_daily_single_real_bet
    _infer_odds_stage_from_minutes_fn = infer_odds_stage_from_minutes
    _get_cached_prediction_fn = get_cached_prediction
    _get_cached_result_fn = get_cached_result
    _all_predictions_cache_ref = all_predictions_cache
    _today_races_cache_ref = today_races_cache
    _last_weight_check_ts_ref = last_weight_check_ts


def fix_mislabeled_20261011_races():
    """10/11に誤登録された10/12(0405: 4回5日目)レースを10/12へ移行し、10/11当日の0404レースを再登録"""
    supabase = get_supabase_client()
    if not supabase:
        print("[Fix] Supabase未接続のため修正処理をスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}

    try:
        print("[Fix] 10/11誤登録データの修正・移行処理を開始します...")

        # 1. keiba_scheduled_races: 10/11に登録された0405レースを10/12に移行
        res_sched = supabase.table("keiba_scheduled_races").select("*").eq("race_date", "2026-10-11").execute()
        sched_rows = res_sched.data or []
        migrated_sched = 0
        for r in sched_rows:
            r_id = r["race_id"]
            if len(r_id) == 12 and r_id[6:10] == "0405":
                new_post_dt = r["post_datetime"].replace("2026-10-11", "2026-10-12")
                supabase.table("keiba_scheduled_races").update({
                    "race_date": "2026-10-12",
                    "post_datetime": new_post_dt,
                    "is_settled": 0,
                    "retry_count": 0,
                    "is_predicted": 0
                }).eq("race_id", r_id).execute()
                migrated_sched += 1

        # 2. keiba_predictions_cache: 0405レースのrace_dateを2026-10-12に移行
        res_cache = supabase.table("keiba_predictions_cache").select("race_id").eq("race_date", "2026-10-11").execute()
        cache_rows = res_cache.data or []
        migrated_cache = 0
        for c in cache_rows:
            r_id = c["race_id"]
            if len(r_id) == 12 and r_id[6:10] == "0405":
                supabase.table("keiba_predictions_cache").update({
                    "race_date": "2026-10-12"
                }).eq("race_id", r_id).execute()
                migrated_cache += 1

        # 3. keiba_race_settlements: 0405レースが誤って保存されていれば削除
        try:
            supabase.table("keiba_race_settlements").delete().filter("race_date", "eq", "2026-10-11").like("race_id", "%0405%").execute()
        except Exception:
            pass

        # 4. keiba_race_results_cache: 0405レースが誤って保存されていれば削除
        try:
            supabase.table("keiba_race_results_cache").delete().filter("race_date", "eq", "2026-10-11").like("race_id", "%0405%").execute()
        except Exception:
            pass

        # 5. 正しい本日 (2026-10-11 / 0404) のレーススケジュールを即時取得・登録
        today_info = _get_today_races_fn("20261011") if _get_today_races_fn else {}
        venues = today_info.get("venues", [])
        synced_count = 0
        for v in venues:
            for r in v.get("races", []):
                r_id = r["race_id"]
                try:
                    fetch_shutuba_table(r_id, race_date="2026-10-11")
                    synced_count += 1
                except Exception as fe:
                    print(f"[Fix] 出馬表登録エラー ({r_id}): {fe}")

        msg = f"修正完了: 10/12移行(スケジュール={migrated_sched}件, 予想キャッシュ={migrated_cache}件), 本日(10/11)登録={synced_count}件"
        print(f"[Fix] {msg}")
        return {"status": "ok", "message": msg}
    except Exception as e:
        err_msg = f"[Fix] 修正処理エラー: {e}"
        print(err_msg)
        return {"status": "error", "message": str(e)}


def clear_all_database(sync_today: bool = False):
    """Supabaseの全テーブル（メタ、キャッシュ、台帳、スケジュール）を完全クリア・初期化"""
    supabase = get_supabase_client()
    if not supabase:
        print("[ClearDB] Supabase未接続のためスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}
    try:
        print("[ClearDB] Supabase全テーブルの初期化クリーンアップを開始します...")
        for tbl in ["keiba_predictions_cache", "keiba_race_results_cache", "keiba_race_settlements", "keiba_scheduled_races"]:
            try:
                supabase.table(tbl).delete().neq("race_id", "0").execute()
            except Exception as e:
                print(f"[ClearDB] {tbl} clear warning: {e}")
        try:
            supabase.table("keiba_app_meta").delete().neq("key", "__keep__").neq("key", "stats_overlay").execute()
        except Exception as e:
            print(f"[ClearDB] app_meta clear warning: {e}")

        invalidate_summary_caches()

        synced = 0
        if sync_today:
            today_str = _today_jst_str_fn() if _today_jst_str_fn else ""
            today_clean = today_str.replace("-", "")
            today_info = _get_today_races_fn(today_clean) if _get_today_races_fn else {}
            venues = today_info.get("venues", [])
            for v in venues:
                for r in v.get("races", []):
                    try:
                        fetch_shutuba_table(r["race_id"], race_date=today_str)
                        synced += 1
                    except Exception:
                        pass

        msg = f"全DBテーブル初期化完了 (本日同期: {synced}レース)"
        print(f"[ClearDB] {msg}")
        return {"status": "ok", "message": msg, "synced_races": synced}
    except Exception as e:
        err_msg = f"[ClearDB] 初期化エラー: {e}"
        print(err_msg)
        return {"status": "error", "message": str(e)}


def check_and_apply_db_migration():
    """DB構成・スキーマバージョンが変更された場合のみ、旧データをまっさらにしてバージョンを更新"""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        saved_version = None
        try:
            m_res = supabase.table("keiba_app_meta").select("value").eq("key", "schema_version").execute()
            if m_res.data and len(m_res.data) > 0:
                saved_version = m_res.data[0]["value"].get("version")
        except Exception:
            pass

        if not saved_version:
            try:
                res = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", "SCHEMA_VERSION").execute()
                if res.data and len(res.data) > 0:
                    saved_version = res.data[0]["data_json"].get("version")
            except Exception:
                pass

        today_str = _today_jst_str_fn() if _today_jst_str_fn else ""
        if saved_version != CURRENT_DB_VERSION:
            print(f"[DB Migration] DB構成変更検知 ({saved_version} -> {CURRENT_DB_VERSION}): DBを初期化します。")
            clear_all_database()
            try:
                supabase.table("keiba_app_meta").upsert({
                    "key": "schema_version",
                    "value": {"version": CURRENT_DB_VERSION, "updated_at": today_str}
                }).execute()
            except Exception:
                pass
            try:
                supabase.table("keiba_predictions_cache").upsert({
                    "race_id": "SCHEMA_VERSION",
                    "race_date": "9999-12-31",
                    "data_json": {"version": CURRENT_DB_VERSION, "updated_at": today_str}
                }).execute()
            except Exception:
                pass
            print(f"[DB Migration] 新バージョン {CURRENT_DB_VERSION} を正常に保存しました。")
        else:
            print(f"[DB Migration] DBバージョン ({saved_version}) は最新です。データ初期化はスキップします。")
    except Exception as e:
        print(f"[DB Migration] バージョン確認エラー: {e}")


def normalize_mislabeled_dates():
    """2026-10-13等のフォールバック日および新潟0401の誤認日(2026-10-03)を正規の開催日へ修正"""
    supabase = get_supabase_client()
    if not supabase:
        print("[NormalizeDates] Supabase未接続のためスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}

    migrated_settlements = 0
    migrated_results = 0
    migrated_predictions = 0

    try:
        # 1. 4回新潟1日目 (2026040401xx) が 2026-10-03 と誤保存されているものを 2026-10-17 へ修正
        for tbl_name in ("keiba_race_settlements", "keiba_race_results_cache", "keiba_predictions_cache", "keiba_scheduled_races"):
            try:
                res_niigata = supabase.table(tbl_name).select("race_id, race_date").like("race_id", "2026040401%").neq("race_date", "2026-10-17").execute()
                for nrow in (res_niigata.data or []):
                    nrid = nrow.get("race_id")
                    if nrid:
                        supabase.table(tbl_name).update({"race_date": "2026-10-17"}).eq("race_id", nrid).execute()
                        if tbl_name == "keiba_race_settlements":
                            migrated_settlements += 1
            except Exception as ne:
                print(f"[NormalizeDates] Niigata 0401 fix warning ({tbl_name}): {ne}")

        res_s = supabase.table("keiba_race_settlements").select("race_id, race_date").eq("race_date", "2026-10-13").execute()
        rows_s = res_s.data or []
        for row in rows_s:
            r_id = row.get("race_id", "")
            correct_date = resolve_race_date_for_id(r_id)

            if correct_date and correct_date != "2026-10-13":
                supabase.table("keiba_race_settlements").update({"race_date": correct_date}).eq("race_id", r_id).execute()
                migrated_settlements += 1

                try:
                    supabase.table("keiba_race_results_cache").update({"race_date": correct_date}).eq("race_id", r_id).execute()
                    migrated_results += 1
                except Exception:
                    pass
                try:
                    supabase.table("keiba_predictions_cache").update({"race_date": correct_date}).eq("race_id", r_id).execute()
                    migrated_predictions += 1
                except Exception:
                    pass

        # keiba_scheduled_races の 2026-10-13 レコードも更新・整理
        try:
            res_sc = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", "2026-10-13").execute()
            for row_sc in (res_sc.data or []):
                sc_id = row_sc.get("race_id", "")
                c_date = resolve_race_date_for_id(sc_id)
                if c_date and c_date != "2026-10-13":
                    supabase.table("keiba_scheduled_races").update({"race_date": c_date}).eq("race_id", sc_id).execute()
                else:
                    supabase.table("keiba_scheduled_races").delete().eq("race_id", sc_id).execute()
        except Exception as e:
            print(f"[NormalizeDates] scheduled_races cleanup error: {e}")

        if migrated_settlements > 0 or migrated_results > 0 or migrated_predictions > 0:
            invalidate_summary_caches()

        msg = f"日付正規化完了: settlements={migrated_settlements}件, results={migrated_results}件, predictions={migrated_predictions}件"
        print(f"[NormalizeDates] {msg}")
        return {"status": "ok", "message": msg}
    except Exception as e:
        err_msg = f"[NormalizeDates] 修正処理エラー: {e}"
        print(err_msg)
        return {"status": "error", "message": str(e)}


def sync_and_process_target_dates(target_dates: List[str]):
    """指定開催日（例: ['2026-09-26', '2026-09-27']）の全レースを keiba_scheduled_races に登録し、全レース予想＆結果検証を実行"""
    supabase = get_supabase_client()
    if not supabase:
        print("[Process Dates] Supabase未接続")
        return {"status": "error", "message": "Supabase not connected"}

    total_scheduled = 0
    total_predicted = 0
    total_settled = 0
    all_races = []

    for d_str in target_dates:
        venues = fetch_venues_for_date_raw(d_str)
        race_items = [r for v in venues for r in v.get("races", [])]
        rows_to_upsert = []
        for r in race_items:
            r_id = r["race_id"]
            r_info = r.get("race_info") or r.get("race_name") or ""
            m = re.search(r"(\d{1,2}):(\d{2})", r_info)
            hour, minute = (int(m.group(1)), int(m.group(2))) if m else (15, 0)
            d_parts = [int(p) for p in d_str.split("-")]
            post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
            rows_to_upsert.append({
                "race_id": r_id,
                "race_date": d_str,
                "race_name": r.get("race_name", r.get("race_no", "")),
                "post_datetime": post_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "is_predicted": 0,
                "is_settled": 0,
                "retry_count": 0
            })
            all_races.append((d_str, r_id))
        if rows_to_upsert:
            try:
                supabase.table("keiba_scheduled_races").upsert(rows_to_upsert, on_conflict="race_id").execute()
                total_scheduled += len(rows_to_upsert)
            except Exception as e:
                print(f"[Schedule] upsert error for {d_str}: {e}")

    # 全レースの予想 ＆ 結果検証を一括実行
    for d_str, r_id in all_races:
        if _predict_race_core_fn:
            try:
                _predict_race_core_fn(r_id, force_refresh=True)
                total_predicted += 1
                if supabase:
                    supabase.table("keiba_scheduled_races").update({"is_predicted": 2}).eq("race_id", r_id).execute()
            except Exception as pe:
                print(f"[Predict Error] {r_id}: {pe}")

        if _verify_race_result_fn:
            try:
                _verify_race_result_fn(r_id, bypass_cache=True)
                total_settled += 1
                if supabase:
                    supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", r_id).execute()
            except Exception as se:
                print(f"[Settle Error] {r_id}: {se}")

    msg = f"{', '.join(target_dates)} 全レース処理完了: スケジュール登録={total_scheduled}R, 予想={total_predicted}R, 検証={total_settled}R"
    print(f"[Batch Process] {msg}")
    return {
        "status": "ok",
        "message": msg,
        "scheduled": total_scheduled,
        "predicted": total_predicted,
        "settled": total_settled
    }


def fix_duplicated_horses_cache_2026():
    """出馬表パース時の重複取得（1番二重登録等）を含む予測キャッシュを自動検知してクリア（当日分のみ高速チェック）"""
    supabase = get_supabase_client()
    if not supabase:
        return
    today_str = _today_jst_str_fn() if _today_jst_str_fn else ""
    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", today_str).execute()
        rids_to_clear = []
        for row in (res.data or []):
            rid = row.get("race_id")
            dj = row.get("data_json") or {}
            horses = dj.get("horses", [])
            umaban_list = [h.get("umaban") for h in horses if "umaban" in h and h.get("umaban") is not None]
            if len(umaban_list) != len(set(umaban_list)):
                rids_to_clear.append(rid)
        if rids_to_clear:
            print(f"[FixDuplicates] 馬番重複を検知した {len(rids_to_clear)} レースのキャッシュをリセットします: {rids_to_clear}")
            supabase.table("keiba_predictions_cache").delete().in_("race_id", rids_to_clear).execute()
    except Exception as e:
        print(f"[FixDuplicates] クリーンアップ例外: {e}")


def heal_today_weights_and_results(target_date: Optional[str] = None, force_resettle: bool = False) -> dict:
    """指定日(デフォルト本日)の馬体重未反映レースおよび結果未検証レースを一括補完・検証"""
    supabase = get_supabase_client()
    if not supabase:
        return {"status": "error", "message": "Supabase not connected"}

    today_str = _today_jst_str_fn() if _today_jst_str_fn else ""
    t_date = target_date or today_str
    normalize_mislabeled_dates()
    if _ensure_daily_single_real_bet_fn:
        _ensure_daily_single_real_bet_fn(t_date)

    now_dt = _now_jst_naive_fn() if _now_jst_naive_fn else datetime.datetime.now()
    sched_res = supabase.table("keiba_scheduled_races").select("*").eq("race_date", t_date).order("race_id").execute()
    raw_sched_rows = sched_res.data or []
    # 【最重要ガード】JRA公式会場コード (04: 新潟, 05: 東京, 08: 京都) の正規レースのみを処理
    sched_rows = [
        sr for sr in raw_sched_rows
        if len(sr.get("race_id", "")) == 12 and sr.get("race_id", "")[4:6] in {"04", "05", "08"}
    ]
    valid_rids = [sr["race_id"] for sr in sched_rows]

    pred_map = {}
    if valid_rids:
        pred_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").in_("race_id", valid_rids).execute()
        pred_map = {r["race_id"]: (r.get("data_json") or {}) for r in (pred_res.data or []) if r.get("race_id")}

    settle_res = supabase.table("keiba_race_settlements").select("race_id, actual_order, is_pass, target_umaban").eq("race_date", t_date).execute()
    settle_rows = settle_res.data or []
    settled_rids = {
        r["race_id"] for r in settle_rows
        if r.get("actual_order") is not None or r.get("is_pass") in (1, True)
    }
    settle_pass_map = {r["race_id"]: r.get("is_pass") for r in settle_rows if r.get("race_id")}
    settle_target_map = {r["race_id"]: r.get("target_umaban") for r in settle_rows if r.get("race_id")}

    healed_weights = []
    settled_now = []
    errors = []

    for sr in sched_rows:
        rid = sr["race_id"]
        p_dt_str = sr.get("post_datetime", "")
        post_dt = _parse_post_datetime_fn(p_dt_str, t_date) if _parse_post_datetime_fn else None
        if not post_dt:
            post_dt = now_dt

        mins_to_post = (post_dt - now_dt).total_seconds() / 60.0 if t_date == today_str else -999.0
        dj = pred_map.get(rid) or {}
        horses = dj.get("horses", [])
        has_weights = any(h.get("horse_weight") is not None for h in horses)

        # 1. 馬体重の補完・最新化 (発走58分前以降の全レース)
        if not has_weights and mins_to_post <= 58.0:
            try:
                if _last_weight_check_ts_ref is not None:
                    _last_weight_check_ts_ref.pop(rid, None)
                if mins_to_post > 0:
                    stg = _infer_odds_stage_from_minutes_fn(mins_to_post) if _infer_odds_stage_from_minutes_fn else "0830"
                    updated_p = _predict_race_core_fn(rid, force_refresh=True, is_auto_final=(stg == "05m"), odds_stage=stg) if _predict_race_core_fn else {}
                else:
                    # 発走後は買い目を変えずに馬体重・性齢のみ補完
                    updated_p = _predict_race_core_fn(rid, force_refresh=False, is_auto_final=False, odds_stage=None) if _predict_race_core_fn else {}
                if any(h.get("horse_weight") is not None for h in (updated_p or {}).get("horses", [])):
                    healed_weights.append(rid)
            except Exception as we:
                errors.append(f"weight({rid}): {we}")

        # 2. 結果検証の補完
        is_real_target = bool((dj.get("hybrid_recommendation") or {}).get("is_real_bet_target"))
        is_virt_target = bool(
            sr.get("is_virtual_bet_target")
            or (dj.get("hybrid_recommendation") or {}).get("is_virtual_bet_target")
            or sr.get("real_bet_level") == "virtual"
            or (sr.get("real_bet_badge") and "仮想" in str(sr.get("real_bet_badge")))
        ) and not is_real_target
        pred_target_u = (dj.get("recommendation") or {}).get("target_umaban")
        needs_target_resettle = pred_target_u is not None and (settle_target_map.get(rid) is not None) and (settle_target_map.get(rid) != pred_target_u)
        needs_real_resettle = is_real_target and (settle_pass_map.get(rid) in (1, True))
        needs_virt_resettle = is_virt_target and (settle_pass_map.get(rid) in (1, True))
        if mins_to_post <= -8.0 and (force_resettle or rid not in settled_rids or int(sr.get("is_settled") or 0) == 0 or needs_real_resettle or needs_virt_resettle or needs_target_resettle):
            try:
                if _verify_race_result_fn:
                    _verify_race_result_fn(rid, bypass_cache=True, is_19h_settle=False)
                supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", rid).execute()
                settled_now.append(rid)
            except Exception as se:
                errors.append(f"settle({rid}): {se}")

    if _all_predictions_cache_ref is not None:
        _all_predictions_cache_ref.pop(t_date, None)
    if _today_races_cache_ref is not None:
        _today_races_cache_ref.pop(t_date, None)
    invalidate_summary_caches()
    del pred_map
    gc.collect()

    return {
        "status": "ok",
        "date": t_date,
        "healed_weights_count": len(healed_weights),
        "healed_weights_races": healed_weights,
        "settled_now_count": len(settled_now),
        "settled_now_races": settled_now,
        "errors": errors
    }


def heal_settlement_marks(date: Optional[str] = None) -> dict:
    """keiba_race_settlements の印・本命馬・確定着順を最新の predictions_cache と results_cache から走査して一括自己修復"""
    supabase = get_supabase_client()
    if not supabase:
        return {"status": "error", "message": "Supabase not connected"}

    # 全レコードをページネーションで漏れなく取得
    all_rows = []
    page_size = 500
    offset = 0
    while True:
        query = supabase.table("keiba_race_settlements").select("*")
        if date:
            query = query.eq("race_date", date)
        res = query.order("race_id", desc=True).range(offset, offset + page_size - 1).execute()
        batch = res.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    rows = all_rows
    repaired = []

    for r in rows:
        rid = r["race_id"]
        r_date = r.get("race_date") or resolve_race_date_for_id(rid)
        pred = _get_cached_prediction_fn(rid) if _get_cached_prediction_fn else None
        result = _get_cached_result_fn(rid) if _get_cached_result_fn else None

        # 結果キャッシュが未取得の場合は netkeiba からリアルタイム取得
        if not result and r.get("hit_type") != "開催中止":
            try:
                fresh_res = fetch_netkeiba_race_result(rid)
                if fresh_res and fresh_res.get("orders"):
                    result = fresh_res
                    r_name_val = r.get("race_name") or fresh_res.get("race_name", f"Race {rid}")
                    save_cached_result(rid, r_name_val, r_date, fresh_res)
            except Exception as e_res_fetch:
                print(f"[HealMarks Result Fetch Error] {rid}: {e_res_fetch}")

        # 予想キャッシュが未取得の場合は生成
        if not pred and _predict_race_core_fn:
            try:
                pred = _predict_race_core_fn(rid, force_refresh=False, is_auto_final=False)
            except Exception as e_pred_fetch:
                print(f"[HealMarks Pred Fetch Error] {rid}: {e_pred_fetch}")

        if pred and result:
            pred_target = (pred.get("recommendation") or {}).get("target_umaban")
            if pred_target is None:
                for h in pred.get("horses", []):
                    if h.get("mark") == "◎":
                        pred_target = int(h.get("umaban"))
                        break
            orders = result.get("orders", {})
            fresh_top3 = extract_top3_orders(pred, orders)
            fresh_marks = extract_mark_orders(pred, orders)

            needs_update = False
            update_payload = {}

            if pred_target is not None and r.get("target_umaban") != pred_target:
                # 本命馬番が不一致 -> process_settlement_calculation で完全再計算
                try:
                    process_settlement_calculation(
                        race_id=rid,
                        race_date=r_date,
                        race_name=r.get("race_name", f"Race {rid}"),
                        pred_data=pred,
                        race_res=result,
                        from_cache=True
                    )
                    repaired.append({"race_id": rid, "type": "target_umaban_recalculated", "old_target": r.get("target_umaban"), "new_target": pred_target})
                    continue
                except Exception as e_recalc:
                    print(f"[HealMarks Recalc Error] {rid}: {e_recalc}")

            # top3_orders の更新判定
            if fresh_top3 and fresh_top3 != r.get("top3_orders"):
                update_payload["top3_orders"] = fresh_top3
                needs_update = True

            # mark_orders の更新判定 (カラム直接 or bet_details 内)
            current_mo = r.get("mark_orders") or (r.get("bet_details") or {}).get("real_bet_meta", {}).get("mark_orders")
            if fresh_marks and fresh_marks != current_mo:
                b_det = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
                rb_meta = b_det.get("real_bet_meta") if isinstance(b_det.get("real_bet_meta"), dict) else {}
                rb_meta["mark_orders"] = fresh_marks
                rb_meta["top3_orders"] = fresh_top3
                b_det["real_bet_meta"] = rb_meta
                b_det["mark_orders"] = fresh_marks
                b_det["top3_orders"] = fresh_top3
                update_payload["bet_details"] = b_det
                update_payload["mark_orders"] = fresh_marks
                needs_update = True

            if needs_update:
                try:
                    supabase.table("keiba_race_settlements").update(update_payload).eq("race_id", rid).execute()
                    repaired.append({"race_id": rid, "type": "marks_synchronized"})
                except Exception as e_up:
                    # mark_orders カラムが存在しない場合のフォールバック
                    if "mark_orders" in update_payload:
                        update_payload.pop("mark_orders", None)
                        try:
                            supabase.table("keiba_race_settlements").update(update_payload).eq("race_id", rid).execute()
                            repaired.append({"race_id": rid, "type": "marks_synchronized_fallback"})
                        except Exception as e_up2:
                            print(f"[HealMarks Update Fallback Error] {rid}: {e_up2}")
                    else:
                        print(f"[HealMarks Update Error] {rid}: {e_up}")

    invalidate_summary_caches()
    return {
        "status": "ok",
        "scanned_count": len(rows),
        "repaired_count": len(repaired),
        "repaired_details": repaired
    }


# -*- coding: utf-8 -*-
"""
app/settlement/settlement_engine.py

成績確定・着順＆払戻照合・収支精算エンジン
- PayoutVerifier を活用した全券種（単勝・複勝・馬単・馬連・3連単・三連複）の照合
- 実馬券 vs 仮想馬券 vs 見送り（Pass Benefit）の厳密な収支集計と台帳永続化
- 印（◎, ◯, ▲）および確定着順の完全同期
- Supabase (keiba_race_settlements, keiba_race_results_cache) 連携
- 依存性注入 (DI) パターンによる疎結合設計
"""

import re
import datetime
from typing import Optional, List, Dict, Tuple, Any, Callable
from fastapi import HTTPException, Query

from app.core.database import get_supabase_client
from app.core.timezone import today_jst_str
from app.settlement.payout_verifier import PayoutVerifier
from app.settlement.summary_calculator import invalidate_summary_caches
from app.scrapers import (
    fetch_netkeiba_race_result,
    normalize_race_res,
    safe_scrape_get,
    safe_decode_html
)


# DI（依存性注入）ハンドラー群
_update_stats_handler: Optional[Callable[[str, str, dict, dict], Any]] = None
_send_fcm_handler: Optional[Callable[..., Any]] = None
_get_prediction_handler: Optional[Callable[[str], Optional[dict]]] = None
_predict_race_core_handler: Optional[Callable[[str], dict]] = None
_sync_virtual_badge_handler: Optional[Callable[[str, dict, Optional[str]], dict]] = None
_period_venues_cache_ref: Optional[Dict[str, Any]] = None


def set_settlement_engine_dependencies(
    update_stats: Optional[Callable[[str, str, dict, dict], Any]] = None,
    send_fcm: Optional[Callable[..., Any]] = None,
    get_prediction: Optional[Callable[[str], Optional[dict]]] = None,
    predict_race_core: Optional[Callable[[str], dict]] = None,
    sync_virtual_badge: Optional[Callable[[str, dict, Optional[str]], dict]] = None,
    period_venues_cache: Optional[Dict[str, Any]] = None
):
    """外部モジュールとの循環参照を防ぐための依存性注入"""
    global _update_stats_handler, _send_fcm_handler, _get_prediction_handler
    global _predict_race_core_handler, _sync_virtual_badge_handler, _period_venues_cache_ref

    if update_stats is not None:
        _update_stats_handler = update_stats
    if send_fcm is not None:
        _send_fcm_handler = send_fcm
    if get_prediction is not None:
        _get_prediction_handler = get_prediction
    if predict_race_core is not None:
        _predict_race_core_handler = predict_race_core
    if sync_virtual_badge is not None:
        _sync_virtual_badge_handler = sync_virtual_badge
    if period_venues_cache is not None:
        _period_venues_cache_ref = period_venues_cache


def extract_top3_orders(pred_data: Optional[dict], orders: Optional[dict]) -> List[dict]:
    """【確定】 1着・2着・3着の確定馬番・馬名・印情報を抽出 (キーがstr/int両対応)"""
    if not orders:
        return []
    orders_int = {}
    for k, v in orders.items():
        try:
            orders_int[int(k)] = int(v)
        except (ValueError, TypeError):
            pass

    horses = pred_data.get("horses", []) if pred_data else []
    horse_by_num = {int(h["umaban"]): h for h in horses if "umaban" in h and str(h["umaban"]).isdigit()}

    top3 = []
    for rank in [1, 2, 3]:
        for u, o in orders_int.items():
            if o == rank:
                h = horse_by_num.get(u, {})
                mark = h.get("mark", "-")
                if mark in ("〇", "○"):
                    mark = "◯"
                top3.append({
                    "order": rank,
                    "order_text": f"{rank}着",
                    "umaban": u,
                    "horse_name": h.get("horse_name", f"馬{u}"),
                    "mark": mark
                })
    top3.sort(key=lambda x: x["order"])
    return top3[:3]


def extract_mark_orders(pred_data: Optional[dict], orders: Optional[dict]) -> List[dict]:
    """◎, ◯, ▲ の各馬の確定着順情報を抽出 (キーがstr/int両対応)"""
    if not pred_data:
        return []
    horses = pred_data.get("horses", [])
    orders_int = {}
    for k, v in (orders or {}).items():
        try:
            orders_int[int(k)] = int(v)
        except (ValueError, TypeError):
            pass

    target_marks = ["◎", "◯", "▲"]
    found = []

    for mark in target_marks:
        h = next((x for x in horses if x.get("mark") == mark or (mark == "◯" and x.get("mark") in ("〇", "○"))), None)
        if h:
            u = int(h["umaban"])
            name = h.get("horse_name", f"馬{u}")
            ord_num = orders_int.get(u)
            if ord_num is not None and isinstance(ord_num, int) and 1 <= ord_num < 90:
                ord_text = f"{ord_num}着"
            else:
                ord_text = "着外" if orders_int else "未確定"
            found.append({
                "mark": mark,
                "umaban": u,
                "horse_name": name,
                "order": ord_num if (ord_num and ord_num < 90) else None,
                "order_text": ord_text,
                "popularity": h.get("popularity")
            })
    return found


def format_mark_orders_text(mark_orders: List[dict]) -> str:
    """印の着順をテキストに整形"""
    if not mark_orders:
        return ""
    parts = []
    for mo in mark_orders:
        parts.append(f"{mo['mark']}{mo['umaban']}番 {mo['horse_name']}({mo['order_text']})")
    return "【確定着順】 " + " / ".join(parts)


def sync_orders_marks(
    top3_orders: Optional[List[dict]],
    mark_orders: Optional[List[dict]],
    pred_data: Optional[dict]
) -> Tuple[List[dict], List[dict]]:
    """確定着順(top3_orders)および印馬着順(mark_orders)の印・馬名を最新の予想データ(pred_data)の印と完全同期"""
    if not pred_data:
        return top3_orders or [], mark_orders or []
    horses = pred_data.get("horses", [])
    if not horses:
        return top3_orders or [], mark_orders or []

    h_map = {}
    for h in horses:
        if "umaban" in h:
            try:
                h_map[int(h["umaban"])] = h
            except (ValueError, TypeError):
                pass

    synced_top3 = []
    for t in (top3_orders or []):
        t_copy = dict(t)
        try:
            u = int(t_copy.get("umaban", 0))
            if u in h_map:
                m = h_map[u].get("mark", "-")
                if m in ("〇", "○"):
                    m = "◯"
                t_copy["mark"] = m
                if h_map[u].get("horse_name"):
                    t_copy["horse_name"] = h_map[u]["horse_name"]
        except Exception:
            pass
        synced_top3.append(t_copy)

    order_info_by_umaban = {}
    for mo in (mark_orders or []):
        try:
            u = int(mo.get("umaban", 0))
            if u:
                order_info_by_umaban[u] = mo
        except Exception:
            pass

    synced_marks = []
    target_marks = ["◎", "◯", "▲"]
    for mark in target_marks:
        h = next((x for x in horses if x.get("mark") == mark or (mark == "◯" and x.get("mark") in ("〇", "○"))), None)
        if h:
            try:
                u = int(h["umaban"])
                name = h.get("horse_name", f"馬{u}")
                prev_info = order_info_by_umaban.get(u, {})
                synced_marks.append({
                    "mark": mark,
                    "umaban": u,
                    "horse_name": name,
                    "order": prev_info.get("order"),
                    "order_text": prev_info.get("order_text", "未確定"),
                    "popularity": h.get("popularity", prev_info.get("popularity"))
                })
            except Exception:
                pass

    if not synced_marks and mark_orders:
        synced_marks = mark_orders

    return synced_top3, synced_marks


def calculate_hybrid_pass_benefit_info(
    h_rec: Optional[dict],
    race_res: Optional[dict],
    target_u: Optional[int],
    actual_order: Optional[int],
    pred_data: Optional[dict] = None
) -> dict:
    """見送りレースにおける仮想損益 (+200円 - 的中時払戻) の算出"""
    hypo_bet = 200
    hypo_pay = 0
    payouts_data = race_res.get("payouts", {}) if race_res else {}
    tan_map = payouts_data.get("tansho", {})
    fuku_map = payouts_data.get("fukusho", {})

    if pred_data:
        p_rec = pred_data.get("recommendation") or {}
        p_target = p_rec.get("target_umaban")
        if p_target is not None:
            target_u = int(p_target)
        else:
            for h in pred_data.get("horses", []):
                if h.get("mark") == "◎":
                    target_u = int(h.get("umaban"))
                    break

    hit_details = []
    pay_tan = 0
    pay_fuku = 0
    if target_u is not None:
        unit_tan = tan_map.get(target_u, 0) or tan_map.get(str(target_u), 0)
        unit_fuku = fuku_map.get(target_u, 0) or fuku_map.get(str(target_u), 0)
        pay_tan = int(1 * unit_tan)
        pay_fuku = int(1 * unit_fuku)
        hypo_pay = pay_tan + pay_fuku
        if pay_tan > 0:
            hit_details.append(f"単勝{target_u}({pay_tan:,}円)")
        if pay_fuku > 0:
            hit_details.append(f"複勝{target_u}({pay_fuku:,}円)")

    pass_benefit = hypo_bet - hypo_pay

    orders = race_res.get("orders", {}) if race_res else {}
    mark_orders = extract_mark_orders(pred_data, orders)
    top3_orders = extract_top3_orders(pred_data, orders)
    marks_summary = format_mark_orders_text(mark_orders)

    raw_reason = h_rec.get("reason", "期待値基準外") if h_rec else "見送り推奨"
    if "条件不一致" in raw_reason:
        reason_str = "条件不一致 (資金保全のため見送り)"
    else:
        reason_str = raw_reason
    hit_summary = f" ({' ＆ '.join(hit_details)} 的中)" if hit_details else " (不的中・払戻0円)"
    benefit_note = "資金保全成功！" if pass_benefit > 0 else (f"機会損失 (-{abs(pass_benefit):,}円)" if pass_benefit < 0 else "損益なし")
    marks_line = f"\n{marks_summary}" if marks_summary else ""

    msg = f"見送り推奨レースです。(判定: {reason_str})\n【仮想損益】仮想投資: {hypo_bet:,}円 / 仮想払戻: {hypo_pay:,}円{hit_summary} → 見送り損益: {pass_benefit:+,}円 ({benefit_note}){marks_line}"

    return {
        "hypothetical_bet": hypo_bet,
        "hypothetical_payout": hypo_pay,
        "pass_benefit": pass_benefit,
        "pay_tan": pay_tan if target_u is not None else 0,
        "pay_fuku": pay_fuku if target_u is not None else 0,
        "actual_order": actual_order,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "message": msg
    }


def get_cached_result(race_id: str) -> Optional[dict]:
    """Supabaseから結果キャッシュを取得"""
    supabase = get_supabase_client()
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_race_results_cache").select("result_json").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            return normalize_race_res(res.data[0]["result_json"])
    except Exception as e:
        print(f"Supabase結果キャッシュ取得エラー ({race_id}): {e}")
    return None


def save_cached_result(race_id: str, race_name: str, race_date: str, result_dict: dict):
    """Supabaseに結果キャッシュを保存"""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        supabase.table("keiba_race_results_cache").upsert({
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date,
            "result_json": result_dict
        }).execute()
    except Exception as e:
        print(f"Supabase結果キャッシュ保存エラー ({race_id}): {e}")


def save_settlement(data: dict):
    """Supabase収支台帳（keiba_race_settlements）への保存＆サマリーキャッシュ破棄"""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        row_dict = {
            "race_id": data["race_id"],
            "race_date": data["race_date"],
            "race_name": data["race_name"],
            "strategy_type": data.get("strategy_type", "hybrid"),
            "target_umaban": data.get("target_umaban"),
            "horse_name": data.get("horse_name"),
            "grade": data.get("grade"),
            "is_pass": 1 if data.get("is_pass") else 0,
            "umaren_bet": data.get("umaren_bet", 0),
            "trio_bet": data.get("trio_bet", 0),
            "total_bet": data.get("total_bet", 0),
            "actual_order": data.get("actual_order"),
            "umaren_payout": data.get("umaren_payout", 0),
            "trio_payout": data.get("trio_payout", 0),
            "total_payout": data.get("total_payout", 0),
            "is_hit": 1 if data.get("is_hit") else 0,
            "hit_type": data.get("hit_type"),
            "hypothetical_bet": data.get("hypothetical_bet", 0),
            "hypothetical_payout": data.get("hypothetical_payout", 0),
            "pass_benefit": data.get("pass_benefit", 0),
            "bet_breakdown": data.get("bet_breakdown", {}),
            "payout_breakdown": data.get("payout_breakdown", {}),
            "bet_details": data.get("bet_details"),
            "payout_details": data.get("payout_details"),
            "tansho_bet": data.get("tansho_bet", 0),
            "fukusho_bet": data.get("fukusho_bet", 0),
            "tansho_payout": data.get("tansho_payout", 0),
            "fukusho_payout": data.get("fukusho_payout", 0)
        }
        if "track_bias" in data and data["track_bias"] is not None:
            row_dict["track_bias"] = data["track_bias"]
        if "odds_alerts" in data and data["odds_alerts"] is not None:
            row_dict["odds_alerts"] = data["odds_alerts"]
        if "top3_orders" in data and data["top3_orders"] is not None:
            row_dict["top3_orders"] = data["top3_orders"]

        try:
            supabase.table("keiba_race_settlements").upsert(row_dict).execute()
        except Exception as col_err:
            if "track_bias" in row_dict or "odds_alerts" in row_dict or "top3_orders" in row_dict:
                b_det = row_dict.get("bet_details") or {}
                if "track_bias" in row_dict:
                    b_det["track_bias"] = row_dict.pop("track_bias")
                if "odds_alerts" in row_dict:
                    b_det["odds_alerts"] = row_dict.pop("odds_alerts")
                if "top3_orders" in row_dict:
                    b_det["top3_orders"] = row_dict.pop("top3_orders")
                row_dict["bet_details"] = b_det
                supabase.table("keiba_race_settlements").upsert(row_dict).execute()
            else:
                raise col_err
        invalidate_summary_caches()
    except Exception as e:
        print(f"Supabase収支台帳保存エラー: {e}")


def resolve_race_date_for_id(race_id: str) -> str:
    """レースIDに対応する正確な開催日 (YYYY-MM-DD) をキャッシュ・DB・netkeibaから逆引き"""
    if _period_venues_cache_ref:
        for d_cached, c_entry in _period_venues_cache_ref.items():
            v_list = c_entry.get("venues", []) if isinstance(c_entry, dict) else (c_entry if isinstance(c_entry, list) else [])
            for v in v_list:
                if isinstance(v, dict) and any(r.get("race_id") == race_id for r in v.get("races", [])):
                    return d_cached

    # 2026年秋開催の会場別確定日程マッピング
    if len(race_id) == 12 and race_id.startswith("2026"):
        venue_code = race_id[4:6]
        day_code = race_id[6:10]
        if venue_code in ("05", "08"):  # 4回東京(05)・4回京都(08)
            if day_code == "0401":
                return "2026-10-03"
            elif day_code == "0402":
                return "2026-10-04"
            elif day_code == "0403":
                return "2026-10-10"
            elif day_code == "0404":
                return "2026-10-11"
            elif day_code == "0405":
                return "2026-10-12"
            elif day_code == "0406":
                return "2026-10-17"
            elif day_code == "0407":
                return "2026-10-18"
        elif venue_code == "04":  # 4回新潟(04)
            if day_code == "0401":
                return "2026-10-17"
            elif day_code == "0402":
                return "2026-10-18"
        elif venue_code in ("06", "07"):  # 4回中山(06)・4回中京(07)
            if day_code == "0408":
                return "2026-09-26"
            elif day_code == "0409":
                return "2026-09-27"

    supabase = get_supabase_client()
    if supabase:
        for tbl in ("keiba_scheduled_races", "keiba_predictions_cache", "keiba_race_settlements"):
            try:
                res = supabase.table(tbl).select("race_date").eq("race_id", race_id).execute()
                if res.data and len(res.data) > 0 and res.data[0].get("race_date"):
                    r_date = res.data[0]["race_date"]
                    if r_date and r_date != "2026-10-13":
                        return r_date
            except Exception:
                pass

    # netkeibaのHTMLから正確な開催年月日を抽出
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = safe_scrape_get(url, timeout=5.0)
        html = safe_decode_html(resp)
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", html)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    except Exception:
        pass

    return today_jst_str()


def get_cached_settlement(race_id: str) -> Optional[dict]:
    """DBから精算キャッシュを取得し、印の最新化・自己修復・フォーマットを適用"""
    supabase = get_supabase_client()
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_race_settlements").select("*").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            d = res.data[0]
            profit = d["total_payout"] - d["total_bet"]
            recovery = round((d["total_payout"] / d["total_bet"]) * 100.0, 2) if d["total_bet"] > 0 else 0.0
            order = d.get("actual_order")

            hypo_bet = None
            hypo_pay = None
            pass_benefit = None
            mark_orders = None

            hit_type = d.get("hit_type")
            u_bet = d.get("umaren_bet") or 0
            t_bet = d.get("trio_bet") or 0
            u_pay = d.get("umaren_payout") or 0
            t_pay = d.get("trio_payout") or 0

            res_cache = get_cached_result(race_id) or {}
            pred_cache = _get_prediction_handler(race_id) if _get_prediction_handler else None
            orders = res_cache.get("orders", {})
            b_det = d.get("bet_details") if isinstance(d.get("bet_details"), dict) else {}
            rb_meta = b_det.get("real_bet_meta") if isinstance(b_det.get("real_bet_meta"), dict) else {}

            # 本命馬番の不整合チェック＆完全自己修復
            if pred_cache and res_cache:
                p_rec = pred_cache.get("recommendation") or {}
                expected_target_u = p_rec.get("target_umaban")
                if expected_target_u is None:
                    for h in pred_cache.get("horses", []):
                        if h.get("mark") == "◎":
                            expected_target_u = int(h.get("umaban"))
                            break

                if expected_target_u is not None and d.get("target_umaban") != expected_target_u:
                    print(f"[Settlement Target Mismatch Heal] {race_id}: DB target={d.get('target_umaban')} vs Pred target={expected_target_u}. Recalculating settlement...")
                    r_date_val = d.get("race_date") or resolve_race_date_for_id(race_id)
                    r_name_val = d.get("race_name") or f"Race {race_id}"
                    return process_settlement_calculation(
                        race_id=race_id,
                        race_date=r_date_val,
                        race_name=r_name_val,
                        pred_data=pred_cache,
                        race_res=res_cache,
                        from_cache=True
                    )

            # 確定着順(top3_orders)および印馬着順(mark_orders)の最新化＆印同期
            if pred_cache and orders:
                top3_orders = extract_top3_orders(pred_cache, orders)
                mark_orders = extract_mark_orders(pred_cache, orders)
            else:
                top3_orders = d.get("top3_orders") or rb_meta.get("top3_orders")
                mark_orders = d.get("mark_orders") or rb_meta.get("mark_orders")
                if pred_cache:
                    top3_orders, mark_orders = sync_orders_marks(top3_orders, mark_orders, pred_cache)

            # 1〜3着(top3_orders)が未キャッシュの場合は自動取得＆永続化
            if not top3_orders and d.get("hit_type") != "中止":
                try:
                    fresh_res = fetch_netkeiba_race_result(race_id)
                    if fresh_res and fresh_res.get("orders"):
                        res_cache = fresh_res
                        orders = fresh_res.get("orders", {})
                        r_date_val = d.get("race_date") or resolve_race_date_for_id(race_id)
                        r_name_val = d.get("race_name") or fresh_res.get("race_name", f"Race {race_id}")
                        save_cached_result(race_id, r_name_val, r_date_val, fresh_res)
                        top3_orders = extract_top3_orders(pred_cache, orders)
                        healed_marks = extract_mark_orders(pred_cache, orders)
                        if top3_orders:
                            rb_meta["top3_orders"] = top3_orders
                            if healed_marks:
                                rb_meta["mark_orders"] = healed_marks
                            b_det["real_bet_meta"] = rb_meta
                            d["bet_details"] = b_det
                            supabase.table("keiba_race_settlements").update({"bet_details": b_det, "top3_orders": top3_orders}).eq("race_id", race_id).execute()
                except Exception as te:
                    print(f"[Settlement Top3 Heal Warning] {race_id}: {te}")

            if top3_orders and top3_orders != d.get("top3_orders"):
                try:
                    if supabase:
                        supabase.table("keiba_race_settlements").update({"top3_orders": top3_orders}).eq("race_id", race_id).execute()
                        d["top3_orders"] = top3_orders
                except Exception as t3e:
                    print(f"[Settlement Top3 Sync Warning] {race_id}: {t3e}")

            is_virtual_or_pass = bool(
                d.get("is_pass")
                or d.get("strategy_type") == "virtual"
                or (d.get("hit_type") and "仮想" in str(d.get("hit_type")))
            )

            if is_virtual_or_pass:
                if pred_cache and res_cache:
                    h_rec = pred_cache.get("hybrid_recommendation") if pred_cache else None
                    expected_target_u = (pred_cache.get("recommendation") or {}).get("target_umaban")
                    cur_target_u = expected_target_u if expected_target_u is not None else d.get("target_umaban")

                    orders_map = res_cache.get("orders", {})
                    cur_order = orders_map.get(cur_target_u) if cur_target_u is not None else order
                    if cur_order is None and cur_target_u is not None:
                        cur_order = orders_map.get(str(cur_target_u))

                    pass_info = calculate_hybrid_pass_benefit_info(h_rec, res_cache, cur_target_u, cur_order, pred_data=pred_cache)
                    hypo_bet = pass_info["hypothetical_bet"]
                    hypo_pay = pass_info["hypothetical_payout"]
                    pass_benefit = pass_info["pass_benefit"]
                    msg = pass_info["message"]
                    mark_orders = pass_info.get("mark_orders") or mark_orders
                    top3_orders = pass_info.get("top3_orders") or top3_orders

                    is_real = bool(rb_meta.get("is_real_bet_target") or rb_meta.get("real_bet_level") in ["high", "standard"])
                    is_virt = bool(
                        (h_rec or {}).get("is_virtual_bet_target")
                        or rb_meta.get("is_virtual_bet_target")
                        or rb_meta.get("real_bet_level") == "virtual"
                        or (rb_meta.get("real_bet_badge") and "仮想" in str(rb_meta.get("real_bet_badge")))
                        or d.get("strategy_type") == "virtual"
                        or (d.get("hit_type") and "仮想" in str(d.get("hit_type")))
                    ) and not is_real

                    if is_virt:
                        pay_tan = pass_info.get("pay_tan", 0)
                        pay_fuku = pass_info.get("pay_fuku", 0)
                        v_profit = hypo_pay - hypo_bet
                        v_rec = round((hypo_pay / hypo_bet) * 100.0, 1) if hypo_bet > 0 else 0.0
                        is_virt_hit = (hypo_pay > 0)
                        hit_type = "仮想的中" if is_virt_hit else "仮想不的中"
                        marks_summary = format_mark_orders_text(mark_orders)
                        marks_line = f"\n{marks_summary}" if marks_summary else ""
                        if is_virt_hit:
                            v_hits = []
                            if pay_tan > 0: v_hits.append(f"単勝{cur_target_u}({pay_tan:,}円)")
                            if pay_fuku > 0: v_hits.append(f"複勝{cur_target_u}({pay_fuku:,}円)")
                            msg = f"【仮想的中】 {' ＆ '.join(v_hits)}！ 仮想払戻: {hypo_pay:,}円 (収支: {v_profit:+,}円 / 回収率: {v_rec}%){marks_line}"
                        else:
                            ord_info = f" (◎{cur_target_u}番: {cur_order}着)" if (cur_order and cur_order < 90) else ""
                            msg = f"【仮想不的中】{ord_info} (仮想投資: {hypo_bet:,}円 / 払戻: 0円 / 収支: {v_profit:,}円 / 回収率: 0.0%){marks_line}"
                else:
                    hypo_bet = d.get("hypothetical_bet", 200)
                    hypo_pay = d.get("hypothetical_payout", 0)
                    pass_benefit = d.get("pass_benefit", hypo_bet - hypo_pay)
                    is_hit_hypo = (hypo_pay > 0)
                    hit_summary = " 的中" if is_hit_hypo else " 不的中"
                    benefit_note = "資金保全成功！" if pass_benefit > 0 else (f"機会損失 (-{abs(pass_benefit):,}円)" if pass_benefit < 0 else "損益なし")
                    marks_summary = format_mark_orders_text(mark_orders) if mark_orders else ""
                    marks_line = f"\n{marks_summary}" if marks_summary else ""
                    msg = f"見送り推奨レースです。\n【仮想損益】仮想投資: {hypo_bet:,}円 / 仮想払戻: {hypo_pay:,}円 ({hit_summary}) → 見送り損益: {pass_benefit:+,}円 ({benefit_note}){marks_line}"
            else:
                marks_summary = format_mark_orders_text(mark_orders) if mark_orders else ""
                marks_line = f"\n{marks_summary}" if marks_summary else ""
                if d["is_hit"] == 1:
                    hits_summary = []
                    b_det = d.get("bet_details") or {}
                    p_det = d.get("payout_details") or []
                    if isinstance(p_det, list) and len(p_det) > 0:
                        for p in p_det:
                            if isinstance(p, dict):
                                hits_summary.append(f"{p.get('type')}{p.get('comb')}({p.get('payout', 0):,}円)")
                    else:
                        if u_pay > 0: hits_summary.append(f"馬連({u_pay:,}円)")
                        if t_pay > 0: hits_summary.append(f"三連複({t_pay:,}円)")
                    hit_str = " ＆ ".join(hits_summary) if hits_summary else "的中"
                    msg = f"【{hit_type}】{hit_str}！ 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery}%){marks_line}"
                else:
                    target_u = d.get("target_umaban")
                    order_info = f" (◎{target_u}番: {order}着)" if (order and order < 90) else ""
                    msg = f"不的中{order_info} (投資: {d['total_bet']:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%){marks_line}"

            return {
                "race_id": d["race_id"],
                "race_name": d["race_name"],
                "is_pass": bool(d.get("is_pass")),
                "grade": d.get("grade"),
                "target_umaban": d.get("target_umaban"),
                "horse_name": d.get("horse_name"),
                "actual_order": order if (order and order < 90) else None,
                "is_hit": bool(d["is_hit"]),
                "total_bet": d["total_bet"],
                "total_payout": d["total_payout"],
                "profit": profit,
                "recovery_rate": recovery,
                "from_cache": True,
                "message": msg,
                "hypothetical_bet": hypo_bet if hypo_bet is not None else d.get("hypothetical_bet", 0),
                "hypothetical_payout": hypo_pay if hypo_pay is not None else d.get("hypothetical_payout", 0),
                "pass_benefit": pass_benefit if pass_benefit is not None else d.get("pass_benefit", 0),
                "strategy_type": d.get("strategy_type", "hybrid"),
                "hit_type": hit_type,
                "bet_breakdown": d.get("bet_breakdown", {}),
                "payout_breakdown": d.get("payout_breakdown", {}),
                "bet_details": d.get("bet_details"),
                "payout_details": d.get("payout_details"),
                "tansho_bet": d.get("tansho_bet", 0),
                "fukusho_bet": d.get("fukusho_bet", 0),
                "umatan_bet": d.get("umatan_bet", 0),
                "umaren_bet": u_bet,
                "trio_bet": t_bet,
                "tansho_payout": d.get("tansho_payout", 0),
                "fukusho_payout": d.get("fukusho_payout", 0),
                "umatan_payout": d.get("umatan_payout", 0),
                "umaren_payout": u_pay,
                "trio_payout": t_pay,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }
    except Exception as e:
        print(f"Supabase収支取得エラー ({race_id}): {e}")
    return None


def process_settlement_calculation(
    race_id: str,
    race_date: str,
    race_name: str,
    pred_data: dict,
    race_res: dict,
    from_cache: bool = False
) -> dict:
    """レース着順・払戻結果と予想データを突き合わせ、収支・損益・的中判定を確定計算"""
    supabase = get_supabase_client()
    t_bias = pred_data.get("track_bias") if isinstance(pred_data, dict) else None
    o_alerts = [
        {
            "umaban": h.get("umaban"),
            "horse_name": h.get("horse_name"),
            "odds": h.get("odds"),
            "odds_trend": h.get("odds_trend"),
            "odds_alert": h.get("odds_alert"),
            "bias_match": h.get("bias_match"),
            "odds_history": h.get("odds_history")
        }
        for h in (pred_data.get("horses", []) if isinstance(pred_data, dict) else [])
        if h.get("odds_alert") or h.get("odds_trend") or h.get("bias_match") or h.get("odds_history")
    ]

    # 開催中止・取りやめレースの全額返還処理
    if race_res.get("is_cancelled"):
        settlement = {
            "race_id": race_id,
            "race_date": race_date,
            "race_name": race_name,
            "strategy_type": "cancelled",
            "target_umaban": None,
            "horse_name": None,
            "grade": "C",
            "is_pass": True,
            "umaren_bet": 0,
            "trio_bet": 0,
            "total_bet": 0,
            "actual_order": None,
            "umaren_payout": 0,
            "trio_payout": 0,
            "total_payout": 0,
            "is_hit": False,
            "hit_type": "開催中止",
            "hypothetical_bet": 0,
            "hypothetical_payout": 0,
            "pass_benefit": 0,
            "bet_breakdown": {},
            "payout_breakdown": {},
            "bet_details": {
                "real_bet_meta": {
                    "is_real_bet_target": False,
                    "real_bet_badge": "⛔【開催中止】全額返還",
                    "real_bet_level": "none",
                    "mark_orders": [],
                    "top3_orders": []
                }
            },
            "payout_details": None,
            "tansho_bet": 0,
            "fukusho_bet": 0,
            "tansho_payout": 0,
            "fukusho_payout": 0,
            "mark_orders": [],
            "top3_orders": [],
            "track_bias": t_bias,
            "odds_alerts": o_alerts
        }
        save_settlement(settlement)
        if supabase:
            try:
                supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
            except Exception:
                pass

        return {
            "race_id": race_id,
            "race_name": race_name,
            "is_pass": True,
            "grade": "C",
            "target_umaban": None,
            "horse_name": None,
            "actual_order": None,
            "is_hit": False,
            "total_bet": 0,
            "total_payout": 0,
            "profit": 0,
            "recovery_rate": 0.0,
            "from_cache": from_cache,
            "message": "【開催中止】積雪・荒天等のためレース取りやめ（馬券全額返還 / 投資0円・払戻0円）。",
            "hypothetical_bet": 0,
            "hypothetical_payout": 0,
            "pass_benefit": 0,
            "strategy_type": "cancelled",
            "hit_type": "開催中止",
            "bet_breakdown": {},
            "payout_breakdown": {},
            "bet_details": None,
            "payout_details": None,
            "umaren_bet": 0,
            "trio_bet": 0,
            "umaren_payout": 0,
            "trio_payout": 0,
            "mark_orders": [],
            "top3_orders": []
        }

    # レース確定結果から各出走馬の直近走データ・統計（stats_dict / stats_overlay）を最新化
    if _update_stats_handler:
        try:
            _update_stats_handler(race_id, race_date, pred_data, race_res)
        except Exception as e_stats:
            print(f"[StatsOverlay] レース確定統計更新エラー ({race_id}): {e_stats}")

    if _sync_virtual_badge_handler:
        try:
            pred_data = _sync_virtual_badge_handler(race_id, pred_data, race_date)
        except Exception as e_v_sync:
            print(f"[VirtualBadgeSync Error] {race_id}: {e_v_sync}")

    rec = pred_data.get("recommendation", {})
    hybrid_rec = pred_data.get("hybrid_recommendation")

    target_u = rec.get("target_umaban")
    h_name = rec.get("horse_name")
    grade = rec.get("grade", "B")
    orders = race_res.get("orders", {})
    actual_order = orders.get(target_u) if target_u is not None else None
    if actual_order is None and target_u is not None:
        actual_order = orders.get(str(target_u))

    real_bet_meta = {
        "is_real_bet_target": bool((hybrid_rec or {}).get("is_real_bet_target", False)),
        "real_bet_badge": (hybrid_rec or {}).get("real_bet_badge"),
        "real_bet_level": (hybrid_rec or {}).get("real_bet_level", "none"),
        "is_virtual_bet_target": bool((hybrid_rec or {}).get("is_virtual_bet_target", False)),
        "virtual_bet_slot": (hybrid_rec or {}).get("virtual_bet_slot"),
        "edge_ratio": (hybrid_rec or {}).get("edge_ratio"),
        "is_final_odds": bool((hybrid_rec or {}).get("is_final_odds", False))
    }

    # ハイブリッド推奨を最優先で照合（見送りレース または 仮想馬券レース）
    if not hybrid_rec or not hybrid_rec.get("is_participate", False):
        is_real = bool(real_bet_meta.get("is_real_bet_target") or real_bet_meta.get("real_bet_level") in ["high", "standard"])
        is_virt = bool(
            (hybrid_rec or {}).get("is_virtual_bet_target")
            or real_bet_meta.get("is_virtual_bet_target")
            or real_bet_meta.get("real_bet_level") == "virtual"
            or (real_bet_meta.get("real_bet_badge") and "仮想" in str(real_bet_meta.get("real_bet_badge")))
        ) and not is_real

        pass_info = calculate_hybrid_pass_benefit_info(hybrid_rec, race_res, target_u, actual_order, pred_data=pred_data)
        hypo_bet = pass_info["hypothetical_bet"]
        hypo_pay = pass_info["hypothetical_payout"]
        pass_benefit = pass_info["pass_benefit"]
        pay_tan = pass_info.get("pay_tan", 0)
        pay_fuku = pass_info.get("pay_fuku", 0)
        mark_orders = pass_info.get("mark_orders", [])
        top3_orders = pass_info.get("top3_orders", [])
        marks_summary = format_mark_orders_text(mark_orders)
        marks_line = f"\n{marks_summary}" if marks_summary else ""

        real_bet_meta["mark_orders"] = mark_orders
        real_bet_meta["top3_orders"] = top3_orders

        if is_virt:
            v_profit = hypo_pay - hypo_bet
            v_rec = round((hypo_pay / hypo_bet) * 100.0, 1) if hypo_bet > 0 else 0.0
            is_virt_hit = (hypo_pay > 0)
            hit_type_virt = "仮想的中" if is_virt_hit else "仮想不的中"

            p_breakdown = {}
            if pay_tan > 0: p_breakdown["単勝"] = pay_tan
            if pay_fuku > 0: p_breakdown["複勝"] = pay_fuku

            p_details = []
            if pay_tan > 0:
                p_details.append({"type": "単勝", "comb": f"{target_u}", "amount": 100, "payout": pay_tan})
            if pay_fuku > 0:
                p_details.append({"type": "複勝", "comb": f"{target_u}", "amount": 100, "payout": pay_fuku})

            if is_virt_hit:
                hit_texts = [f"{p['type']}{p['comb']}({p['payout']:,}円)" for p in p_details]
                msg_virt = f"【仮想的中】 {' ＆ '.join(hit_texts)}！ 仮想払戻: {hypo_pay:,}円 (収支: {v_profit:+,}円 / 回収率: {v_rec}%){marks_line}"
            else:
                ord_info = f" (◎{target_u}番: {actual_order}着)" if (actual_order and actual_order < 90) else ""
                msg_virt = f"【仮想不的中】{ord_info} (仮想投資: {hypo_bet:,}円 / 払戻: 0円 / 収支: {v_profit:,}円 / 回収率: 0.0%){marks_line}"

            settlement = {
                "race_id": race_id,
                "race_date": race_date,
                "race_name": race_name,
                "strategy_type": "virtual",
                "target_umaban": target_u,
                "horse_name": h_name,
                "grade": grade,
                "is_pass": False,
                "tansho_bet": 100,
                "fukusho_bet": 100,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": hypo_bet,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "tansho_payout": pay_tan,
                "fukusho_payout": pay_fuku,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": hypo_pay,
                "is_hit": is_virt_hit,
                "hit_type": hit_type_virt,
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "bet_breakdown": {"単勝": 100, "複勝": 100},
                "payout_breakdown": p_breakdown,
                "bet_details": {"real_bet_meta": real_bet_meta},
                "payout_details": p_details,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders,
                "track_bias": t_bias,
                "odds_alerts": o_alerts
            }
            save_settlement(settlement)
            if supabase:
                try:
                    supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
                except Exception:
                    pass

            return {
                "race_id": race_id,
                "race_name": race_name,
                "is_pass": False,
                "grade": grade,
                "target_umaban": target_u,
                "horse_name": h_name,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "is_hit": is_virt_hit,
                "total_bet": hypo_bet,
                "total_payout": hypo_pay,
                "profit": v_profit,
                "recovery_rate": v_rec,
                "from_cache": from_cache,
                "message": msg_virt,
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "strategy_type": "virtual",
                "hit_type": hit_type_virt,
                "bet_breakdown": {"単勝": 100, "複勝": 100},
                "payout_breakdown": p_breakdown,
                "bet_details": {"real_bet_meta": real_bet_meta},
                "payout_details": p_details,
                "tansho_bet": 100,
                "fukusho_bet": 100,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "tansho_payout": pay_tan,
                "fukusho_payout": pay_fuku,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }
        else:
            msg = pass_info["message"]
            settlement = {
                "race_id": race_id,
                "race_date": race_date,
                "race_name": race_name,
                "strategy_type": hybrid_rec.get("strategy_type", "skip") if hybrid_rec else "skip",
                "target_umaban": target_u,
                "horse_name": h_name,
                "grade": grade,
                "is_pass": True,
                "tansho_bet": 0,
                "fukusho_bet": 0,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": 0,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "tansho_payout": 0,
                "fukusho_payout": 0,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": 0,
                "is_hit": False,
                "hit_type": "見送り",
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "bet_breakdown": {},
                "payout_breakdown": {},
                "bet_details": {"real_bet_meta": real_bet_meta},
                "payout_details": None,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders,
                "track_bias": t_bias,
                "odds_alerts": o_alerts
            }
            save_settlement(settlement)
            if supabase:
                try:
                    supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
                except Exception:
                    pass

            return {
                "race_id": race_id,
                "race_name": race_name,
                "is_pass": True,
                "grade": grade,
                "target_umaban": target_u,
                "horse_name": h_name,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "is_hit": False,
                "total_bet": 0,
                "total_payout": 0,
                "profit": 0,
                "recovery_rate": 0.0,
                "from_cache": from_cache,
                "message": msg,
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "strategy_type": hybrid_rec.get("strategy_type", "skip") if hybrid_rec else "skip",
                "hit_type": "見送り",
                "bet_breakdown": {},
                "payout_breakdown": {},
                "bet_details": None,
                "payout_details": None,
                "tansho_bet": 0,
                "fukusho_bet": 0,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "tansho_payout": 0,
                "fukusho_payout": 0,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }

    # 参戦レースの場合: PayoutVerifier で全券種を一括照合
    tansho_bets = hybrid_rec.get("tansho_bets") or []
    fukusho_bets = hybrid_rec.get("fukusho_bets") or []
    umatan_bets = hybrid_rec.get("umatan_bets") or []
    umaren_bets = hybrid_rec.get("umaren_bets") or []
    sanrentan_bets = hybrid_rec.get("sanrentan_bets") or []
    trio_bets = hybrid_rec.get("trio_bets") or []
    trio_bet = hybrid_rec.get("trio_bet")
    if not trio_bets and trio_bet and isinstance(trio_bet, dict) and trio_bet.get("combination"):
        trio_bets = [trio_bet]

    all_tickets = []
    for b in tansho_bets:
        all_tickets.append({"ticket_type": "単勝", "combination": b.get("combination", [b.get("target_umaban")]), "amount": b.get("amount", 100)})
    for b in fukusho_bets:
        all_tickets.append({"ticket_type": "複勝", "combination": b.get("combination", [b.get("target_umaban")]), "amount": b.get("amount", 100)})
    for b in umatan_bets:
        all_tickets.append({"ticket_type": "馬単", "combination": b.get("combination", []), "amount": b.get("amount", 100)})
    for b in umaren_bets:
        all_tickets.append({"ticket_type": "馬連", "combination": b.get("combination", []), "amount": b.get("amount", 100)})
    for b in sanrentan_bets:
        all_tickets.append({"ticket_type": "3連単", "combination": b.get("combination", []), "amount": b.get("amount", 100)})
    for b in trio_bets:
        all_tickets.append({"ticket_type": "三連複", "combination": b.get("combination", []), "amount": b.get("amount", 100)})

    payouts_data = race_res.get("payouts", {})
    total_payout, all_hits, bet_breakdown, payout_breakdown = PayoutVerifier.verify_all_tickets(all_tickets, payouts_data)

    tan_bet_tot = bet_breakdown.get("単勝", 0)
    fuku_bet_tot = bet_breakdown.get("複勝", 0)
    utan_bet_tot = bet_breakdown.get("馬単", 0)
    u_bet_tot = bet_breakdown.get("馬連", 0)
    s_bet_tot = bet_breakdown.get("3連単", 0)
    t_bet_tot = bet_breakdown.get("三連複", 0)

    tansho_payout = payout_breakdown.get("単勝", 0)
    fukusho_payout = payout_breakdown.get("複勝", 0)
    umatan_payout = payout_breakdown.get("馬単", 0)
    umaren_payout = payout_breakdown.get("馬連", 0)
    sanrentan_payout = payout_breakdown.get("3連単", 0)
    trio_payout = payout_breakdown.get("三連複", 0)

    total_bet = sum(bet_breakdown.values())
    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = total_payout > 0

    hit_types_found = [h["type"] for h in all_hits]
    if len(all_hits) >= 2:
        hit_type = "W的中🔥"
    elif "馬単" in hit_types_found:
        hit_type = "馬単的中🎯"
    elif "単勝" in hit_types_found:
        hit_type = "単勝的中🎯"
    elif "複勝" in hit_types_found:
        hit_type = "複勝的中🎯"
    elif "3連単" in hit_types_found:
        hit_type = "3連単的中🎯"
    elif "三連複" in hit_types_found:
        hit_type = "三連複的中🎯"
    elif "馬連" in hit_types_found:
        hit_type = "馬連的中🎯"
    else:
        hit_type = "不的中"

    orders = race_res.get("orders", {})
    mark_orders = extract_mark_orders(pred_data, orders)
    top3_orders = extract_top3_orders(pred_data, orders)
    marks_summary = format_mark_orders_text(mark_orders)
    marks_line = f"\n{marks_summary}" if marks_summary else ""

    if is_hit:
        hit_texts = [f"{h['type']}{h['comb']}({h['payout']:,}円)" for h in all_hits]
        msg = f"【{hit_type}】{' ＆ '.join(hit_texts)}！ 払戻: {total_payout:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery_rate}%){marks_line}"
    else:
        order_info = f" (◎{target_u}番: {actual_order}着)" if (actual_order and actual_order < 90) else ""
        msg = f"不的中{order_info} (投資: {total_bet:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%){marks_line}"

    bet_breakdown = {}
    payout_breakdown = {}
    if tan_bet_tot > 0:
        bet_breakdown["単勝"] = tan_bet_tot
        payout_breakdown["単勝"] = tansho_payout
    if fuku_bet_tot > 0:
        bet_breakdown["複勝"] = fuku_bet_tot
        payout_breakdown["複勝"] = fukusho_payout
    if utan_bet_tot > 0:
        bet_breakdown["馬単"] = utan_bet_tot
        payout_breakdown["馬単"] = umatan_payout
    if u_bet_tot > 0:
        bet_breakdown["馬連"] = u_bet_tot
        payout_breakdown["馬連"] = umaren_payout
    if s_bet_tot > 0:
        bet_breakdown["3連単"] = s_bet_tot
        payout_breakdown["3連単"] = sanrentan_payout
    if t_bet_tot > 0:
        bet_breakdown["三連複"] = t_bet_tot
        payout_breakdown["三連複"] = trio_payout

    real_bet_meta["mark_orders"] = mark_orders
    real_bet_meta["top3_orders"] = top3_orders

    settlement = {
        "race_id": race_id,
        "race_date": race_date,
        "race_name": race_name,
        "strategy_type": hybrid_rec.get("strategy_type", "cascade"),
        "target_umaban": target_u,
        "horse_name": h_name,
        "grade": grade,
        "is_pass": False,
        "tansho_bet": tan_bet_tot,
        "fukusho_bet": fuku_bet_tot,
        "umatan_bet": utan_bet_tot,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "total_bet": total_bet,
        "actual_order": actual_order if (actual_order and actual_order < 90) else None,
        "tansho_payout": tansho_payout,
        "fukusho_payout": fukusho_payout,
        "umatan_payout": umatan_payout,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout,
        "total_payout": total_payout,
        "is_hit": is_hit,
        "hit_type": hit_type,
        "hypothetical_bet": 0,
        "hypothetical_payout": 0,
        "pass_benefit": 0,
        "bet_breakdown": bet_breakdown,
        "payout_breakdown": payout_breakdown,
        "bet_details": {
            "tansho_bets": tansho_bets,
            "fukusho_bets": fukusho_bets,
            "umatan_bets": umatan_bets,
            "umaren_bets": umaren_bets,
            "sanrentan_bets": sanrentan_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bet,
            "real_bet_meta": real_bet_meta
        },
        "payout_details": all_hits,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "track_bias": t_bias,
        "odds_alerts": o_alerts
    }
    save_settlement(settlement)
    if supabase:
        try:
            supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
        except Exception:
            pass

    # 実戦推奨レースの結果確定プッシュ通知
    is_today_race = (race_date == today_jst_str())
    if not from_cache and is_today_race and hybrid_rec.get("is_real_bet_target") and _send_fcm_handler:
        try:
            if is_hit:
                _send_fcm_handler(
                    title=f"🎯【実戦的中！】{race_name} 払戻: {total_payout:,}円",
                    body=f"【{hit_type}】収支: {'+' if profit >= 0 else ''}{profit:,}円 (回収率: {recovery_rate}%)",
                    data={"type": "SETTLEMENT_HIT", "race_id": race_id}
                )
            else:
                o_str = f" (◎{target_u}番 {actual_order}着)" if actual_order else ""
                _send_fcm_handler(
                    title=f"🏁【結果確定】{race_name} 不的中",
                    body=f"投資: {total_bet:,}円 / 払戻: 0円{o_str} (収支: {profit:,}円)",
                    data={"type": "SETTLEMENT_MISS", "race_id": race_id}
                )
        except Exception as ne:
            print(f"[FCM通知エラー]: {ne}")

    return {
        "race_id": race_id,
        "race_name": race_name,
        "is_pass": False,
        "grade": grade,
        "target_umaban": target_u,
        "horse_name": h_name,
        "actual_order": actual_order if (actual_order and actual_order < 90) else None,
        "is_hit": is_hit,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "recovery_rate": recovery_rate,
        "from_cache": from_cache,
        "message": msg,
        "strategy_type": hybrid_rec.get("strategy_type", "cascade"),
        "hit_type": hit_type,
        "bet_breakdown": bet_breakdown,
        "payout_breakdown": payout_breakdown,
        "bet_details": {
            "tansho_bets": tansho_bets,
            "fukusho_bets": fukusho_bets,
            "umatan_bets": umatan_bets,
            "umaren_bets": umaren_bets,
            "sanrentan_bets": sanrentan_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bet
        },
        "payout_details": all_hits,
        "tansho_bet": tan_bet_tot,
        "fukusho_bet": fuku_bet_tot,
        "umatan_bet": utan_bet_tot,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "tansho_payout": tansho_payout,
        "fukusho_payout": fukusho_payout,
        "umatan_payout": umatan_payout,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "is_real_bet_target": bool(hybrid_rec.get("is_real_bet_target", False))
    }


def verify_race_result(
    race_id: str,
    bypass_cache: bool = Query(False, description="キャッシュをバイパスして最新結果を取得"),
    is_19h_settle: bool = Query(False, description="19時の一括確定モード")
) -> dict:
    """
    着順・払戻結果を自動スクレイピングし、予想と照合して確定成績を算出・永続化する。
    """
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    if not bypass_cache:
        cached_settlement = get_cached_settlement(race_id)
        if cached_settlement and (
            cached_settlement.get("actual_order") is not None
            or cached_settlement.get("is_pass")
            or cached_settlement.get("hit_type") == "中止"
            or cached_settlement.get("top3_orders")
        ):
            return cached_settlement

    race_date = resolve_race_date_for_id(race_id)

    cached_data = get_cached_result(race_id) if not bypass_cache else None
    from_cache = False

    if cached_data:
        c_orders = cached_data.get("orders", {}) if isinstance(cached_data, dict) else {}
        c_payouts = cached_data.get("payouts", {}) if isinstance(cached_data, dict) else {}
        c_order_vals = [int(v) for v in c_orders.values() if str(v).isdigit()]
        c_has_top3 = (1 in c_order_vals) and (2 in c_order_vals) and (3 in c_order_vals)
        c_has_pay = bool(c_payouts.get("tansho") or c_payouts.get("umaren") or c_payouts.get("fukusho"))
        if (c_has_top3 and c_has_pay) or (isinstance(cached_data, dict) and cached_data.get("is_cancelled")):
            from_cache = True
            race_res = cached_data
        else:
            race_res = fetch_netkeiba_race_result(race_id)
    else:
        race_res = fetch_netkeiba_race_result(race_id)

    if not race_res:
        raise HTTPException(status_code=400, detail="レース結果がまだ確定していないか、取得できませんでした。発走後しばらくしてから再試行してください。")

    # 予想情報を取得 (DIハンドラー経由)
    if not _predict_race_core_handler:
        raise HTTPException(status_code=500, detail="推論コアハンドラーが初期化されていません。")
    pred_data = _predict_race_core_handler(race_id)

    rec = pred_data.get("recommendation", {})
    race_name = pred_data.get("race_name", race_res.get("race_name", f"Race {race_id}"))

    target_u = rec.get("target_umaban")
    orders = race_res.get("orders", {})
    payouts = race_res.get("payouts", {})
    actual_order = orders.get(target_u) if target_u is not None else None
    if actual_order is None and target_u is not None:
        actual_order = orders.get(str(target_u))

    has_top3 = (1 in orders.values()) and (2 in orders.values()) and (3 in orders.values())
    has_payouts = (len(payouts.get("tansho", {})) > 0) or (len(payouts.get("umaren", {})) > 0) or (len(payouts.get("fukusho", {})) > 0)
    is_cancelled = bool(race_res.get("is_cancelled"))

    is_fully_confirmed = (has_top3 and has_payouts) or is_cancelled
    if not is_19h_settle and not is_fully_confirmed:
        raise HTTPException(
            status_code=400,
            detail=f"着順または払戻金が実数値で確定していません（入線直後・審議中等のため未確定）。確定後に再度照合してください。(確認状況: 1〜3着確定={has_top3}, 払戻金確定={has_payouts})"
        )

    save_cached_result(race_id, race_name, race_date, race_res)

    return process_settlement_calculation(
        race_id=race_id,
        race_date=race_date,
        race_name=race_name,
        pred_data=pred_data,
        race_res=race_res,
        from_cache=from_cache
    )

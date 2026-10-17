# -*- coding: utf-8 -*-
"""
app/ml/track_bias.py

リアルタイムトラックバイアス分析エンジン
当日同競馬場・同トラック種別の前半レース結果から、内枠/外枠・先行/差し有利度を動的算出し、各馬のマッチ度を判定する
"""

from typing import Dict, Any, List, Optional


def evaluate_horse_bias_match(
    horse: Dict[str, Any],
    bias_info: Dict[str, Any]
) -> str:
    """
    出走馬の枠番・脚質と馬場バイアス指数とのマッチ度 (bias_match) を判定
    戻り値: "絶好バイアス", "有利", "合致", "逆風", ""
    """
    if not bias_info:
        return ""

    wakuban = int(horse.get("wakuban", 0))
    running_style = str(horse.get("running_style", ""))
    inner_fav = bool(bias_info.get("is_inner_favorable", False))
    outer_fav = bool(bias_info.get("is_outer_favorable", False))
    front_fav = bool(bias_info.get("is_front_favorable", False))

    is_inner = (1 <= wakuban <= 3)
    is_outer = (6 <= wakuban <= 8)
    is_front = any(s in running_style for s in ("逃げ", "先行"))

    if inner_fav and is_inner and front_fav and is_front:
        return "絶好バイアス (内前有利)"
    elif outer_fav and is_outer:
        return "絶好バイアス (外枠有利)"
    elif (inner_fav and is_inner) or (outer_fav and is_outer):
        return "有利 (枠順合致)"
    elif inner_fav and is_outer:
        return "逆風 (外枠逆風)"
    elif outer_fav and is_inner:
        return "逆風 (内枠逆風)"

    return ""


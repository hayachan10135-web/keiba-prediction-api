# -*- coding: utf-8 -*-
"""
app/core/race_utils.py

レース情報（会場名・レース番号・レース名）の厳格な分離・正規化・バリデーションユーティリティ
- Rule 5 (Fail-Fast 原則・サイレントフォールバック禁止)
- race_name から会場・R番号・時刻・頭数・コース条件を完全排除し、純粋なレース名のみを抽出
"""

import re
from typing import Tuple

VENUE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}

VENUE_NAME_TO_CODE = {v: k for k, v in VENUE_CODE_MAP.items()}


def get_venue_name(race_id: str) -> str:
    """
    12桁の race_id から JRA 公式競馬場名を厳格に取得。
    不正なコードの場合はサイレントフォールバックせず ValueError を送出 (Fail-Fast)。
    """
    if not race_id or len(race_id) != 12 or not race_id.isdigit():
        raise ValueError(f"不正な race_id 形式です (12桁数字必須): '{race_id}'")
    v_code = race_id[4:6]
    venue = VENUE_CODE_MAP.get(v_code)
    if not venue:
        raise ValueError(f"未知の JRA 会場コードです: '{v_code}' (race_id: {race_id})")
    return venue


def get_race_no(race_id: str) -> int:
    """
    12桁の race_id からレース番号 (1〜12) を厳格に取得。
    範囲外の場合はサイレントフォールバックせず ValueError を送出 (Fail-Fast)。
    """
    if not race_id or len(race_id) != 12 or not race_id.isdigit():
        raise ValueError(f"不正な race_id 形式です (12桁数字必須): '{race_id}'")
    r_no = int(race_id[10:12])
    if not (1 <= r_no <= 12):
        raise ValueError(f"不正なレース番号です (1〜12必須): {r_no} (race_id: {race_id})")
    return r_no


def clean_pure_race_name(raw_name: str, race_id: str = "") -> str:
    """
    どんな生文字列 (例: '5R 2歳新馬 12:05 ダ1200m 15頭', '札幌2R ２歳未勝利', '中山11R 中山記念(G2)') からも、
    先頭の会場プレフィックス・R番号、末尾の発走時刻・コース・頭数を完全除去し、
    純粋なレース名 ('2歳新馬', '２歳未勝利', '中山記念(G2)') のみを抽出する。
    ※「中山記念」「新潟大賞典」「京都記念」等のレース名自体の会場名は絶対に除去しない。
    """
    if not raw_name:
        return ""
    txt = str(raw_name).strip()

    # 1. 先頭の "会場名 + R番号" または "R番号" プレフィックスを削除
    # 例: "東京11R ", "札幌2R - ", "10R ", "5R - ", "２Ｒ "
    txt = re.sub(r"^(?:東京|中山|京都|阪神|中京|小倉|新潟|福島|札幌|函館)?\s*[0-9０-９]{1,2}\s*[rRＲ]\s*[-－:：]?\s*", "", txt)

    # 2. まれに "東京 11R" のように会場とRの間にスペースがある場合
    txt = re.sub(r"^(?:東京|中山|京都|阪神|中京|小倉|新潟|福島|札幌|函館)\s+[0-9０-９]{1,2}\s*[rRＲ]?\s*[-－:：]?\s*", "", txt)

    # 3. 末尾の発走時刻、コース、頭数を削除 (例: "12:05 ダ1200m 15頭", "14:40 芝1600m 18頭")
    txt = re.sub(r"\s*[0-9]{1,2}:[0-9]{2}\s*(?:発走)?.*$", "", txt)
    txt = re.sub(r"\s*(?:芝|ダ|ダート|障|障害)[0-9]{3,4}m.*$", "", txt)
    txt = re.sub(r"\s*[0-9]{1,2}頭.*$", "", txt)

    # 4. 余分な空白の除去
    txt = " ".join(txt.split()).strip()

    return txt


def extract_race_components(raw_name: str, race_id: str) -> Tuple[str, int, str]:
    """
    (venue_name, race_no, pure_race_name) のタプルを厳格に返却。
    純粋なレース名が空の場合は ValueError を送出 (Fail-Fast)。
    """
    venue = get_venue_name(race_id)
    r_no = get_race_no(race_id)
    p_name = clean_pure_race_name(raw_name, race_id)
    if not p_name:
        raise ValueError(f"レース名を抽出できませんでした (生値: '{raw_name}', race_id: '{race_id}')")
    return venue, r_no, p_name


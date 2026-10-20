# -*- coding: utf-8 -*-
"""
app/core/ticket_spec.py

券種非依存の共通仕様定義 ＆ 券種レジストリ (Unified Ticket Architecture)
- JRA全券種 (単勝, 複勝, 枠連, 馬連, ワイド, 馬単, 3連複, 3連単) の仕様を集約
- 表記ゆれの完全正規化 ("三連複", "３連複", "trio" 等 -> "3連複")
- 組番フォーマット & 払戻データマッピング (ordered/unordered 判定)
- 払戻金自動照合 (lookup_payout)
- レガシー個別フィールド (tansho_bets, wide_amount, umaren_amount 等) の自動一括生成
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass(frozen=True)
class TicketTypeSpec:
    """単一券種の仕様定義"""
    canonical_name: str          # 標準名 (例: "単勝", "複勝", "馬連", "ワイド", "馬単", "3連複", "3連単")
    aliases: Tuple[str, ...]     # 表記ゆれエイリアス (小文字/全角/英名)
    payout_map_key: str          # race_res["payouts"] 内のキー ("tansho", "wide", "trio" 等)
    legacy_field_prefix: str     # レガシー個別フィールド接頭辞 ("tansho", "wide", "umaren" 等)
    num_horses: int              # 必要馬番数 (1, 2, 3)
    is_ordered: bool             # 着順固定フラグ (単勝/馬単/3連単は True, 複勝/馬連/ワイド/3連複は False)
    separator: str               # 表示用区切り文字 ("-" または "→")
    default_odds_factor: float   # 概算オッズフォールバック係数


# JRA公式全券種の定義テーブル
TICKET_SPECS: List[TicketTypeSpec] = [
    TicketTypeSpec(
        canonical_name="単勝",
        aliases=("単勝", "たんしょう", "tansho", "tan", "win"),
        payout_map_key="tansho",
        legacy_field_prefix="tansho",
        num_horses=1,
        is_ordered=True,
        separator="",
        default_odds_factor=1.0,
    ),
    TicketTypeSpec(
        canonical_name="複勝",
        aliases=("複勝", "ふくしょう", "fukusho", "fuku", "place"),
        payout_map_key="fukusho",
        legacy_field_prefix="fukusho",
        num_horses=1,
        is_ordered=False,
        separator="",
        default_odds_factor=0.35,
    ),
    TicketTypeSpec(
        canonical_name="枠連",
        aliases=("枠連", "わくれん", "wakuren", "bracket_quinella"),
        payout_map_key="wakuren",
        legacy_field_prefix="wakuren",
        num_horses=2,
        is_ordered=False,
        separator="-",
        default_odds_factor=0.7,
    ),
    TicketTypeSpec(
        canonical_name="馬連",
        aliases=("馬連", "うまれん", "umaren", "quinella"),
        payout_map_key="umaren",
        legacy_field_prefix="umaren",
        num_horses=2,
        is_ordered=False,
        separator="-",
        default_odds_factor=0.8,
    ),
    TicketTypeSpec(
        canonical_name="ワイド",
        aliases=("ワイド", "ﾜｲﾄﾞ", "wide", "quinella_place"),
        payout_map_key="wide",
        legacy_field_prefix="wide",
        num_horses=2,
        is_ordered=False,
        separator="-",
        default_odds_factor=0.4,
    ),
    TicketTypeSpec(
        canonical_name="馬単",
        aliases=("馬単", "うまたん", "umatan", "exacta"),
        payout_map_key="umatan",
        legacy_field_prefix="umatan",
        num_horses=2,
        is_ordered=True,
        separator="→",
        default_odds_factor=1.6,
    ),
    TicketTypeSpec(
        canonical_name="3連複",
        aliases=("3連複", "３連複", "三連複", "sanrenpuku", "trio"),
        payout_map_key="trio",
        legacy_field_prefix="trio",
        num_horses=3,
        is_ordered=False,
        separator="-",
        default_odds_factor=2.5,
    ),
    TicketTypeSpec(
        canonical_name="3連単",
        aliases=("3連単", "３連単", "三連単", "sanrentan", "tierce", "trifecta"),
        payout_map_key="tierce",
        legacy_field_prefix="sanrentan",
        num_horses=3,
        is_ordered=True,
        separator="→",
        default_odds_factor=12.0,
    ),
]


class TicketTypeRegistry:
    """全券種の仕様検索・正規化・払戻照合を行うレジストリ"""
    _spec_by_canonical: Dict[str, TicketTypeSpec] = {spec.canonical_name: spec for spec in TICKET_SPECS}
    _spec_by_alias: Dict[str, TicketTypeSpec] = {}

    for spec in TICKET_SPECS:
        for alias in spec.aliases:
            _spec_by_alias[alias.lower().strip()] = spec

    @classmethod
    def get_spec(cls, ticket_type: str) -> Optional[TicketTypeSpec]:
        """券種名またはエイリアスから TicketTypeSpec を取得"""
        if not ticket_type:
            return None
        cleaned = str(ticket_type).lower().strip()
        return cls._spec_by_alias.get(cleaned) or cls._spec_by_canonical.get(ticket_type)

    @classmethod
    def normalize_name(cls, ticket_type: str) -> str:
        """表記ゆれ（'三連複', 'ﾜｲﾄﾞ' 等）を標準名（'3連複', 'ワイド'）へ正規化"""
        spec = cls.get_spec(ticket_type)
        return spec.canonical_name if spec else ticket_type

    @classmethod
    def format_combination_key(cls, ticket_type: str, combination: List[int]) -> str:
        """
        券種に応じた標準組番文字列を生成
        順不同（馬連・ワイド・3連複等）は自動ソートして "3-7", "3-7-11"
        順序固定（馬単・3連単）は順序を保持して "7→3", "7→3→11"
        """
        spec = cls.get_spec(ticket_type)
        if not spec or not combination:
            return "-".join(str(u) for u in combination)

        if spec.is_ordered:
            return spec.separator.join(str(u) for u in combination)
        else:
            sorted_comb = sorted(combination)
            return spec.separator.join(str(u) for u in sorted_comb)

    @classmethod
    def lookup_payout(
        cls,
        ticket_type: str,
        combination: List[int],
        amount: int,
        payouts_data: Dict[str, Any]
    ) -> Tuple[int, int, str]:
        """
        JRA払戻データ (payouts_data) から該当買い目の払戻金額を照合・算出
        戻り値: (払戻金額[円], 100円あたり払戻金[円], 照合に成功した組番キー文字列)
        """
        spec = cls.get_spec(ticket_type)
        if not spec or not payouts_data or not combination:
            return 0, 0, ""

        p_map = payouts_data.get(spec.payout_map_key, {})
        if not isinstance(p_map, dict):
            return 0, 0, ""

        # 1. 単勝・複勝 (1頭)
        if spec.num_horses == 1:
            u = combination[0]
            unit_p = p_map.get(u, 0) or p_map.get(str(u), 0)
            if unit_p and int(unit_p) > 0:
                unit_val = int(unit_p)
                pay = int((amount / 100.0) * unit_val)
                return pay, unit_val, str(u)
            return 0, 0, str(u)

        # 2. 2頭以上の場合 (キー照合)
        candidate_keys: List[str] = []
        if spec.is_ordered:
            # 順序固定: "7-3" または "7→3"
            candidate_keys.append("-".join(str(u) for u in combination))
            candidate_keys.append("→".join(str(u) for u in combination))
        else:
            # 順不同: 昇順ソート "3-7"
            s_comb = sorted(combination)
            candidate_keys.append("-".join(str(u) for u in s_comb))
            candidate_keys.append("→".join(str(u) for u in s_comb))

        for ck in candidate_keys:
            unit_p = p_map.get(ck, 0)
            if unit_p and int(unit_p) > 0:
                unit_val = int(unit_p)
                pay = int((amount / 100.0) * unit_val)
                disp_key = cls.format_combination_key(ticket_type, combination)
                return pay, unit_val, disp_key

        disp_key = cls.format_combination_key(ticket_type, combination)
        return 0, 0, disp_key

    @classmethod
    def build_legacy_ticket_fields(
        cls,
        tickets: List[Union[Dict[str, Any], Any]]
    ) -> Dict[str, Any]:
        """
        共通チケット配列から、後方互換性を持つ全券種個別フィールドおよび合計金額を一括生成
        - tansho_bets, fukusho_bets, umatan_bets, umaren_bets, wide_bets, trio_bets, sanrentan_bets
        - tansho_amount, fukusho_amount, umatan_amount, umaren_amount, wide_amount, trio_amount, sanrentan_amount
        - total_amount
        - tickets (正規化済みチケット辞書リスト)
        """
        res: Dict[str, Any] = {
            "tansho_bets": [],
            "fukusho_bets": [],
            "umatan_bets": [],
            "umaren_bets": [],
            "wide_bets": [],
            "trio_bets": [],
            "sanrentan_bets": [],
            "tansho_amount": 0,
            "fukusho_amount": 0,
            "umatan_amount": 0,
            "umaren_amount": 0,
            "wide_amount": 0,
            "trio_amount": 0,
            "sanrentan_amount": 0,
            "hypothetical_umaren_bets": [],
            "trio_bet": None,
            "total_amount": 0,
            "tickets": [],
        }

        dict_tickets: List[Dict[str, Any]] = []
        for t in tickets:
            if hasattr(t, "to_dict"):
                dict_tickets.append(t.to_dict())
            elif isinstance(t, dict):
                dict_tickets.append(dict(t))

        total_amt = 0
        for t in dict_tickets:
            raw_type = t.get("ticket_type", "")
            spec = cls.get_spec(raw_type)
            norm_type = spec.canonical_name if spec else raw_type
            t["ticket_type"] = norm_type

            amt = int(t.get("amount", 0) or 0)
            total_amt += amt

            if spec:
                pfx = spec.legacy_field_prefix
                bets_key = f"{pfx}_bets"
                amt_key = f"{pfx}_amount"
                if bets_key in res:
                    res[bets_key].append(t)
                if amt_key in res:
                    res[amt_key] += amt

        res["total_amount"] = total_amt
        res["tickets"] = dict_tickets
        if res["trio_bets"]:
            res["trio_bet"] = res["trio_bets"][0]

        return res

    @classmethod
    def get_empty_legacy_ticket_fields(cls) -> Dict[str, Any]:
        """見送り・安全退避時用の全券種空フィールドを生成"""
        return cls.build_legacy_ticket_fields([])


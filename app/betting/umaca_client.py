# -*- coding: utf-8 -*-
"""
app/betting/umaca_client.py

UMACAスマート (www.ipat.jra.go.jp/sp/umaca/) 高速HTTPクライアント (BaseBettingClient 実装)
"""

import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from app.core.config import settings
from app.betting.base_client import BaseBettingClient

TICKET_TYPE_MAP = {
    "単勝": "1",
    "複勝": "2",
    "枠連": "3",
    "馬連": "4",
    "ワイド": "5",
    "ﾜｲﾄﾞ": "5",
    "馬単": "6",
    "3連複": "7",
    "３連複": "7",
    "三連複": "7",
    "3連単": "8",
    "３連単": "8",
    "三連単": "8",
}

JYO_HEX_TO_NAME = {
    "1": "札幌", "2": "函館", "3": "福島", "4": "新潟", "5": "東京",
    "6": "中山", "7": "中京", "8": "京都", "9": "阪神", "A": "小倉"
}
NAME_TO_JYO_HEX = {v: k for k, v in JYO_HEX_TO_NAME.items()}


class UMACAClient(BaseBettingClient):
    """UMACAスマート軽量HTTPクライアント"""

    BASE_URL = "https://www.ipat.jra.go.jp/sp/umaca"

    def __init__(
        self,
        card_no: Optional[str] = None,
        pin: Optional[str] = None,
        birthday: Optional[str] = None,
    ):
        self.card_no = (card_no or settings.UMACA_CARD_NUMBER).strip()
        self.pin = (pin or settings.UMACA_PIN).strip()
        self.birthday = (birthday or settings.UMACA_BIRTHDAY).strip()

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Referer": f"{self.BASE_URL}/",
        })
        self.form0_data: Dict[str, str] = {}
        self.mg_list: List[str] = []
        self.last_balance: int = 0

    @property
    def platform_name(self) -> str:
        return "umaca"

    def is_configured(self) -> bool:
        return bool(self.card_no and self.pin and self.birthday)

    @staticmethod
    def _decode_euc(resp: requests.Response) -> str:
        try:
            return resp.content.decode("euc-jp", errors="replace")
        except Exception:
            return resp.text

    @staticmethod
    def _extract_form_inputs(html: str, form_name: str = "FORM0") -> Dict[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        form = soup.find("form", {"name": form_name}) or soup.find("form")
        if not form:
            return {}
        data: Dict[str, str] = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            if name:
                data[name] = inp.get("value", "")
        return data

    @staticmethod
    def _extract_mg_array(html: str) -> List[str]:
        m = re.search(r"Mg\s*=\s*new\s+Array\(([^)]+)\)", html)
        if not m:
            return []
        return [s.strip().strip('"').strip("'") for s in m.group(1).split(",") if s.strip()]

    @staticmethod
    def _parse_balance(html: str, form_inputs: Dict[str, str]) -> int:
        uzj = form_inputs.get("uzj", "")
        if len(uzj) >= 35 and uzj[-10:].isdigit():
            return int(uzj[-10:])
        m = re.search(r"投票用残高[\s\S]{0,120}?([\d,]+)\s*</li>", html)
        if m:
            return int(m.group(1).replace(",", ""))
        m2 = re.search(r"(?:投票用残高|購入限度額)[^\d\n]*([\d,]+)\s*円", html)
        if m2:
            return int(m2.group(1).replace(",", ""))
        return 0

    def login(self) -> Dict[str, Any]:
        """UMACAスマートへログインし、投票用残高を取得"""
        if not self.is_configured():
            return {
                "success": False,
                "error": "UMACA認証情報が未設定です。環境変数 (UMACA_CARD_NUMBER, UMACA_PIN, UMACA_BIRTHDAY) を設定してください。"
            }
        try:
            r_top = self.session.get(f"{self.BASE_URL}/", timeout=10)
            top_html = self._decode_euc(r_top)
            if "受付時間外" in top_html or "投票受付時間外" in top_html:
                return {"success": False, "error": "ただいまUMACAスマート投票受付時間外です。"}

            top_inputs = self._extract_form_inputs(top_html)
            uh_val = top_inputs.get("uh", "04wn1G")

            uu_login = f"{self.card_no}{self.birthday}{self.pin}"
            login_payload = {
                "uh": uh_val,
                "g": "780",
                "uu": uu_login,
                "ulf": "0",
            }
            r_menu = self.session.post(f"{self.BASE_URL}/pw_732_i.cgi", data=login_payload, timeout=12)
            menu_html = self._decode_euc(r_menu)

            if "エラー" in menu_html or "間違っています" in menu_html or "ログイン画面から" in menu_html:
                return {"success": False, "error": "UMACAスマート認証エラー (カード番号・生年月日・PINをご確認ください)"}

            form0 = self._extract_form_inputs(menu_html, "FORM0")
            if not form0 or "u" not in form0:
                return {"success": False, "error": "UMACAスマートのセッション取得に失敗しました（受付時間外またはメンテナンス中）"}

            self.form0_data = form0
            self.mg_list = self._extract_mg_array(menu_html)
            balance = self._parse_balance(menu_html, form0)
            self.last_balance = balance

            return {
                "success": True,
                "balance": balance,
                "message": f"UMACAスマート接続成功！ (投票用残高: {balance:,}円)"
            }
        except Exception as e:
            return {"success": False, "error": f"UMACAスマート通信例外: {e}"}

    @staticmethod
    def _cbt(flag: str, num_val: int) -> int:
        if flag == "1":
            return 0x80 >> (num_val - 1)
        elif flag == "2":
            return 0x200 >> (num_val - 1)
        elif flag == "3":
            return 0x80000 >> (num_val - 1)
        elif flag == "4":
            return 0x800000000 >> (num_val - 1)
        elif flag == "5":
            return 0x000020000 >> (num_val - 1)
        raise ValueError(f"Invalid CBT flag: {flag}")

    @classmethod
    def encode_ticket_code(
        cls,
        idx: int,
        jyo_hex: str,
        race_no: int,
        day_hex: str,
        ticket_type: str,
        combination: List[int],
        amount: int,
    ) -> str:
        siki = TICKET_TYPE_MAP.get(ticket_type, "1")
        race_hex = f"{int(race_no):X}"
        idx_hex = f"{int(idx):02X}"
        amt_unit = max(1, int(amount) // 100)

        if siki in ("1", "2"):
            u1 = int(combination[0])
            val = 0x100000 + (0x100000 + cls._cbt("3", u1))
            mask14 = f"{f'{val:x}'[-5:]}000000000"
        elif siki in ("4", "5"):
            u1, u2 = sorted([int(combination[0]), int(combination[1])])
            val = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            mask14 = f"{f'{val:x}'[-9:]}00000"
        elif siki == "6":
            u1, u2 = int(combination[0]), int(combination[1])
            val = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            mask14 = f"{f'{val:x}'[-9:]}00000"
        elif siki == "7":
            u1, u2, u3 = sorted([int(combination[0]), int(combination[1]), int(combination[2])])
            val1 = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            val2 = 0x100000 + (0x100000 + cls._cbt("3", u3))
            mask14 = f"{val1:x}"[-9:] + f"{val2:x}"[-5:]
        elif siki == "8":
            u1, u2, u3 = int(combination[0]), int(combination[1]), int(combination[2])
            val1 = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            val2 = 0x100000 + (0x100000 + cls._cbt("3", u3))
            mask14 = f"{val1:x}"[-9:] + f"{val2:x}"[-5:]
        elif siki == "3":
            w1, w2 = sorted([int(combination[0]), int(combination[1])])
            val1 = 0x100 + (0x100 + cls._cbt("1", w1))
            val2 = 0x1000 + (0x1000 + cls._cbt("2", w2))
            mask14 = f"{val1:x}"[-2:] + "00" + f"{val2:x}"[-3:] + "0000000"
        else:
            raise ValueError(f"Unsupported ticket_type: {ticket_type}")

        return f"10{idx_hex}{jyo_hex.upper()}{race_hex}{day_hex.upper()}0{siki}{mask14}{amt_unit:04d}"

    def _resolve_venue_mg(self, venue_name: str) -> Optional[Tuple[str, str]]:
        target_jyo = NAME_TO_JYO_HEX.get(venue_name)
        if not target_jyo:
            return None
        for mg in self.mg_list:
            if len(mg) >= 3 and mg[1].upper() == target_jyo.upper() and mg.endswith("111"):
                return mg[1].upper(), mg[2].upper()
        for mg in self.mg_list:
            if len(mg) >= 3 and mg[1].upper() == target_jyo.upper():
                return mg[1].upper(), mg[2].upper()
        return None

    def execute_bets(
        self,
        venue_name: str,
        race_no: int,
        items: List[Dict[str, Any]],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        if not items:
            return {"status": "skipped", "reason": "買い目が空です"}

        total_amount = sum(int(item.get("amount", 0)) for item in items)
        print(f"[UMACAClient] 発注開始: {venue_name}{race_no}R {len(items)}件 計{total_amount:,}円 (dry_run={dry_run})")

        l_res = self.login()
        if not l_res.get("success"):
            return {"status": "failed", "step": "login", "error": l_res.get("error")}

        current_balance = l_res.get("balance", 0)
        if current_balance < total_amount:
            msg = f"UMACA投票用残高不足です (必要: {total_amount:,}円 / 現在残高: {current_balance:,}円)。"
            print(f"[UMACAClient] ⚠️ {msg}")
            return {"status": "aborted", "step": "balance_check", "reason": msg}

        try:
            r_740 = self.session.post(f"{self.BASE_URL}/pw_740_i.cgi", data=self.form0_data, timeout=12)
            html_740 = self._decode_euc(r_740)
            mg_740 = self._extract_mg_array(html_740)
            if mg_740:
                self.mg_list = mg_740

            venue_mg = self._resolve_venue_mg(venue_name)
            if not venue_mg:
                return {
                    "status": "failed",
                    "step": "select_venue",
                    "error": f"本日の発売一覧に競馬場「{venue_name}」が見つかりませんでした (Mg={self.mg_list})"
                }
            jyo_hex, day_hex = venue_mg

            form_s = self._extract_form_inputs(html_740, "s")
            if not form_s:
                return {"status": "failed", "step": "form_s", "error": "投票フォーム(s)の取得に失敗しました"}

            encoded_count = 0
            encoded_amount = 0
            for item in items:
                t_type = item.get("ticket_type", "単勝")
                combo = item.get("combination") or []
                amt = int(item.get("amount", 100))
                if not combo or amt < 100:
                    continue
                unit_amt = max(1, amt // 100) * 100
                code27 = self.encode_ticket_code(encoded_count, jyo_hex, race_no, day_hex, t_type, combo, unit_amt)
                key = f"{encoded_count + 1:03d}"
                form_s[key] = code27
                encoded_count += 1
                encoded_amount += unit_amt

            if encoded_count == 0 or encoded_count != len(items):
                return {"status": "failed", "step": "encode_tickets", "error": "有効な買い目が完全に生成できませんでした"}

            r_741 = self.session.post(f"{self.BASE_URL}/pw_741_i.cgi", data=form_s, timeout=12)
            html_741 = self._decode_euc(r_741)

            # 確認画面の Nb 配列を検証 (先頭文字 '1' = 正常発売中, '2' = 締切, '3'/'A' = 取消)
            nb_matches = re.findall(r'Nb\[\d+\]\s*=\s*"([^"]+)"', html_741)
            active_nbs = [v for v in nb_matches if v != "0"]
            if not active_nbs:
                return {"status": "failed", "step": "confirm_741", "error": "確認画面で買い目が受理されませんでした"}

            # 確定仕様: 締切・取消が1点でもあればセッション破棄（完全一致原則）
            for nb_val in active_nbs:
                status_flag = nb_val[0]
                if status_flag == "2":
                    return {"status": "failed", "step": "confirm_741", "error": f"{venue_name}{race_no}R は既に投票締切時刻を過ぎています"}
                elif status_flag in ("3", "A"):
                    return {"status": "failed", "step": "confirm_741", "error": f"{venue_name}{race_no}R の指定馬番は出走取消または中止です"}

            form_741 = self._extract_form_inputs(html_741, "FORM0")
            if not form_741:
                return {"status": "failed", "step": "confirm_form", "error": "確認画面フォーム(FORM0)が見つかりませんでした"}

            if dry_run:
                return {
                    "status": "dry_run_success",
                    "receipt_no": "DRY-RUN",
                    "total_amount": encoded_amount,
                    "items_count": encoded_count,
                    "balance": current_balance,
                    "validated_codes": active_nbs,
                }

            # 4. 最終投票確定 (pw_742_i.cgi)
            for i, nb_val in enumerate(active_nbs):
                form_741[f"{i + 1:03d}"] = nb_val
            form_741["l"] = str(encoded_amount)
            form_741["s"] = str(encoded_amount)

            r_742 = self.session.post(f"{self.BASE_URL}/pw_742_i.cgi", data=form_741, timeout=15)
            html_742 = self._decode_euc(r_742)

            if "エラー" in html_742 and "受け付けました" not in html_742:
                soup_err = BeautifulSoup(html_742, "html.parser")
                compact_err = " ".join(soup_err.get_text().split())[:200]
                return {"status": "failed", "step": "submit_742", "error": f"最終投票送信エラー: {compact_err}"}

            # 投票結果画面の厳格検証 (Fail-Fast: 完了文言または受付番号が確認できない場合は失敗)
            is_umaca_success = ("受け付けました" in html_742) or ("【投票結果】" in html_742) or ("投票完了" in html_742)
            m_rcpt = re.search(r"受付番号[\s\S]{0,80}?(\d{4,})", html_742)
            if not m_rcpt:
                m_rcpt = re.search(r"cReceiptNo\s*=\s*[\"'](\d+)[\"']", html_742)

            if not is_umaca_success or not m_rcpt:
                soup_742 = BeautifulSoup(html_742, "html.parser")
                compact_err = " ".join(soup_742.get_text().split())[:200]
                print(f"[UMACAClient] [ERROR] 投票不成立を検知: 受付完了画面または受付番号が確認できませんでした (画面内容: {compact_err})")
                return {
                    "status": "failed",
                    "step": "verify_result_742",
                    "error": f"投票結果画面が確認できませんでした（不成立）: {compact_err}"
                }

            receipt_no = m_rcpt.group(1)

            form_742 = self._extract_form_inputs(html_742, "FORM0")
            balance_after = self._parse_balance(html_742, form_742) or (current_balance - encoded_amount)
            self.last_balance = balance_after

            print(f"[UMACAClient] [SUCCESS] 投票結果画面を検証完了！ 受付番号: {receipt_no} (合計: {encoded_amount:,}円 / 投票後残高: {balance_after:,}円)")
            return {
                "status": "success",
                "platform": "umaca",
                "receipt_no": receipt_no,
                "total_amount": total_amount,
                "items_count": encoded_count,
                "balance": balance_after,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"status": "failed", "step": "submit", "error": str(e)}


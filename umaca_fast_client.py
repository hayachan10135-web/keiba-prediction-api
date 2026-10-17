# -*- coding: utf-8 -*-
"""
backend-render/umaca_fast_client.py

【JRA UMACAスマート 超軽量 HTTP API クライアント】
Playwright 等の重いブラウザバイナリを使わず、純粋な requests.Session で
UMACAスマート投票サイト (https://www.ipat.jra.go.jp/sp/umaca/) の CGI と直接通信します。

【特徴】
1. メモリ消費量 わずか数MB (Render 512MB RAM でも極めて安全・安定)
2. 実行速度 約 0.5 秒の高速レスポンス
3. 認証番号はすべて環境変数 (UMACA_CARD_NUMBER, UMACA_PIN, UMACA_BIRTHDAY) から読み込み
4. 安全規定: 自動入金は行わず、投票用残高の範囲内でのみ動作。
"""

import os
import re
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    "1": "札幌",
    "2": "函館",
    "3": "福島",
    "4": "新潟",
    "5": "東京",
    "6": "中山",
    "7": "中京",
    "8": "京都",
    "9": "阪神",
    "A": "小倉",
}

NAME_TO_JYO_HEX = {v: k for k, v in JYO_HEX_TO_NAME.items()}


def _load_fallback_env() -> Dict[str, str]:
    """ローカル実行時など環境変数が未設定の場合に .env ファイルから読み込む"""
    candidates = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", "auto_betting", ".env"),
    ]
    env_map: Dict[str, str] = {}
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        env_map[k.strip()] = v.strip()
                break
            except Exception:
                pass
    return env_map


class UMACAFastClient:
    """UMACAスマート (https://www.ipat.jra.go.jp/sp/umaca/) 軽量HTTPクライアント"""

    BASE_URL = "https://www.ipat.jra.go.jp/sp/umaca"

    def __init__(
        self,
        card_no: Optional[str] = None,
        pin: Optional[str] = None,
        birthday: Optional[str] = None,
    ):
        fallback = _load_fallback_env()
        self.card_no = (card_no or os.getenv("UMACA_CARD_NUMBER") or fallback.get("UMACA_CARD_NUMBER") or "").strip()
        self.pin = (pin or os.getenv("UMACA_PIN") or fallback.get("UMACA_PIN") or "").strip()
        self.birthday = (birthday or os.getenv("UMACA_BIRTHDAY") or fallback.get("UMACA_BIRTHDAY") or "").strip()

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
            ),
            "Referer": f"{self.BASE_URL}/",
        })
        self.form0_data: Dict[str, str] = {}
        self.mg_list: List[str] = []
        self.last_balance: int = 0

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
        # 1. hidden input "uzj" (35桁: 件数5桁 + 専用残高10桁 + 精算可能10桁 + 投票用残高10桁)
        uzj = form_inputs.get("uzj", "")
        if len(uzj) >= 35 and uzj[-10:].isdigit():
            return int(uzj[-10:])
        # 2. HTMLテキストから抽出
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
            # 1. トップページ取得
            r_top = self.session.get(f"{self.BASE_URL}/", timeout=10)
            top_html = self._decode_euc(r_top)
            if "受付時間外" in top_html or "投票受付時間外" in top_html:
                return {"success": False, "error": "ただいまUMACAスマート投票受付時間外です。"}

            top_inputs = self._extract_form_inputs(top_html)
            uh_val = top_inputs.get("uh", "04wn1G")

            # 2. pw_732_i.cgi へログインPOST
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
                return {"success": False, "error": "UMACAスマート認証エラー (UMACA番号・生年月日・暗証番号をご確認ください)"}

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
        """
        UMACAスマート (740_260206.js) の 27文字買い目コードを生成
        フォーマット: "10" + idx(2桁HEX) + jyo(1桁HEX) + race(1桁HEX) + day(1桁HEX) + "0" + siki(1桁) + mask(14桁HEX) + unit(4桁DEC)
        """
        siki = TICKET_TYPE_MAP.get(ticket_type, "1")
        race_hex = f"{int(race_no):X}"
        idx_hex = f"{int(idx):02X}"
        amt_unit = max(1, int(amount) // 100)

        if siki in ("1", "2"):
            # 単勝・複勝
            u1 = int(combination[0])
            val = 0x100000 + (0x100000 + cls._cbt("3", u1))
            h_str = f"{val:x}"[-5:]
            mask14 = f"{h_str}000000000"
        elif siki in ("4", "5"):
            # 馬連・ワイド (昇順ソート)
            u1, u2 = sorted([int(combination[0]), int(combination[1])])
            val = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            h_str = f"{val:x}"[-9:]
            mask14 = f"{h_str}00000"
        elif siki == "6":
            # 馬単 (順序維持: 1着 u1, 2着 u2)
            u1, u2 = int(combination[0]), int(combination[1])
            val = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            h_str = f"{val:x}"[-9:]
            mask14 = f"{h_str}00000"
        elif siki == "7":
            # 3連複 (昇順ソート)
            u1, u2, u3 = sorted([int(combination[0]), int(combination[1]), int(combination[2])])
            val1 = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            val2 = 0x100000 + (0x100000 + cls._cbt("3", u3))
            mask14 = f"{val1:x}"[-9:] + f"{val2:x}"[-5:]
        elif siki == "8":
            # 3連単 (順序維持: 1着 u1, 2着 u2, 3着 u3)
            u1, u2, u3 = int(combination[0]), int(combination[1]), int(combination[2])
            val1 = 0x1000000000 + (0x1000000000 + cls._cbt("4", u1)) + (0x1000000000 + cls._cbt("5", u2))
            val2 = 0x100000 + (0x100000 + cls._cbt("3", u3))
            mask14 = f"{val1:x}"[-9:] + f"{val2:x}"[-5:]
        elif siki == "3":
            # 枠連
            w1, w2 = sorted([int(combination[0]), int(combination[1])])
            val1 = 0x100 + (0x100 + cls._cbt("1", w1))
            val2 = 0x1000 + (0x1000 + cls._cbt("2", w2))
            mask14 = f"{val1:x}"[-2:] + "00" + f"{val2:x}"[-3:] + "0000000"
        else:
            raise ValueError(f"Unsupported ticket_type: {ticket_type}")

        return f"10{idx_hex}{jyo_hex.upper()}{race_hex}{day_hex.upper()}0{siki}{mask14}{amt_unit:04d}"

    def _resolve_venue_mg(self, venue_name: str) -> Optional[Tuple[str, str]]:
        """Mg配列から指定競馬場の (jyo_hex, day_hex) を取得"""
        target_jyo = NAME_TO_JYO_HEX.get(venue_name)
        if not target_jyo:
            return None

        # 当日通常発売（末尾 "111"）を優先探索
        for mg in self.mg_list:
            if len(mg) >= 3 and mg[1].upper() == target_jyo.upper() and mg.endswith("111"):
                return mg[1].upper(), mg[2].upper()

        # なければ一致する最初の開催を返す
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
        """
        UMACAスマートへ指定された買い目を一括発注
        items: [{"ticket_type": "単勝", "combination": [11], "amount": 1000}, ...]
        """
        if not items:
            return {"status": "skipped", "reason": "買い目が空です"}

        total_amount = sum(int(item.get("amount", 0)) for item in items)
        print(f"[UMACAFast] 発注開始: {venue_name}{race_no}R {len(items)}件 計{total_amount:,}円 (dry_run={dry_run})")

        # 1. ログイン＆残高確認
        l_res = self.login()
        if not l_res.get("success"):
            return {"status": "failed", "step": "login", "error": l_res.get("error")}

        current_balance = l_res.get("balance", 0)
        if current_balance < total_amount:
            msg = f"UMACA投票用残高不足です (必要: {total_amount:,}円 / 現在残高: {current_balance:,}円)。"
            print(f"[UMACAFast] ⚠️ {msg}")
            return {"status": "aborted", "step": "balance_check", "reason": msg}

        try:
            # 2. 通常投票画面 (pw_740_i.cgi) へ遷移して最新の投票セッションと Mg 配列を取得
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

            # 3. フォーム s のパラメータを構築して確認画面 (pw_741_i.cgi) へPOST
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

            if encoded_count == 0:
                return {"status": "failed", "step": "encode_tickets", "error": "有効な買い目がありません"}

            r_741 = self.session.post(f"{self.BASE_URL}/pw_741_i.cgi", data=form_s, timeout=12)
            html_741 = self._decode_euc(r_741)

            # 確認画面の Nb 配列を検証 (先頭文字 '1' = 正常発売中, '2' = 締切, '3' = 取消)
            nb_matches = re.findall(r'Nb\[\d+\]\s*=\s*"([^"]+)"', html_741)
            active_nbs = [v for v in nb_matches if v != "0"]
            if not active_nbs:
                return {"status": "failed", "step": "confirm_741", "error": "確認画面で買い目が受理されませんでした"}

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
                print(f"[UMACAFast] 🛡️ DRY-RUN 成功: 確認画面 (pw_741_i.cgi) にて {encoded_count}件 (計{encoded_amount:,}円) の受理を確認しました。")
                return {
                    "status": "dry_run_success",
                    "receipt_no": "DRY-RUN",
                    "total_amount": encoded_amount,
                    "items_count": encoded_count,
                    "balance": current_balance,
                    "validated_codes": active_nbs,
                }

            # 4. 最終投票確定 (pw_742_i.cgi) へPOST
            # SendNB() と ToSend() の処理を再現
            for i, nb_val in enumerate(active_nbs):
                form_741[f"{i + 1:03d}"] = nb_val
            form_741["l"] = str(encoded_amount)
            form_741["s"] = str(encoded_amount)

            r_742 = self.session.post(f"{self.BASE_URL}/pw_742_i.cgi", data=form_741, timeout=15)
            html_742 = self._decode_euc(r_742)

            if "エラー" in html_742 and "受け付けました" not in html_742:
                return {"status": "failed", "step": "submit_742", "error": "最終投票送信時にエラーが返されました"}

            # 受付番号の抽出
            m_rcpt = re.search(r"受付番号[\s\S]{0,80}?(\d{4})", html_742)
            if not m_rcpt:
                m_rcpt = re.search(r"cReceiptNo\s*=\s*[\"'](\d+)[\"']", html_742)
            receipt_no = m_rcpt.group(1) if m_rcpt else "受付完了"

            form_742 = self._extract_form_inputs(html_742, "FORM0")
            balance_after = self._parse_balance(html_742, form_742) or (current_balance - encoded_amount)
            self.last_balance = balance_after

            print(f"[UMACAFast] 🎉 投票完了！ 受付番号: {receipt_no} (合計: {encoded_amount:,}円 / 投票後残高: {balance_after:,}円)")
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

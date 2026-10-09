def fetch_shutuba_table(race_id: str):
    # 1. 出馬表基本情報の取得 (PC版)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    race_title_tag = soup.select_one(".RaceName, .RaceName_Text, h1")
    race_name = race_title_tag.get_text(strip=True) if race_title_tag else f"Race {race_id}"

    # 2. スマホ版単勝オッズページから確実にオッズ・人気を取得
    odds_map = {}
    try:
        sp_odds_url = f"https://race.sp.netkeiba.com/?pid=odds&race_id={race_id}&type=b1"
        sp_headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
        }
        sp_resp = requests.get(sp_odds_url, headers=sp_headers, timeout=10)
        try:
            sp_html = sp_resp.content.decode("euc-jp")
        except UnicodeDecodeError:
            sp_html = sp_resp.content.decode("utf-8", errors="replace")
            
        sp_soup = BeautifulSoup(sp_html, "html.parser")
        
        # スマホ版テーブルから馬番・オッズ・人気を抽出
        for row in sp_soup.select("tr"):
            tds = row.find_all("td")
            if len(tds) >= 2:
                row_text = row.get_text()
                # 行内の数字とオッズ（小数）を抽出
                m_num = re.search(r"^\s*(\d{1,2})\b", tds[0].get_text(strip=True))
                # 各tdから小数オッズを探索
                odds_val = None
                pop_val = None
                for td in tds:
                    txt = td.get_text(strip=True)
                    if re.match(r"^\d+\.\d+$", txt):
                        odds_val = float(txt)
                    elif re.match(r"^\d+$", txt) and int(txt) <= 20 and td != tds[0]:
                        pop_val = int(txt)

                if m_num and odds_val is not None:
                    u_num = int(m_num.group(1))
                    odds_map[u_num] = {"odds": odds_val, "popularity": pop_val}
                    
    except Exception as e:
        print(f"スマホ版オッズ取得エラー ({race_id}): {e}")

    horses = []
    rows = soup.select("tr.HorseList, table.Shutuba_Table tbody tr")

    for row in rows:
        umaban_tag = row.select_one(".Umaban, td[class*='Umaban']")
        if not umaban_tag:
            continue
        umaban_text = umaban_tag.get_text(strip=True)
        if not umaban_text.isdigit():
            continue
        umaban = int(umaban_text)

        waku_tag = row.select_one(".Waku, td[class*='Waku']")
        waku_match = re.search(r"\d+", waku_tag.get_text(strip=True)) if waku_tag else None
        wakuban = int(waku_match.group()) if waku_match else 1

        name_tag = row.select_one(".HorseName a, .Horse_Info a, .Horse02 a")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        jockey_tag = row.select_one(".Jockey a, .Jockey")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

        kinryo_tag = row.select_one(".Barei, .Weight, .Kinryo")
        kinryo = 55.0
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        odds_info = odds_map.get(umaban, {})
        odds = odds_info.get("odds")
        popularity = odds_info.get("popularity")

        horses.append({
            "umaban": umaban,
            "wakuban": wakuban,
            "horse_name": horse_name,
            "jockey": jockey,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity
        })

    return race_name, horses

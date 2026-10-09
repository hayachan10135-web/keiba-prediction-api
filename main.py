# PC版のChrome User-Agentに変更（スマホ版への強制リダイレクトを回避）
HEADERS_PC = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def fetch_shutuba_table(race_id: str):
    """出馬表ページから出走馬データをスクレイピング（PC版）"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # レース名
    race_title_tag = soup.select_one(".RaceName, .RaceName_Text, h1")
    race_name = race_title_tag.get_text(strip=True) if race_title_tag else f"Race {race_id}"

    horses = []
    # PC版出馬表テーブルの各行（HorseList または trタグ全般を走査）
    rows = soup.select("tr.HorseList, table.Shutuba_Table tbody tr")

    for row in rows:
        # 馬番セル（.Umaban または 2列目）
        umaban_tag = row.select_one(".Umaban, td[class*='Umaban']")
        if not umaban_tag:
            continue
        umaban_text = umaban_tag.get_text(strip=True)
        if not umaban_text.isdigit():
            continue
        umaban = int(umaban_text)

        # 枠番
        waku_tag = row.select_one(".Waku, td[class*='Waku']")
        waku_match = re.search(r"\d+", waku_tag.get_text(strip=True)) if waku_tag else None
        wakuban = int(waku_match.group()) if waku_match else 1

        # 馬名
        name_tag = row.select_one(".HorseName a, .Horse_Info a, .Horse02 a")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        # 騎手名
        jockey_tag = row.select_one(".Jockey a, .Jockey")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

        # 斤量 (例: "55.0" や "牡2 55.0")
        kinryo_tag = row.select_one(".Barei, .Weight, .Kinryo")
        kinryo = 55.0
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        # 単勝オッズ
        odds_tag = row.select_one(".Popular, td[class*='Popular'], .Odds")
        odds = 10.0
        if odds_tag:
            om = re.search(r"(\d+(?:\.\d+)?)", odds_tag.get_text(strip=True))
            if om:
                odds = float(om.group(1))

        # 人気順
        pop_tag = row.select_one(".Ninki, span.Ninki")
        pop_match = re.search(r"\d+", pop_tag.get_text(strip=True)) if pop_tag else None
        popularity = int(pop_match.group()) if pop_match else None

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

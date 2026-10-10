def build_prediction_and_recs(race_name: str, horses: List[dict]) -> dict:
    if not horses:
        return {
            "race_name": race_name,
            "horses": [],
            "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}
        }

    # 12特徴量の入力行列を構築
    feature_rows = []
    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = h["kinryo"]
        o = h["odds"] if h["odds"] is not None else 50.0
        p = h["popularity"] if h["popularity"] is not None else 10.0
        
        # 過去走特徴量（出馬表単体取得時のデフォルト補完値）
        career_races = h.get("career_races", 5.0)
        career_top3_rate = h.get("career_top3_rate", 0.25)
        prev1_order = h.get("prev1_order", 5.0)
        prev1_pop = h.get("prev1_pop", 5.0)
        prev1_odds = h.get("prev1_odds", 15.0)
        prev2_order = h.get("prev2_order", 5.0)
        days_since_prev = h.get("days_since_prev", 35.0)

        feature_rows.append([
            w, u, k, o, p,
            career_races, career_top3_rate,
            prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev
        ])

    if model:
        try:
            scores = model.predict(feature_rows)
            for idx, h in enumerate(horses):
                h["score"] = round(float(scores[idx]), 4)
        except Exception as e:
            print(f"推論エラー (フォールバック計算に移行): {e}")
            for idx, h in enumerate(horses):
                odds = h["odds"] if h["odds"] else 50.0
                h["score"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)
    else:
        for idx, h in enumerate(horses):
            odds = h["odds"] if h["odds"] else 50.0
            h["score"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)

    # 期待回収率スコア順にソートして印を付与
    sorted_horses = sorted(horses, key=lambda x: x["score"], reverse=True)
    marks = ["◎", "◯", "▲", "△", "☆"]
    for idx, h in enumerate(sorted_horses):
        h["mark"] = marks[idx] if idx < len(marks) else "-"

    honmei = sorted_horses[0]["umaban"] if len(sorted_horses) > 0 else None
    taiko = sorted_horses[1]["umaban"] if len(sorted_horses) > 1 else None
    kuro = sorted_horses[2]["umaban"] if len(sorted_horses) > 2 else None
    osae = [h["umaban"] for h in sorted_horses[3:5]]

    recs = {
        "tansho": [honmei] if honmei else [],
        "fukusho": [honmei, taiko] if taiko else ([honmei] if honmei else []),
        "umaren": [],
        "wide": [],
        "sanrenpuku": []
    }

    if honmei and taiko:
        opponents = [taiko] + ([kuro] if kuro else []) + osae
        recs["umaren"] = [f"{honmei}-{opp}" for opp in opponents]
        recs["wide"] = [f"{honmei}-{opp}" for opp in opponents[:3]]
        
        second_tier_list = [str(x) for x in ([taiko] + ([kuro] if kuro else []))]
        third_tier_list = [str(x) for x in opponents]
        second_tier_str = ",".join(second_tier_list)
        third_tier_str = ",".join(third_tier_list)
        recs["sanrenpuku"] = [f"{honmei} - {second_tier_str} - {third_tier_str}"]

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "horses": horses_sorted_by_num,
        "recommendations": recs
    }

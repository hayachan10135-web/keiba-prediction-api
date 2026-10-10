def build_prediction_and_recs(race_name: str, race_id: str, horses: List[dict]) -> dict:
    if not horses:
        return {
            "race_name": race_name,
            "horses": [],
            "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}
        }

    # モデルの期待する特徴量数を判定 (5次元 / 12次元 / 16次元)
    expected_num_features = 16
    if model:
        try:
            expected_num_features = model.num_feature()
        except Exception:
            expected_num_features = 16

    # 競馬場コード (race_id の 4〜6 桁目)
    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    feature_rows = []
    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = h["kinryo"]
        o = h["odds"] if h["odds"] is not None else 50.0
        p = h["popularity"] if h["popularity"] is not None else 10.0

        if expected_num_features == 16:
            surface_type = 0
            distance = 1600.0
            distance_diff = 0.0
            career_races = h.get("career_races", 5.0)
            career_top3_rate = h.get("career_top3_rate", 0.25)
            prev1_order = h.get("prev1_order", 5.0)
            prev1_pop = h.get("prev1_pop", 5.0)
            prev1_odds = h.get("prev1_odds", 15.0)
            prev2_order = h.get("prev2_order", 5.0)
            days_since_prev = h.get("days_since_prev", 35.0)

            feature_rows.append([
                w, u, k, o, p,
                venue_code, surface_type, distance, distance_diff,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev
            ])
        elif expected_num_features == 12:
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
        else:
            feature_rows.append([w, u, k, o, p])

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
            base_score = 1.0 / (1.0 + (odds ** 0.5))
            h["score"] = round(float(base_score), 4)

    # スコア降順にソート
    sorted_horses = sorted(horses, key=lambda x: x["score"], reverse=True)

    # --- 印の動的割り当てロジック ---
    # デフォルトは全頭 "-"
    for h in sorted_horses:
        h["mark"] = "-"

    # 基本の5頭 (1〜5位)
    base_marks = ["◎", "◯", "▲", "△", "×"]
    for idx in range(min(5, len(sorted_horses))):
        sorted_horses[idx]["mark"] = base_marks[idx]

    # 6位以降の「☆」割り当て判定（最大3頭まで）
    if len(sorted_horses) >= 6:
        sorted_horses[5]["mark"] = "☆"
        base_hoshi_score = sorted_horses[5]["score"]
        # 僅差の基準値 (スコア差が 0.015 以内)
        DIFF_THRESHOLD = 0.015

        # 7位馬の判定
        if len(sorted_horses) >= 7:
            if (base_hoshi_score - sorted_horses[6]["score"]) <= DIFF_THRESHOLD:
                sorted_horses[6]["mark"] = "☆"

                # 8位馬の判定（7位が☆かつ8位も僅差の場合のみ）
                if len(sorted_horses) >= 8:
                    if (base_hoshi_score - sorted_horses[7]["score"]) <= DIFF_THRESHOLD:
                        sorted_horses[7]["mark"] = "☆"

    # --- 買い目推奨の生成 ---
    marked_horses = [h for h in sorted_horses if h["mark"] != "-"]
    honmei = sorted_horses[0]["umaban"] if len(sorted_horses) > 0 else None
    taiko = sorted_horses[1]["umaban"] if len(sorted_horses) > 1 else None
    kuro = sorted_horses[2]["umaban"] if len(sorted_horses) > 2 else None

    # 相手候補（対抗以降、印がついた全頭）
    opponents = [h["umaban"] for h in marked_horses[1:]]

    recs = {
        "tansho": [honmei] if honmei else [],
        "fukusho": [honmei, taiko] if taiko else ([honmei] if honmei else []),
        "umaren": [],
        "wide": [],
        "sanrenpuku": []
    }

    if honmei and opponents:
        recs["umaren"] = [f"{honmei}-{opp}" for opp in opponents]
        recs["wide"] = [f"{honmei}-{opp}" for opp in opponents[:4]]  # ワイドは上位4頭へ
        
        # 3連複フォーメーション: 軸(◎) - 2列目(◯,▲) - 3列目(印全頭)
        second_tier_list = [str(x) for x in ([taiko] + ([kuro] if kuro else []))]
        third_tier_list = [str(x) for x in opponents]
        second_tier_str = ",".join(second_tier_list)
        third_tier_str = ",".join(third_tier_list)
        recs["sanrenpuku"] = [f"{honmei} - {second_tier_str} - {third_tier_str}"]

    # 出力は馬番順（1〜16番）に戻して返却
    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "horses": horses_sorted_by_num,
        "recommendations": recs
    }

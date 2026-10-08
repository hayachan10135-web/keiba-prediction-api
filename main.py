import io
import re
import requests
import pandas as pd
import lightgbm as lgb
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Keiba Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "horse_ranker_model.txt"
try:
    model = lgb.Booster(model_file=MODEL_PATH)
except Exception as e:
    model = None
    print(f"モデルのロードに失敗しました: {e}")

FEATURE_COLS = [
    "枠番", "馬番", "斤量", "人気", "sex_code", "年齢", 
    "体重", "体重増減", "distance", "track_type_code", "condition_code"
]

def fetch_race_table(race_id: str) -> pd.DataFrame:
    """出馬表または確定レース結果からテーブルを取得"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 1. まずは出馬表URLを試す
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.encoding = "EUC-JP"
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="RaceTable01")

    # 2. なければ確定結果URLを試す
    if not table:
        url = f"https://db.netkeiba.com/race/{race_id}/"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="race_table_01")

    if not table:
        return pd.DataFrame()

    df = pd.read_html(io.StringIO(str(table)))[0]
    # カラム名に含まれる改行や全角・半角スペースを完全に除去
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]

    # メタデータ抽出
    intro_text = soup.get_text()
    dist_match = re.search(r"(芝|ダ|障).*?(\d{3,4})m", intro_text)
    track_type = dist_match.group(1) if dist_match else "芝"
    distance = int(dist_match.group(2)) if dist_match else 1600

    cond_match = re.search(r"(?:芝|ダート|障)\s*:\s*(\w+)", intro_text)
    track_condition = cond_match.group(1) if cond_match else "良"

    df["race_id"] = str(race_id)
    df["distance"] = distance
    df["track_type"] = track_type
    df["track_condition"] = track_condition
    return df

def find_column(df: pd.DataFrame, candidates: list) -> str:
    """指定した候補名に部分一致するカラムを探す"""
    for cand in candidates:
        for col in df.columns:
            if cand in col:
                return col
    return None

def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """推論用の特徴量変換（カラム揺れを自動吸収）"""
    data = df.copy()

    # 枠番
    col_waku = find_column(data, ["枠番", "枠"])
    data["枠番"] = pd.to_numeric(data[col_waku], errors="coerce").fillna(0).astype(int) if col_waku else 0

    # 馬番
    col_uma = find_column(data, ["馬番", "馬"])
    data["馬番"] = pd.to_numeric(data[col_uma], errors="coerce").fillna(0).astype(int) if col_uma else 0

    # 馬名
    col_name = find_column(data, ["馬名"])
    data["馬名"] = data[col_name].astype(str) if col_name else "馬名未設定"

    # 斤量
    col_kinryo = find_column(data, ["斤量", "負担重量"])
    data["斤量"] = pd.to_numeric(data[col_kinryo], errors="coerce").fillna(55.0) if col_kinryo else 55.0

    # 人気
    col_ninki = find_column(data, ["人気"])
    data["人気"] = pd.to_numeric(data[col_ninki], errors="coerce").fillna(10.0) if col_ninki else 10.0

    # 性齢
    col_sex_age = find_column(data, ["性齢", "性/齢"])
    if col_sex_age:
        data["性別"] = data[col_sex_age].astype(str).str[0]
        data["年齢"] = pd.to_numeric(data[col_sex_age].astype(str).str[1:], errors="coerce").fillna(3)
    else:
        data["性別"] = "牡"
        data["年齢"] = 3

    # 馬体重
    col_weight = find_column(data, ["馬体重", "体重"])
    def parse_weight(val):
        match = re.search(r"(\d+)(?:\(([-+]?\d+)\))?", str(val))
        if match:
            w = float(match.group(1))
            diff = float(match.group(2)) if match.group(2) else 0.0
            return w, diff
        return 470.0, 0.0

    if col_weight:
        parsed = [parse_weight(w) for w in data[col_weight]]
        data["体重"] = [p[0] for p in parsed]
        data["体重増減"] = [p[1] for p in parsed]
    else:
        data["体重"] = 470.0
        data["体重増減"] = 0.0

    track_map = {"芝": 1, "ダ": 2, "障": 3}
    cond_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
    sex_map = {"牡": 1, "牝": 2, "セ": 3}

    data["track_type_code"] = data["track_type"].map(track_map).fillna(1).astype(int)
    data["condition_code"] = data["track_condition"].map(cond_map).fillna(1).astype(int)
    data["sex_code"] = data["性別"].map(sex_map).fillna(1).astype(int)
    data["distance"] = pd.to_numeric(data["distance"], errors="coerce").fillna(1600).astype(int)

    return data

@app.get("/")
def root():
    return {"status": "ok", "message": "Keiba Prediction API is running"}

@app.get("/predict/{race_id}")
def predict(race_id: str):
    if model is None:
        raise HTTPException(status_code=500, detail="モデルがロードされていません")

    raw_df = fetch_race_table(race_id)
    if raw_df.empty:
        raise HTTPException(status_code=404, detail="出馬表を取得できませんでした")

    df_inf = preprocess_for_inference(raw_df)
    X = df_inf[FEATURE_COLS]

    scores = model.predict(X)
    df_inf["prediction_score"] = scores
    df_sorted = df_inf.sort_values(by="prediction_score", ascending=False).reset_index(drop=True)

    marks = ["◎", "○", "▲", "△", "☆"]
    predictions = []
    for rank, row in df_sorted.iterrows():
        predictions.append({
            "predicted_rank": rank + 1,
            "mark": marks[rank] if rank < len(marks) else "−",
            "umaban": int(row["馬番"]),
            "wakuban": int(row["枠番"]),
            "horse_name": str(row["馬名"]),
            "score": round(float(row["prediction_score"]), 4)
        })

    return {
        "race_id": race_id,
        "count": len(predictions),
        "predictions": predictions
    }

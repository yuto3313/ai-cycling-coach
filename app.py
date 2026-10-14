import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import json
import os
import re
import base64
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

CONFIG_FILE = "config.json"

# ==========================================
# 0. 設定の自動保存・読み込みマネージャー
# ==========================================
DEFAULT_CONFIG = {
    "athlete_id": "",
    "intervals_key": "",
    "gemini_key": "",
    "ftp": 280,
    "weight": 60.0,
    "target_ftp": 300,
    "race_name": "Mt.富士ヒルクライム",
    "race_date": (datetime.now().date() + timedelta(days=60)).strftime("%Y-%m-%d"),
    "race_type": "ヒルクライム (60分前後・L4/SST持続力重視)",
    "race_goal": "ゴールド〜プラチナ水準のペーシング維持、後半のタレ防止",
    "weekday_hours": 1.5,
    "weekend_hours": 3.5,
}

def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception:
            pass
    try:
        for k in ["athlete_id", "intervals_key", "gemini_key"]:
            if not config.get(k) and k in st.secrets:
                config[k] = st.secrets[k]
    except Exception:
        pass
    return config

def save_config(new_config: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(new_config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ==========================================
# 1. 週間プラン用の構造化データスキーマ
# ==========================================
class DailyWorkoutItem(BaseModel):
    date: str = Field(description="日付 (YYYY-MM-DD)")
    day_label: str = Field(description="曜日ラベル (例: 月曜)")
    is_rest_day: bool = Field(description="完全休養日の場合はTrue、練習日はFalse")
    title: str = Field(description="ワークアウトのタイトル (休養日の場合は 'Rest Day')")
    duration_min: int = Field(description="予定時間(分)。休養日は0")
    target_tss: int = Field(description="目標TSS。休養日は0")
    rationale: str = Field(description="この日のメニューの狙いとアドバイス")
    workout_code: str = Field(
        description="Intervals.icuのワークアウト記法テキスト(Warmup / Main Set / Cooldown)。休養日は空文字"
    )

class WeeklyPlanResponse(BaseModel):
    weekly_summary: str = Field(description="今週の全体方針・パワーカーブ弱点克服のアプローチ解説")
    days: list[DailyWorkoutItem] = Field(description="7日分のワークアウト配列")


# ==========================================
# 2. Intervals.icu API クライアント
# ==========================================
class IntervalsClient:
    BASE_URL = "https://intervals.icu/api/v1"

    def __init__(self, athlete_id: str, api_key: str):
        self.athlete_id = athlete_id.strip()
        self.auth = ("API_KEY", api_key.strip())

    def get_wellness(self, days: int = 30) -> list:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/wellness"
        res = requests.get(url, auth=self.auth, params={"oldest": start_date, "newest": end_date}, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_activities(self, days: int = 30) -> list:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/activities"
        res = requests.get(url, auth=self.auth, params={"oldest": start_date, "newest": end_date}, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_activity_detail(self, activity_id: str) -> dict:
        url = f"{self.BASE_URL}/activity/{activity_id}"
        res = requests.get(url, auth=self.auth, params={"intervals": "true"}, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_power_curves(self, curves: str = "42d") -> dict:
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/power-curves.json"
        res = requests.get(url, auth=self.auth, params={"curves": curves, "type": "Ride"}, timeout=10)
        res.raise_for_status()
        return res.json()

    def post_workout(self, name: str, description: str, target_date: str) -> dict:
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/events"
        payload = {
            "category": "WORKOUT",
            "start_date_local": f"{target_date}T00:00:00",
            "name": name,
            "description": description,
            "type": "Ride"
        }
        res = requests.post(url, auth=self.auth, json=payload, timeout=10)
        res.raise_for_status()
        return res.json()


# ==========================================
# 3. データ解析 & パワーカーブ弱点診断ロジック
# ==========================================
def extract_pdc_metrics(pdc_raw: dict, weight: float, current_ftp: int) -> dict:
    target_durations = {"15s": 15, "1m": 60, "5m": 300, "20m": 1200, "60m": 3600}
    extracted = {k: 0 for k in target_durations}
    eftp = None

    try:
        curve_list = pdc_raw.get("list", []) if isinstance(pdc_raw, dict) else []
        if curve_list:
            curve = curve_list[0]
            secs = curve.get("secs", [])
            watts = curve.get("watts", []) or curve.get("values", [])
            sec_to_watt = dict(zip(secs, watts))

            for label, target_sec in target_durations.items():
                if target_sec in sec_to_watt:
                    extracted[label] = int(sec_to_watt[target_sec] or 0)
                elif secs:
                    closest_sec = min(secs, key=lambda x: abs(x - target_sec))
                    if abs(closest_sec - target_sec) <= 30:
                        extracted[label] = int(sec_to_watt.get(closest_sec, 0) or 0)

            models = curve.get("powerModels", [])
            if models and isinstance(models, list):
                eftp = models[0].get("ftp")
    except Exception:
        pass

    w_5m = extracted["5m"]
    w_20m = extracted["20m"]
    diagnosis = "パワーカーブデータ未取得（ライド履歴が増えると自動診断されます）"

    if w_5m > 0 and w_20m > 0:
        ratio_20_to_5 = round((w_20m / w_5m) * 100, 1)
        est_20m_ftp = round(w_20m * 0.95)
        if ratio_20_to_5 < 81.0:
            diagnosis = (
                f"【VO2max優位・閾値持久力課題型】(20分/5分維持率: {ratio_20_to_5}% / 基準82〜86%) "
                f"5分パワー({w_5m}W)に対して20分閾値({w_20m}W)のタレが大きいため、"
                f"SST〜L4領域の滞在時間(TiZ)を伸ばすメニューが最優先です。"
            )
        elif ratio_20_to_5 > 86.5:
            diagnosis = (
                f"【ディーゼル(持久)型・VO2max天井課題型】(20分/5分維持率: {ratio_20_to_5}%) "
                f"20分パワー({w_20m}W)の維持率が非常に高い一方、5分領域({w_5m}W)の天井が詰まっています。"
                f"L5(VO2max)インターバルで有酸素の天井を引き上げるとFTPが一段伸びます。"
            )
        else:
            diagnosis = (
                f"【オールラウンドバランス型】(20分/5分維持率: {ratio_20_to_5}% / 20分換算FTP: {est_20m_ftp}W) "
                f"5分({w_5m}W)と20分({w_20m}W)のバランスが良好です。目標レースの要求特性に合わせてL4とL5を周期化しましょう。"
            )

    return {
        "watts": extracted,
        "wkg": {k: round(v / weight, 2) if weight > 0 else 0.0 for k, v in extracted.items()},
        "eftp": eftp,
        "diagnosis": diagnosis
    }


def parse_intervals_data(wellness_raw: list, activities_raw: list, pdc_raw: dict, weight: float, ftp: int):
    latest_ctl, latest_atl, latest_tsb = 0.0, 0.0, 0.0
    latest_weight = None
    if wellness_raw:
        latest = wellness_raw[-1]
        latest_ctl = round(latest.get("ctl", 0.0) or 0.0, 1)
        latest_atl = round(latest.get("atl", 0.0) or 0.0, 1)
        latest_tsb = round(latest_ctl - latest_atl, 1)
        latest_weight = latest.get("weight")

    sorted_acts = sorted(
        activities_raw,
        key=lambda x: x.get("start_date_local", ""),
        reverse=True
    )

    act_list = []
    raw_rides = []
    eftp_from_act = None

    for act in sorted_acts:
        if not act.get("type") or "Ride" not in act.get("type", ""):
            continue
        if not eftp_from_act and act.get("icu_pm_ftp"):
            eftp_from_act = act.get("icu_pm_ftp")

        moving_time_min = round((act.get("moving_time") or 0) / 60)
        dist_km = round((act.get("distance") or 0) / 1000, 1)
        elev_m = round(act.get("total_elevation_gain") or 0)
        np_w = round(act.get("icu_weighted_avg_watts") or 0)
        avg_w = round(act.get("average_watts") or 0)
        vi = round(np_w / avg_w, 2) if avg_w > 0 else 1.0
        decoupling = round(act.get("decoupling") or 0.0, 1) if act.get("decoupling") is not None else None
        ef = round(act.get("icu_efficiency_factor") or 0.0, 2) if act.get("icu_efficiency_factor") else None

        act_list.append({
            "日付": act.get("start_date_local", "")[:10],
            "ワークアウト名": act.get("name", "Ride"),
            "時間(分)": moving_time_min,
            "距離(km)": dist_km,
            "獲得標高(m)": elev_m,
            "TSS": round(act.get("icu_training_load") or 0),
            "NP(W)": np_w,
            "平均(W)": avg_w,
            "IF": round((act.get("icu_intensity") or 0) / 100, 2) if act.get("icu_intensity") else 0.0,
            "心拍(bpm)": round(act.get("average_heartrate") or 0),
            "ケイデンス": round(act.get("average_cadence") or 0),
            "デカップリング(%)":

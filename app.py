import streamlit as st
import requests
import pandas as pd
import json
import os
import re
import math
import hashlib
import statistics
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import altair as alt
import time

CONFIG_FILE = "config.json"
CACHE_FILE = "intervals_cache.json"
MEMORY_FILE = "coach_memory.json"
USER_DATA_DIR = "user_data"

# ==========================================
# 0. 設定・キャッシュ・AI記憶の自動保存マネージャー（マルチユーザー＆厳格セキュリティ対応）
# ==========================================
DEFAULT_CONFIG = {
    "athlete_id": "",
    "athlete_name": "",
    "intervals_key": "",
    "gemini_key": "",
    "sarashina_key": "",
    "model": "gemini-3.8-flash",
    "ftp": 280,
    "weight": 60.0,
    "target_ftp": 300,
    "race_name": "Mt.富士ヒルクライム",
    "race_date": (datetime.now().date() + timedelta(days=60)).strftime("%Y-%m-%d"),
    "race_type": "ヒルクライム (60分前後・L4/SST持続力重視)",
    "race_goal": "ゴールド〜プラチナ水準のペーシング維持、後半のタレ防止",
    "weekday_hours": 1.5,
    "weekend_hours": 3.5,
    # 外的要因（事故・怪我・病気・制限事項）
    "ext_status": "なし（通常通り）",
    "ext_detail": "",
    "ext_intensity_cap": "制限なし（コンディションに応じる）",
    # 公開設定
    "public_default_guest": False,
}

DEFAULT_MEMORY = {
    "messages": [],               # チャット履歴 [{"role": "user"|"assistant", "content": "...", "timestamp": "..."}]
    "generated_workout": None,    # 最新の「今日のメニュー」テキスト
    "daily_workout_history": [],  # 過去のメニュー生成履歴
    "weekly_plan": None,          # 最新の週間プラン dict
    "weekly_plan_history": [],    # 過去の週間プラン履歴
    "ride_reviews": {},           # アクティビティID別の評価履歴
}


def compute_user_uid(intervals_key: str, athlete_id: str = "") -> str:
    """他ユーザーのAPIキーから固有の識別ハッシュ（16文字）を生成（本人のみアクセス可能）"""
    if not intervals_key or not intervals_key.strip():
        return ""
    raw = f"icu_user:{intervals_key.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_owner_token(owner_intervals_key: str) -> str:
    """オーナー専用の暗号化トークン（16文字）を生成"""
    if not owner_intervals_key or not owner_intervals_key.strip():
        return ""
    raw = f"owner_master_token:{owner_intervals_key.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_active_file_paths() -> tuple[str, str, str]:
    """現在のセッションモード（owner または guest）に応じた (config_path, cache_path, memory_path) を返す"""
    mode = st.session_state.get("user_mode", "owner")
    if mode == "owner":
        return CONFIG_FILE, CACHE_FILE, MEMORY_FILE
    uid = st.session_state.get("guest_uid", "")
    if not uid:
        return "", "", ""
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "", uid)[:32]
    return (
        os.path.join(USER_DATA_DIR, f"user_{safe_uid}_config.json"),
        os.path.join(USER_DATA_DIR, f"user_{safe_uid}_cache.json"),
        os.path.join(USER_DATA_DIR, f"user_{safe_uid}_memory.json"),
    )


def _normalize_model_name(config: dict) -> dict:
    if config.get("model") in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
        config["model"] = "gemini-3.8-flash"
    elif config.get("model") in ["gemini-2.5-pro", "gemini-1.5-pro"]:
        config["model"] = "gemini-3.1-pro"
    return config


def load_owner_config() -> dict:
    """オーナー（自分）専用の設定を config.json および st.secrets から読み込む"""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception:
            pass
    try:
        for k in ["athlete_id", "intervals_key", "gemini_key", "sarashina_key"]:
            if not config.get(k) and k in st.secrets:
                config[k] = st.secrets[k]
    except Exception:
        pass
    return _normalize_model_name(config)


def load_config() -> dict:
    """現在のログインユーザー（オーナー または ゲスト）の設定を読み込む"""
    mode = st.session_state.get("user_mode", "owner")
    if mode == "owner":
        return load_owner_config()

    # ゲスト（他の人）モードの場合：オーナーのキーや設定は一切読み込まない
    config = DEFAULT_CONFIG.copy()
    if "guest_config" in st.session_state and isinstance(st.session_state.guest_config, dict):
        config.update(st.session_state.guest_config)

    cfg_path, _, _ = get_active_file_paths()
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if not config.get(k) or k not in st.session_state.get("guest_config", {}):
                        config[k] = v
        except Exception:
            pass
    return _normalize_model_name(config)


def save_config(new_config: dict):
    """現在のログインユーザーの設定を保存する（ゲストの場合はオーナーのconfig.jsonを上書きしない）"""
    mode = st.session_state.get("user_mode", "owner")
    if mode != "owner":
        st.session_state.guest_config = new_config.copy()
        uid = compute_user_uid(new_config.get("intervals_key", ""), new_config.get("athlete_id", ""))
        if uid:
            st.session_state.guest_uid = uid
            try:
                st.query_params["mode"] = "guest"
                st.query_params["uid"] = uid
            except Exception:
                pass

    cfg_path, _, _ = get_active_file_paths()
    if not cfg_path:
        return
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(new_config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_memory() -> dict:
    mem = {
        "messages": [],
        "generated_workout": None,
        "daily_workout_history": [],
        "weekly_plan": None,
        "weekly_plan_history": [],
        "ride_reviews": {},
    }
    _, _, mem_path = get_active_file_paths()
    if mem_path and os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k in mem:
                    if k in saved:
                        mem[k] = saved[k]
        except Exception:
            pass
    return mem


def save_memory():
    """現在のst.session_state内の記憶データを現在のユーザー専用ファイルへ永続化"""
    _, _, mem_path = get_active_file_paths()
    if not mem_path:
        return
    try:
        mem_data = {
            "messages": st.session_state.get("messages", [])[-60:],
            "generated_workout": st.session_state.get("generated_workout"),
            "daily_workout_history": st.session_state.get("daily_workout_history", [])[-20:],
            "weekly_plan": st.session_state.get("weekly_plan"),
            "weekly_plan_history": st.session_state.get("weekly_plan_history", [])[-10:],
            "ride_reviews": st.session_state.get("ride_reviews", {}),
        }
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(mem_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def save_metrics_cache(metrics: dict):
    if not metrics:
        return
    _, cache_path, _ = get_active_file_paths()
    if not cache_path:
        return
    try:
        data = {
            "ctl": metrics.get("ctl"),
            "atl": metrics.get("atl"),
            "tsb": metrics.get("tsb"),
            "ramp_rate": metrics.get("ramp_rate"),
            "weight": metrics.get("weight"),
            "wellness_readiness": metrics.get("wellness_readiness", {}),
            "wellness_history": metrics.get("wellness_history", []),
            "pmc_history": metrics.get("pmc_history", []),
            "activities": metrics.get("activities", []),
            "pdc": metrics.get("pdc", {}),
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_metrics_cache():
    _, cache_path, _ = get_active_file_paths()
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            act_list = data.get("activities", [])
            data["activities_df"] = pd.DataFrame(act_list).drop(columns=["id"], errors="ignore") if act_list else pd.DataFrame()
            return data
        except Exception:
            pass
    return None


def reload_user_session_data():
    """ユーザー切り替え時に、現在のユーザー用のキャッシュと記憶をセッションに再ロードする"""
    mem = load_memory()
    st.session_state.metrics = load_metrics_cache()
    st.session_state.messages = mem.get("messages", [])
    st.session_state.generated_workout = mem.get("generated_workout")
    st.session_state.daily_workout_history = mem.get("daily_workout_history", [])
    st.session_state.weekly_plan = mem.get("weekly_plan")
    st.session_state.weekly_plan_history = mem.get("weekly_plan_history", [])
    st.session_state.ride_reviews = mem.get("ride_reviews", {})


# ==========================================
# 1. 週間プラン用の構造化データスキーマ (JSON出力用)
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
    weekly_summary: str = Field(description="今週の全体方針・HRV/睡眠や外的要因への配慮・パワーカーブ弱点克服のアプローチ解説")
    days: list[DailyWorkoutItem] = Field(description="7日分のワークアウト配列")


# ==========================================
# 2. Intervals.icu API クライアント
# ==========================================
class IntervalsClient:
    BASE_URL = "https://intervals.icu/api/v1"

    def __init__(self, athlete_id: str, api_key: str):
        cleaned_id = athlete_id.strip() if athlete_id else ""
        self.athlete_id = cleaned_id if cleaned_id else "0"
        self.auth = ("API_KEY", api_key.strip())

    def get_athlete_profile(self) -> dict:
        """Intervals.icuからアスリートID・氏名・体重・FTPを自動取得"""
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}"
        res = requests.get(url, auth=self.auth, timeout=10)
        res.raise_for_status()
        data = res.json()
        resolved_id = str(data.get("id") or self.athlete_id or "")
        name = str(data.get("name") or "")
        weight = data.get("icu_weight") or data.get("weight")
        ftp = None
        for s in data.get("sportSettings", []):
            types_list = s.get("types") or []
            if "Ride" in types_list or "VirtualRide" in types_list:
                ftp = s.get("ftp") or s.get("indoor_ftp")
                if ftp:
                    break
        if not ftp:
            ftp = data.get("ftp") or data.get("icu_ftp")
        return {
            "athlete_id": resolved_id,
            "name": name,
            "weight": float(weight) if weight else None,
            "ftp": int(ftp) if ftp else None,
        }

    def get_wellness(self, days: int = 40) -> list:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/wellness"
        res = requests.get(url, auth=self.auth, params={"oldest": start_date, "newest": end_date}, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_activities(self, days: int = 40) -> list:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/activities"
        res = requests.get(url, auth=self.auth, params={"oldest": start_date, "newest": end_date}, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_power_curves(self, curves: str = "1y") -> dict:
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

    def get_activity_segments(self, activity_id: str) -> list:
        """Stravaセグメント一覧と各セグメントのタイム・パワー・心拍を算出"""
        try:
            url_seg = f"{self.BASE_URL}/activity/{activity_id}/segments"
            res_seg = requests.get(url_seg, auth=self.auth, timeout=10)
            if res_seg.status_code != 200:
                return []
            segs = res_seg.json()
            if not segs or not isinstance(segs, list):
                return []

            url_streams = f"{self.BASE_URL}/activity/{activity_id}/streams.json"
            res_streams = requests.get(url_streams, auth=self.auth, params={"types": "time,watts,heartrate,distance"}, timeout=10)
            if res_streams.status_code != 200:
                return [
                    {
                        "セグメント名": s.get("name", "Unknown"),
                        "タイム": "--",
                        "距離(km)": "--",
                        "平均パワー(W)": "--",
                        "最大パワー(W)": "--",
                        "平均心拍(bpm)": "--"
                    }
                    for s in segs
                ]

            streams_list = res_streams.json()
            s_dict = {s["type"]: s["data"] for s in streams_list if "type" in s and "data" in s}
            times = s_dict.get("time", [])
            watts = s_dict.get("watts", [])
            hrs = s_dict.get("heartrate", [])
            dists = s_dict.get("distance", [])

            results = []
            for s in segs:
                si = s.get("start_index", 0)
                ei = s.get("end_index", 0)
                dur = times[ei] - times[si] if ei < len(times) and si < len(times) else 0
                dist_m = dists[ei] - dists[si] if ei < len(dists) and si < len(dists) else 0
                w_slice = [w for w in watts[si:ei+1] if w is not None] if watts else []
                avg_w = round(sum(w_slice) / len(w_slice)) if w_slice else 0
                max_w = max(w_slice) if w_slice else 0
                hr_slice = [h for h in hrs[si:ei+1] if h is not None] if hrs else []
                avg_hr = round(sum(hr_slice) / len(hr_slice)) if hr_slice else 0

                m, sec = divmod(int(dur), 60)
                time_str = f"{m}:{sec:02d}"
                results.append({
                    "セグメント名": s.get("name", "Segment"),
                    "タイム": time_str,
                    "距離(km)": round(dist_m / 1000, 2) if dist_m > 0 else "--",
                    "平均パワー(W)": avg_w,
                    "最大パワー(W)": max_w,
                    "平均心拍(bpm)": avg_hr,
                })
            return results
        except Exception:
            return []


# ==========================================
# 3. ワークアウト解析・科学指標計算・ZWO出力・チャート生成
# ==========================================
def parse_intervals_workout(workout_text: str, ftp: int):
    lines = workout_text.strip().splitlines()
    raw_sections = []
    current_repeat = 1
    current_block = []
    current_sec_name = "Main"

    for line in lines:
        s = line.strip()
        if not s:
            if current_block:
                raw_sections.append((current_repeat, current_block, current_sec_name))
                current_block = []
                current_repeat = 1
            continue

        if not s.startswith("-"):
            if current_block:
                raw_sections.append((current_repeat, current_block, current_sec_name))
                current_block = []
            m_rep = re.search(r"(\d+)\s*x", s, re.IGNORECASE)
            current_repeat = int(m_rep.group(1)) if m_rep else 1
            if "warm" in s.lower():
                current_sec_name = "Warmup"
            elif "cool" in s.lower():
                current_sec_name = "Cooldown"
            else:
                current_sec_name = s
        else:
            current_block.append(s)

    if current_block:
        raw_sections.append((current_repeat, current_block, current_sec_name))

    steps = []
    for rep, block, sec_name in raw_sections:
        for _ in range(rep):
            for step_line in block:
                dur_sec = 0
                m_h = re.search(r"(\d+(?:\.\d+)?)\s*h", step_line, re.IGNORECASE)
                m_m = re.search(r"(\d+(?:\.\d+)?)\s*m(?!s|p)", step_line, re.IGNORECASE)
                m_s = re.search(r"(\d+(?:\.\d+)?)\s*s", step_line, re.IGNORECASE)
                if m_h:
                    dur_sec += int(float(m_h.group(1)) * 3600)
                if m_m:
                    dur_sec += int(float(m_m.group(1)) * 60)
                if m_s:
                    dur_sec += int(float(m_s.group(1)))
                if dur_sec == 0:
                    dur_sec = 300

                m_p_range = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%", step_line)
                m_p = re.search(r"(\d+(?:\.\d+)?)\s*%", step_line)
                m_w = re.search(r"(\d+(?:\.\d+)?)\s*w\b", step_line, re.IGNORECASE)

                if m_p_range:
                    p_low = float(m_p_range.group(1)) / 100.0
                    p_high = float(m_p_range.group(2)) / 100.0
                    power_rel = (p_low + p_high) / 2.0
                elif m_p:
                    power_rel = float(m_p.group(1)) / 100.0
                elif m_w and ftp > 0:
                    power_rel = float(m_w.group(1)) / float(ftp)
                else:
                    power_rel = 0.65

                m_cad = re.search(r"(\d+)\s*rpm", step_line, re.IGNORECASE)
                cadence = int(m_cad.group(1)) if m_cad else None

                steps.append({
                    "section": sec_name,
                    "duration": dur_sec,
                    "power": round(power_rel, 3),
                    "cadence": cadence,
                    "raw": step_line
                })

    timeline = []
    current_time_sec = 0

    def get_zone(p):
        if p < 0.56:
            return ("Z1 回復", "#94a3b8")
        elif p < 0.76:
            return ("Z2 有酸素", "#38bdf8")
        elif p < 0.91:
            return ("Z3 テンポ/SST", "#22c55e")
        elif p < 1.06:
            return ("Z4 閾値(FTP)", "#eab308")
        elif p < 1.21:
            return ("Z5 VO2max", "#f97316")
        else:
            return ("Z6 無酸素", "#ef4444")

    for s in steps:
        dur = s["duration"]
        p = s["power"]
        watts = int(round(p * ftp))
        zone_label, color = get_zone(p)
        start_min = round(current_time_sec / 60, 2)
        end_min = round((current_time_sec + dur) / 60, 2)
        timeline.append({
            "start_min": start_min,
            "end_min": end_min,
            "duration_sec": dur,
            "duration_min": round(dur / 60, 1),
            "power_pct": int(round(p * 100)),
            "power_w": watts,
            "zone": zone_label,
            "color": color,
            "cadence": str(s["cadence"]) if s["cadence"] else "--",
            "section": s["section"]
        })
        current_time_sec += dur

    return steps, pd.DataFrame(timeline)


def calculate_workout_stats(steps: list, ftp: int) -> dict:
    """ワークアウト配列から総時間・平均W・NP・IF・推定TSS・消費kJ・推奨糖質補給量を科学的に算出"""
    if not steps or ftp <= 0:
        return {}
    total_sec = sum(s["duration"] for s in steps)
    if total_sec <= 0:
        return {}

    total_joules = sum(s["duration"] * (s["power"] * ftp) for s in steps)
    avg_watts = round(total_joules / total_sec)

    # NP (4乗平均の4乗根)
    weighted_p4 = sum(s["duration"] * ((s["power"] * ftp) ** 4) for s in steps) / total_sec
    np_watts = round(weighted_p4 ** 0.25)
    if_val = round(np_watts / ftp, 2)
    tss_val = round((total_sec * np_watts * if_val) / (ftp * 3600) * 100)
    kj_val = round(total_joules / 1000)

    # 運動時間とIFに応じた推奨糖質補給量 (g) と水分量 (ml)
    hours = total_sec / 3600.0
    if hours <= 0.75 and if_val < 0.85:
        carb_rate = 25
    elif if_val >= 0.90 or hours >= 2.0:
        carb_rate = 75
    elif if_val >= 0.80 or hours >= 1.25:
        carb_rate = 55
    else:
        carb_rate = 40
    recommended_carbs_g = round(carb_rate * hours)
    recommended_water_ml = round(600 * hours)

    # ゾーン別滞在時間 (分)
    zone_mins = {"Z1 回復": 0.0, "Z2 有酸素": 0.0, "Z3 テンポ/SST": 0.0, "Z4 閾値(FTP)": 0.0, "Z5 VO2max": 0.0, "Z6 無酸素": 0.0}
    for s in steps:
        p = s["power"]
        m = s["duration"] / 60.0
        if p < 0.56:
            zone_mins["Z1 回復"] += m
        elif p < 0.76:
            zone_mins["Z2 有酸素"] += m
        elif p < 0.91:
            zone_mins["Z3 テンポ/SST"] += m
        elif p < 1.06:
            zone_mins["Z4 閾値(FTP)"] += m
        elif p < 1.21:
            zone_mins["Z5 VO2max"] += m
        else:
            zone_mins["Z6 無酸素"] += m

    return {
        "total_min": round(total_sec / 60),
        "avg_watts": avg_watts,
        "np_watts": np_watts,
        "if": if_val,
        "tss": tss_val,
        "kj": kj_val,
        "carbs_g": recommended_carbs_g,
        "water_ml": recommended_water_ml,
        "zone_mins": {k: round(v, 1) for k, v in zone_mins.items() if v > 0},
    }


def generate_zwo_xml(workout_name: str, steps: list, ftp: int, description: str = "Generated by AI Cycling Coach") -> str:
    root = ET.Element("workout_file")
    ET.SubElement(root, "author").text = "AI Cycling Coach"
    ET.SubElement(root, "name").text = workout_name
    ET.SubElement(root, "description").text = description
    ET.SubElement(root, "sportType").text = "bike"

    workout_elem = ET.SubElement(root, "workout")

    for s in steps:
        dur = str(s["duration"])
        power_str = str(s["power"])
        sec = s.get("section", "Main")

        attribs = {"Duration": dur}
        if s.get("cadence"):
            attribs["Cadence"] = str(s["cadence"])

        if sec == "Warmup":
            attribs["PowerLow"] = str(round(max(0.4, s["power"] - 0.15), 2))
            attribs["PowerHigh"] = power_str
            ET.SubElement(workout_elem, "Warmup", attrib=attribs)
        elif sec == "Cooldown":
            attribs["PowerLow"] = str(round(max(0.4, s["power"] - 0.1), 2))
            attribs["PowerHigh"] = power_str
            ET.SubElement(workout_elem, "Cooldown", attrib=attribs)
        else:
            attribs["Power"] = power_str
            ET.SubElement(workout_elem, "SteadyState", attrib=attribs)

    xml_str = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(xml_str)
    return reparsed.toprettyxml(indent="  ")


def _apply_high_contrast_chart_theme(chart, height: int):
    """ダークモード時でもチャート背景を純白、軸ラベル・タイトルを濃紺(#0f172a)に固定して視認性を保証する"""
    return (
        chart.properties(height=height)
        .configure(background="#ffffff", padding={"left": 12, "right": 12, "top": 10, "bottom": 10})
        .configure_axis(
            labelColor="#0f172a",
            titleColor="#0f172a",
            labelFontSize=11,
            titleFontSize=12,
            labelFontWeight="bold",
            titleFontWeight="bold",
            gridColor="#e2e8f0",
            domainColor="#64748b",
            tickColor="#64748b",
        )
        .configure_legend(
            labelColor="#0f172a",
            titleColor="#0f172a",
        )
        .configure_view(stroke="#cbd5e1", fill="#ffffff")
    )


def build_workout_chart(df: pd.DataFrame, ftp: int):
    if df.empty:
        return None

    max_w = max(ftp * 1.25, df["power_w"].max() * 1.1)

    chart = alt.Chart(df).mark_rect(stroke="#1e293b", strokeWidth=0.6, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("start_min:Q", title="経過時間 (分)", axis=alt.Axis(grid=True)),
        x2="end_min:Q",
        y=alt.Y("power_w:Q", title="目標パワー (W)", scale=alt.Scale(domain=[0, max_w])),
        color=alt.Color("color:N", scale=None, title="ゾーン"),
        tooltip=[
            alt.Tooltip("section:N", title="セクション"),
            alt.Tooltip("start_min:Q", title="開始 (分)", format=".1f"),
            alt.Tooltip("end_min:Q", title="終了 (分)", format=".1f"),
            alt.Tooltip("power_pct:Q", title="強度 (%FTP)"),
            alt.Tooltip("power_w:Q", title="目標 (W)"),
            alt.Tooltip("cadence:N", title="ケイデンス (rpm)"),
            alt.Tooltip("zone:N", title="ゾーン")
        ]
    )

    ftp_df = pd.DataFrame([{"ftp": ftp, "label": f"FTP: {ftp}W"}])
    ftp_line = alt.Chart(ftp_df).mark_rule(
        color="#dc2626", strokeDash=[6, 4], strokeWidth=2.2
    ).encode(
        y="ftp:Q",
        tooltip=[alt.Tooltip("label:N", title="基準ライン")]
    )

    return _apply_high_contrast_chart_theme(chart + ftp_line, 235)


def build_tiz_bar_chart(zone_mins: dict):
    """ワークアウトのゾーン別滞在時間（TiZ）横棒チャート"""
    if not zone_mins:
        return None
    color_map = {
        "Z1 回復": "#64748b",
        "Z2 有酸素": "#0284c7",
        "Z3 テンポ/SST": "#16a34a",
        "Z4 閾値(FTP)": "#d97706",
        "Z5 VO2max": "#ea580c",
        "Z6 無酸素": "#dc2626",
    }
    order_list = list(color_map.keys())
    rows = [{"zone": k, "minutes": v, "color": color_map.get(k, "#0284c7")} for k, v in zone_mins.items()]
    df_z = pd.DataFrame(rows)
    ch = alt.Chart(df_z).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("minutes:Q", title="滞在時間 (分)"),
        y=alt.Y("zone:N", sort=order_list, title="パワーゾーン"),
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("zone:N", title="ゾーン"), alt.Tooltip("minutes:Q", title="時間 (分)")]
    )
    return _apply_high_contrast_chart_theme(ch, 165)


def build_pdc_chart(pdc_info: dict, weight: float, current_ftp: int, target_ftp: int):
    data = []
    dur_map = [
        ("15秒 (Sprint)", 15, "15s"),
        ("1分 (Anaerobic)", 60, "1m"),
        ("5分 (VO2max)", 300, "5m"),
        ("20分 (FTP)", 1200, "20m"),
        ("60分 (Endurance)", 3600, "60m")
    ]
    for label, sec, key in dur_map:
        w = pdc_info["watts"].get(key, 0)
        wkg = pdc_info["wkg"].get(key, 0.0)
        data.append({
            "duration_label": label,
            "order": sec,
            "watts": w,
            "wkg": wkg
        })
    df_pts = pd.DataFrame(data)

    cur_pwr = round(current_ftp / weight, 2) if weight > 0 else 0
    tar_pwr = round(target_ftp / weight, 2) if weight > 0 else 0

    base = alt.Chart(df_pts).encode(
        x=alt.X("duration_label:N", sort=[d[0] for d in dur_map], title="持続時間"),
        y=alt.Y("wkg:Q", title="パワーウェイトレシオ (W/kg)", scale=alt.Scale(domain=[0, max(tar_pwr + 1.5, df_pts["wkg"].max() + 0.5)])),
        tooltip=[
            alt.Tooltip("duration_label:N", title="持続時間"),
            alt.Tooltip("watts:Q", title="実績パワー (W)"),
            alt.Tooltip("wkg:Q", title="実績 PWR (W/kg)")
        ]
    )

    line = base.mark_line(color="#0284c7", strokeWidth=3, point=alt.OverlayMarkDef(size=75, color="#0284c7"))
    area = base.mark_area(opacity=0.18, color="#0284c7")

    ref_df = pd.DataFrame([
        {"level": f"目標FTP ({tar_pwr} W/kg)", "val": tar_pwr, "color": "#dc2626"},
        {"level": f"現在FTP ({cur_pwr} W/kg)", "val": cur_pwr, "color": "#16a34a"},
    ])
    rule = alt.Chart(ref_df).mark_rule(strokeDash=[5, 4], strokeWidth=2.2).encode(
        y="val:Q",
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("level:N", title="基準")]
    )

    return _apply_high_contrast_chart_theme(area + line + rule, 240)


def build_rider_profile_chart(pdc_info: dict):
    """Coggan基準に基づく5軸ライダー脚質スコアチャート (0-100点)"""
    scores = pdc_info.get("profile_scores", {})
    if not scores:
        return None
    rows = [
        {"axis": "15秒 スプリント (NM)", "score": scores.get("15s", 0), "color": "#7c3aed"},
        {"axis": "1分 無酸素耐性 (AC)", "score": scores.get("1m", 0), "color": "#dc2626"},
        {"axis": "5分 有酸素天井 (VO2max)", "score": scores.get("5m", 0), "color": "#ea580c"},
        {"axis": "20分 乳酸閾値 (LT/FTP)", "score": scores.get("20m", 0), "color": "#d97706"},
        {"axis": "60分 長時間持久力 (Stamina)", "score": scores.get("60m", 0), "color": "#16a34a"},
    ]
    df_r = pd.DataFrame(rows)
    ch = alt.Chart(df_r).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X("score:Q", title="競技レベルスコア (100 = 国内トップアマ/富士ヒルプラチナ級)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("axis:N", sort=[r["axis"] for r in rows], title="能力領域"),
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("axis:N", title="領域"), alt.Tooltip("score:Q", title="スコア (/100)")]
    )
    return _apply_high_contrast_chart_theme(ch, 200)


def build_wellness_chart(wellness_history: list):
    """直近のHRVと睡眠スコアの推移チャート"""
    if not wellness_history:
        return None
    df_w = pd.DataFrame(wellness_history)
    if df_w.empty or ("hrv" not in df_w.columns and "sleep_score" not in df_w.columns):
        return None

    base = alt.Chart(df_w).encode(
        x=alt.X("date:N", sort=None, title="日付")
    )

    hrv_line = base.mark_line(color="#7c3aed", strokeWidth=2.8, point=True).encode(
        y=alt.Y("hrv:Q", title="HRV (ms) / 睡眠スコア"),
        tooltip=[
            alt.Tooltip("date:N", title="日付"),
            alt.Tooltip("hrv:Q", title="HRV (ms)"),
            alt.Tooltip("sleep_score:Q", title="睡眠スコア"),
            alt.Tooltip("sleep_hours:Q", title="睡眠時間 (h)"),
            alt.Tooltip("rhr:Q", title="安静時心拍 (bpm)"),
        ]
    )

    sleep_line = base.mark_line(color="#0284c7", strokeDash=[4, 2], strokeWidth=2.2, point=True).encode(
        y=alt.Y("sleep_score:Q"),
        tooltip=[
            alt.Tooltip("date:N", title="日付"),
            alt.Tooltip("sleep_score:Q", title="睡眠スコア"),
            alt.Tooltip("sleep_hours:Q", title="睡眠時間 (h)"),
            alt.Tooltip("hrv:Q", title="HRV (ms)"),
        ]
    )

    return _apply_high_contrast_chart_theme(hrv_line + sleep_line, 205)


def build_pmc_chart(pmc_history: list):
    """過去40日間のPMC (CTL / ATL / TSB) 推移チャート（Altair v6完全互換）"""
    if not pmc_history:
        return None
    df_p = pd.DataFrame(pmc_history)
    if df_p.empty:
        return None

    df_p["tsb_color"] = df_p["tsb"].apply(
        lambda v: "#16a34a" if v >= 0 else ("#d97706" if v >= -20 else "#dc2626")
    )

    base = alt.Chart(df_p).encode(x=alt.X("date:N", sort=None, title="日付"))

    tsb_bars = base.mark_bar(opacity=0.45).encode(
        y=alt.Y("tsb:Q", title="CTL / ATL / TSB"),
        color=alt.Color("tsb_color:N", scale=None),
        tooltip=[
            alt.Tooltip("date:N", title="日付"),
            alt.Tooltip("ctl:Q", title="CTL (Fitness)"),
            alt.Tooltip("atl:Q", title="ATL (Fatigue)"),
            alt.Tooltip("tsb:Q", title="TSB (Form)"),
        ]
    )

    ctl_line = base.mark_line(color="#0284c7", strokeWidth=2.8).encode(y="ctl:Q")
    atl_line = base.mark_line(color="#e11d48", strokeWidth=2.2, strokeDash=[3, 2]).encode(y="atl:Q")

    return _apply_high_contrast_chart_theme(tsb_bars + ctl_line + atl_line, 205)


def build_ef_trend_chart(activities: list):
    """過去40日間の有酸素効率 (EF = NP / 平均心拍) ＆ TSS 推移チャート"""
    valid = [
        {
            "date": a["日付"][5:],
            "full_date": a["日付"],
            "name": a["ワークアウト名"],
            "ef": a.get("EF", 0.0),
            "tss": a.get("TSS", 0),
            "np": a.get("NP(W)", 0),
            "hr": a.get("心拍(bpm)", 0),
        }
        for a in reversed(activities)
        if a.get("EF", 0) > 0.5
    ]
    if len(valid) < 2:
        return None
    df_e = pd.DataFrame(valid)
    base = alt.Chart(df_e).encode(x=alt.X("date:N", sort=None, title="ライド日"))
    bars = base.mark_bar(opacity=0.3, color="#64748b").encode(
        y=alt.Y("tss:Q", title="TSS (棒)"),
        tooltip=[
            alt.Tooltip("full_date:N", title="日付"),
            alt.Tooltip("name:N", title="ライド名"),
            alt.Tooltip("ef:Q", title="EF (NP/HR)"),
            alt.Tooltip("tss:Q", title="TSS"),
            alt.Tooltip("np:Q", title="NP (W)"),
            alt.Tooltip("hr:Q", title="平均心拍 (bpm)"),
        ]
    )
    line = base.mark_line(color="#16a34a", strokeWidth=3, point=alt.OverlayMarkDef(size=70, color="#16a34a")).encode(
        y=alt.Y("ef:Q", title="Efficiency Factor (EF = NP/HR)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("full_date:N", title="日付"),
            alt.Tooltip("name:N", title="ライド名"),
            alt.Tooltip("ef:Q", title="EF (NP/HR)"),
            alt.Tooltip("np:Q", title="NP (W)"),
            alt.Tooltip("hr:Q", title="平均心拍 (bpm)"),
        ]
    )
    return _apply_high_contrast_chart_theme(alt.layer(bars, line).resolve_scale(y="independent"), 220)


# ==========================================
# 4. データ解析 & HRV/睡眠レディネス & CP/W' パワーカーブ診断
# ==========================================
def analyze_wellness_readiness(wellness_raw: list, latest_tsb: float) -> tuple[dict, list, list]:
    """
    Intervals.icu の wellness データから HRV・睡眠スコア・睡眠時間・安静時心拍・PMC推移を解析し、
    本日のレディネススコア(0-100)と推奨メニュー強度を算出する。
    """
    history = []
    pmc_history = []
    hrv_vals = []
    rhr_vals = []
    sleep_score_vals = []
    sleep_hour_vals = []

    for w in wellness_raw:
        d_str = w.get("id", "")
        short_d = d_str[5:] if len(d_str) >= 10 else d_str
        ctl_v = round(float(w.get("ctl") or 0.0), 1)
        atl_v = round(float(w.get("atl") or 0.0), 1)
        tsb_v = round(ctl_v - atl_v, 1)
        if d_str and (ctl_v > 0 or atl_v > 0):
            pmc_history.append({
                "date": short_d,
                "full_date": d_str,
                "ctl": ctl_v,
                "atl": atl_v,
                "tsb": tsb_v,
            })

        hrv = w.get("hrv") or w.get("hrvSDNN")
        rhr = w.get("restingHR")
        s_score = w.get("sleepScore")
        s_secs = w.get("sleepSecs")
        s_hours = round(s_secs / 3600, 2) if s_secs else None
        s_qual = w.get("sleepQuality")

        if hrv is not None:
            hrv_vals.append(float(hrv))
        if rhr is not None:
            rhr_vals.append(float(rhr))
        if s_score is not None:
            sleep_score_vals.append(float(s_score))
        if s_hours is not None:
            sleep_hour_vals.append(float(s_hours))

        if any(v is not None for v in [hrv, rhr, s_score, s_hours]):
            history.append({
                "date": short_d,
                "full_date": d_str,
                "hrv": round(float(hrv), 1) if hrv is not None else None,
                "rhr": int(rhr) if rhr is not None else None,
                "sleep_score": int(round(float(s_score))) if s_score is not None else None,
                "sleep_hours": round(s_hours, 1) if s_hours is not None else None,
                "sleep_quality": s_qual,
            })

    latest_rec = history[-1] if history else {}
    latest_hrv = latest_rec.get("hrv")
    latest_rhr = latest_rec.get("rhr")
    latest_sleep_score = latest_rec.get("sleep_score")
    latest_sleep_hours = latest_rec.get("sleep_hours")
    latest_date = latest_rec.get("full_date", "--")

    hrv_base = round(statistics.median(hrv_vals), 1) if len(hrv_vals) >= 3 else None
    rhr_base = round(statistics.median(rhr_vals), 1) if len(rhr_vals) >= 3 else None
    sleep_score_base = round(statistics.mean(sleep_score_vals), 1) if len(sleep_score_vals) >= 3 else None

    score = 75
    reasons = []

    if latest_hrv is not None and hrv_base and hrv_base > 0:
        ratio = latest_hrv / hrv_base
        diff_pct = round((ratio - 1.0) * 100, 1)
        if ratio >= 1.05:
            score += 10
            reasons.append(f"HRVが良好 ({latest_hrv}ms / 基準中央値{hrv_base}ms比 +{diff_pct}%)：副交感神経の回復が十分です")
        elif ratio >= 0.92:
            score += 4
            reasons.append(f"HRVは安定圏内 ({latest_hrv}ms / 基準中央値{hrv_base}ms)")
        elif ratio >= 0.82:
            score -= 14
            reasons.append(f"HRVが基準より低下 ({latest_hrv}ms / 基準中央値{hrv_base}ms比 {diff_pct}%)：自律神経系に疲労・ストレスが残っています")
        else:
            score -= 26
            reasons.append(f"HRVが大幅に低下 ({latest_hrv}ms / 基準中央値{hrv_base}ms比 {diff_pct}%)：強い未回復または体調変化のサインです")
    elif latest_hrv is not None:
        reasons.append(f"最新HRV: {latest_hrv}ms")

    if latest_sleep_score is not None:
        if latest_sleep_score >= 82:
            score += 10
            reasons.append(f"睡眠スコアが非常に良好 ({latest_sleep_score}点)")
        elif latest_sleep_score >= 72:
            score += 3
            reasons.append(f"睡眠スコアは標準的 ({latest_sleep_score}点)")
        elif latest_sleep_score >= 60:
            score -= 10
            reasons.append(f"睡眠スコアがやや低め ({latest_sleep_score}点)")
        else:
            score -= 20
            reasons.append(f"睡眠スコアが低い ({latest_sleep_score}点)：睡眠による回復が不十分です")

    if latest_sleep_hours is not None:
        if latest_sleep_hours < 5.5:
            score -= 12
            reasons.append(f"睡眠時間が短い ({latest_sleep_hours}時間)")
        elif latest_sleep_hours < 6.2:
            score -= 5
            reasons.append(f"睡眠時間がやや少なめ ({latest_sleep_hours}時間)")
        elif latest_sleep_hours >= 7.5:
            score += 5
            reasons.append(f"十分な睡眠時間を確保 ({latest_sleep_hours}時間)")

    if latest_rhr is not None and rhr_base:
        rhr_diff = round(latest_rhr - rhr_base, 1)
        if rhr_diff >= 5:
            score -= 15
            reasons.append(f"安静時心拍が基準より+{rhr_diff}bpm上昇 ({latest_rhr}bpm / 基準{rhr_base}bpm)：疲労・脱水・風邪初期に注意")
        elif rhr_diff >= 3:
            score -= 6
            reasons.append(f"安静時心拍が基準よりやや高め ({latest_rhr}bpm / 基準{rhr_base}bpm)")
        elif rhr_diff <= -2:
            score += 4
            reasons.append(f"安静時心拍が落ち着いています ({latest_rhr}bpm / 基準{rhr_base}bpm)")

    if latest_tsb < -25:
        score -= 10
        reasons.append(f"TSBが深いマイナス ({latest_tsb}) で筋疲労が蓄積")
    elif latest_tsb > 5:
        score += 4

    score = max(10, min(100, int(round(score))))

    if not reasons:
        reasons.append("HRV・睡眠データが未同期です")
        status_label = "データ未取得"
        recommendation = "主観的な体調に合わせてメニューを選択してください。"
    elif score >= 80:
        status_label = "🟢 絶好調 (High Readiness)"
        recommendation = "自律神経・睡眠ともに充実しています！VO2max(L5)インターバルや閾値(L4)など高強度メニューで限界を引き上げるのに最適な日です。"
    elif score >= 65:
        status_label = "🟡 良好・標準 (Normal Readiness)"
        recommendation = "コンディションは安定しています。予定通りのSST〜L4、または質の高いエンデュランス(L2/L3)メニューをこなせます。"
    elif score >= 48:
        status_label = "🟠 疲労気味・要注意 (Modulated)"
        recommendation = "HRV低下または睡眠・心拍にやや疲れが見られます。無理なL5高強度は避け、SST下限〜L2有酸素持久走や技術練習に調整することを推奨します。"
    else:
        status_label = "🔴 要回復 (Low Readiness)"
        recommendation = "自律神経の疲労や睡眠不足が顕著です。今日は高強度を避け、L1アクティブリカバリー（回復走30〜45分）または完全休養にして回復を最優先してください。"

    readiness_info = {
        "date": latest_date,
        "score": score,
        "status_label": status_label,
        "recommendation": recommendation,
        "reasons": reasons,
        "hrv": latest_hrv,
        "hrv_base": hrv_base,
        "rhr": latest_rhr,
        "rhr_base": rhr_base,
        "sleep_score": latest_sleep_score,
        "sleep_score_base": sleep_score_base,
        "sleep_hours": latest_sleep_hours,
    }
    return readiness_info, history[-14:], pmc_history


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
    wkg_dict = {k: round(v / weight, 2) if weight > 0 else 0.0 for k, v in extracted.items()}

    # 2パラメータ Critical Power (CP) & W' (無酸素仕事容量 kJ) の算出 (Monod & Scherrer 線形仕事モデル)
    cp_watts = 0
    w_prime_kj = 0.0
    if w_5m > 0 and w_20m > 0 and w_5m > w_20m:
        work_5m = w_5m * 300.0
        work_20m = w_20m * 1200.0
        cp_watts = int(round((work_20m - work_5m) / (1200.0 - 300.0)))
        w_prime_kj = round((work_5m - cp_watts * 300.0) / 1000.0, 1)

    # 5軸脚質スコア (100点満点: 国内トップアマ / Cat1 基準)
    benchmarks_wkg = {"15s": 16.5, "1m": 9.5, "5m": 6.4, "20m": 5.4, "60m": 5.0}
    profile_scores = {}
    for k, bench in benchmarks_wkg.items():
        val_wkg = wkg_dict.get(k, 0.0)
        profile_scores[k] = min(100, max(10, int(round((val_wkg / bench) * 100)))) if val_wkg > 0 else 0

    # ライダー脚質（Phenotype）の自動判定
    phenotype = "オールラウンダー"
    diagnosis = "パワーカーブデータ未取得（ライド履歴が増えると自動診断されます）"

    if w_5m > 0 and w_20m > 0:
        ratio_20_to_5 = round((w_20m / w_5m) * 100, 1)
        est_20m_ftp = round(w_20m * 0.95)
        if wkg_dict["15s"] >= 14.0 and wkg_dict["1m"] >= 8.0 and ratio_20_to_5 < 82.0:
            phenotype = "⚡ パンチャー / スプリンター型"
        elif ratio_20_to_5 >= 86.0 and wkg_dict["20m"] >= 4.5:
            phenotype = "🏔️ クライマー / TTディーゼル型"
        elif wkg_dict["5m"] >= 5.5 and ratio_20_to_5 < 82.5:
            phenotype = "🚀 VO2max アタッカー型"
        else:
            phenotype = "👑 オールラウンダー・クライマー型"

        if ratio_20_to_5 < 81.0:
            diagnosis = (
                f"【脚質: {phenotype}｜VO2max優位・閾値持久力課題】(20分/5分維持率: {ratio_20_to_5}% / 基準82〜86%) "
                f"5分パワー({w_5m}W)と無酸素容量(W': {w_prime_kj}kJ)が優秀な一方、20分領域({w_20m}W)への維持率に伸びしろがあります。"
                f"SST〜L4(閾値)の滞在時間(TiZ)を伸ばすメニューでCP({cp_watts}W)を引き上げましょう。"
            )
        elif ratio_20_to_5 > 86.5:
            diagnosis = (
                f"【脚質: {phenotype}｜高持久ディーゼル・VO2max天井課題】(20分/5分維持率: {ratio_20_to_5}%) "
                f"20分パワー({w_20m}W)とCP({cp_watts}W)の維持率が極めて高い一方、5分領域({w_5m}W)の天井が詰まっています。"
                f"L5(VO2max)インターバルで最大酸素摂取量の天井を押し上げるとFTPがさらに一段跳ね上がります。"
            )
        else:
            diagnosis = (
                f"【脚質: {phenotype}｜黄金バランス型】(20分/5分維持率: {ratio_20_to_5}% / 20分換算FTP: {est_20m_ftp}W / CP: {cp_watts}W / W': {w_prime_kj}kJ) "
                f"有酸素天井(5分:{w_5m}W)と閾値持久力(20分:{w_20m}W)のバランスが理想的です。目標レースの要求に合わせてL4巡航とL5強化を周期化しましょう。"
            )

    return {
        "watts": extracted,
        "wkg": wkg_dict,
        "eftp": eftp,
        "cp": cp_watts,
        "w_prime_kj": w_prime_kj,
        "phenotype": phenotype,
        "profile_scores": profile_scores,
        "diagnosis": diagnosis
    }


def parse_intervals_data(wellness_raw: list, activities_raw: list, pdc_raw: dict, weight: float, ftp: int):
    latest_ctl, latest_atl, latest_tsb, latest_ramp = 0.0, 0.0, 0.0, 0.0
    latest_weight = None
    if wellness_raw:
        latest = wellness_raw[-1]
        latest_ctl = round(latest.get("ctl", 0.0) or 0.0, 1)
        latest_atl = round(latest.get("atl", 0.0) or 0.0, 1)
        latest_tsb = round(latest_ctl - latest_atl, 1)
        latest_ramp = round(float(latest.get("rampRate") or 0.0), 1)
        latest_weight = latest.get("weight")

    readiness_info, wellness_hist, pmc_hist = analyze_wellness_readiness(wellness_raw, latest_tsb)

    act_list = []
    eftp_from_act = None
    for act in activities_raw:
        if not act.get("type") or "Ride" not in act.get("type", ""):
            continue
        if act.get("icu_pm_ftp"):
            eftp_from_act = act.get("icu_pm_ftp")
        moving_time_sec = act.get("moving_time") or 0
        moving_time_min = round(moving_time_sec / 60)
        np_w = round(act.get("icu_weighted_avg_watts") or act.get("weighted_average_watts") or 0)
        avg_w = round(act.get("icu_average_watts") or act.get("average_watts") or 0)
        avg_hr = round(act.get("average_heartrate") or 0)
        if_val = round((act.get("icu_intensity") or 0) / 100, 2) if act.get("icu_intensity") else (round(np_w / ftp, 2) if ftp > 0 and np_w > 0 else 0.0)
        vi_val = round(np_w / avg_w, 2) if avg_w > 0 and np_w > 0 else 1.0
        ef_val = round(np_w / avg_hr, 2) if avg_hr > 0 and np_w > 0 else 0.0
        kj_val = round((avg_w * moving_time_sec) / 1000) if avg_w > 0 else 0

        act_list.append({
            "id": act.get("id"),
            "日付": act.get("start_date_local", "")[:10],
            "ワークアウト名": act.get("name", "Ride"),
            "時間(分)": moving_time_min,
            "距離(km)": round((act.get("distance") or 0) / 1000, 1),
            "TSS": round(act.get("icu_training_load") or 0),
            "NP(W)": np_w,
            "平均(W)": avg_w,
            "IF": if_val,
            "心拍(bpm)": avg_hr,
            "EF": ef_val,
            "VI": vi_val,
            "仕事量(kJ)": kj_val,
        })

    pdc_info = extract_pdc_metrics(pdc_raw, weight, ftp)
    if not pdc_info["eftp"] and eftp_from_act:
        pdc_info["eftp"] = eftp_from_act

    return {
        "ctl": latest_ctl,
        "atl": latest_atl,
        "tsb": latest_tsb,
        "ramp_rate": latest_ramp,
        "weight": latest_weight,
        "wellness_readiness": readiness_info,
        "wellness_history": wellness_hist,
        "pmc_history": pmc_hist,
        "activities": act_list,
        "activities_df": pd.DataFrame(act_list).drop(columns=["id"], errors="ignore") if act_list else pd.DataFrame(),
        "pdc": pdc_info
    }


def evaluate_ride_science_score(ride: dict, readiness_score: int) -> tuple[int, dict, list]:
    """ライドの実行品質をスポーツ科学の観点から100点満点で自動採点し、糖質リカバリー量を算出する"""
    if_val = ride.get("IF", 0.0)
    vi_val = ride.get("VI", 1.0)
    ef_val = ride.get("EF", 0.0)
    tss_val = ride.get("TSS", 0)
    dur_min = ride.get("時間(分)", 0)
    kj_val = ride.get("仕事量(kJ)", 0)

    # 1. ペーシング安定度 (VI評価: 30点)
    if 1.0 <= vi_val <= 1.05:
        p_vi = 29
        vi_comment = f"極めて一定でロスのない理想的なペーシング (VI: {vi_val})"
    elif vi_val <= 1.12:
        p_vi = 25
        vi_comment = f"安定したペーシング (VI: {vi_val})"
    else:
        p_vi = 21
        vi_comment = f"インターバルまたは加減速の多い展開 (VI: {vi_val})"

    # 2. 目的強度と負荷バランス (35点)
    if 35 <= tss_val <= 180:
        p_load = 33
    elif tss_val > 180:
        p_load = 29
    else:
        p_load = 26

    # 3. 有酸素効率・心拍応答 (20点)
    if ef_val >= 1.85:
        p_ef = 19
        ef_comment = f"非常に高い有酸素効率 (EF: {ef_val} W/bpm)"
    elif ef_val >= 1.50:
        p_ef = 17
        ef_comment = f"良好な有酸素効率 (EF: {ef_val} W/bpm)"
    elif ef_val > 0:
        p_ef = 14
        ef_comment = f"有酸素効率 EF: {ef_val} W/bpm"
    else:
        p_ef = 15
        ef_comment = "心拍データなし"

    # 4. コンディション整合性 (15点)
    if readiness_score < 50 and if_val > 0.88:
        p_cond = 9
        cond_comment = "要注意：レディネス低下時の高強度実施のため入念な回復ケアが必要"
    else:
        p_cond = 14
        cond_comment = "現在のレディネスと整合した適切な負荷設定"

    total_score = min(100, p_vi + p_load + p_ef + p_cond)

    # グリコーゲン回復のための推奨糖質(g)・タンパク質(g)計算
    carb_ratio = 0.70 if if_val >= 0.85 else (0.55 if if_val >= 0.72 else 0.35)
    burned_carbs_g = round((kj_val * carb_ratio) / 4.0) if kj_val > 0 else round(dur_min * 1.1)
    immediate_carbs_g = max(30, min(120, round(burned_carbs_g * 0.6)))
    protein_g = 25 if tss_val >= 70 else 20

    nutrition = {
        "kj": kj_val,
        "burned_carbs_g": burned_carbs_g,
        "immediate_carbs_g": immediate_carbs_g,
        "protein_g": protein_g,
    }
    notes = [vi_comment, ef_comment, cond_comment]
    return total_score, nutrition, notes


def solve_hillclimb_physics(
    distance_km: float,
    elevation_m: float,
    rider_weight_kg: float,
    bike_weight_kg: float,
    watts: float,
    alt_m: float = 1500.0,
    crr: float = 0.0045,
    cda: float = 0.32,
    drivetrain_loss: float = 0.025,
    draft_saving_pct: float = 0.0,
) -> dict:
    """
    ニュートン力学（重力抵抗＋転がり抵抗＋標高別空気密度＆ドラフティング軽減率による空気抵抗＋駆動ロス）から
    登坂速度・予想タイム・標高による有酸素パワー減衰補正・必要ワット数を計算する。
    """
    dist_m = max(100.0, distance_km * 1000.0)
    grade = elevation_m / dist_m
    theta = math.atan(grade)
    total_mass = rider_weight_kg + bike_weight_kg
    g = 9.80665

    # 標高に応じた空気密度 rho (kg/m^3)
    rho = 1.225 * math.exp(-alt_m / 8500.0)
    # ドラフティングによる実効CdA (空気抵抗軽減率 0〜50%)
    draft_clamped = max(0.0, min(55.0, float(draft_saving_pct)))
    eff_cda = cda * (1.0 - draft_clamped / 100.0)
    eff_power = watts * (1.0 - drivetrain_loss)

    f_grav = total_mass * g * math.sin(theta)
    f_roll = total_mass * g * math.cos(theta) * crr

    # 二分探索で平衡速度 v (m/s) を解く
    low_v, high_v = 0.5, 25.0
    for _ in range(50):
        mid_v = (low_v + high_v) / 2.0
        f_aero = 0.5 * rho * eff_cda * (mid_v ** 2)
        req_p = (f_grav + f_roll + f_aero) * mid_v
        if req_p > eff_power:
            high_v = mid_v
        else:
            low_v = mid_v

    v_ms = (low_v + high_v) / 2.0
    time_sec = dist_m / v_ms
    vam = (elevation_m / (time_sec / 3600.0)) if time_sec > 0 else 0
    aero_watts = ((0.5 * rho * eff_cda * (v_ms ** 3)) / (1.0 - drivetrain_loss)) if (1.0 - drivetrain_loss) > 0 else 0.0

    # 高地（標高）による有酸素能力の低下率 (Bassett et al. モデル: 1000m超で1000mあたり約6.3%低下)
    altitude_penalty_pct = round(max(0.0, (alt_m - 800.0) / 1000.0 * 6.3), 1) if alt_m > 800 else 0.0

    return {
        "time_sec": time_sec,
        "time_min": round(time_sec / 60.0, 2),
        "speed_kmh": round(v_ms * 3.6, 1),
        "v_ms": v_ms,
        "vam": int(round(vam)),
        "grade_pct": round(grade * 100, 1),
        "altitude_penalty_pct": altitude_penalty_pct,
        "eff_cda": round(eff_cda, 3),
        "aero_watts": round(aero_watts, 1),
        "rho": rho,
        "f_grav_roll": f_grav + f_roll,
    }


def determine_training_phase(days_left: int) -> str:
    if days_left <= 0:
        return "レース当日・オフシーズン"
    elif days_left <= 10:
        return "テーパリング期（疲労抜き・キレ出し重視）"
    elif days_left <= 28:
        return "ピーク・特異期（レース強度直結重視）"
    elif days_left <= 84:
        return "ビルド期（FTP向上・VO2max強化）"
    else:
        return "ベース期（有酸素基盤・SST/L2ボリューム）"


def build_memory_context_str() -> str:
    """保存されている過去のメニュー、ライド評価、チャット要約をプロンプト文字列として構築"""
    parts = []

    reviews = st.session_state.get("ride_reviews", {})
    if reviews:
        rev_items = list(reviews.values())[-3:]
        rev_lines = []
        for r in rev_items:
            short_rev = (r.get("review", "")[:250] + "...") if len(r.get("review", "")) > 250 else r.get("review", "")
            rev_lines.append(f"- [{r.get('date')}] {r.get('name')} (体感メモ: {r.get('memo') or 'なし'}): {short_rev}")
        parts.append("【過去のライド振り返り・評価記憶（直近3件）】\n" + "\n".join(rev_lines))

    w_hist = st.session_state.get("daily_workout_history", [])
    if w_hist:
        w_lines = [f"- [{w.get('date')}] {w.get('title')} (条件: {w.get('condition', '')})" for w in w_hist[-3:]]
        parts.append("【直近提案した日次ワークアウト履歴】\n" + "\n".join(w_lines))

    wk_plan = st.session_state.get("weekly_plan")
    if wk_plan and isinstance(wk_plan, dict):
        summary = wk_plan.get("weekly_summary", "")[:200]
        parts.append(f"【現在進行中の週間プラン方針】\n{summary}")

    msgs = st.session_state.get("messages", [])
    if msgs:
        m_lines = []
        for m in msgs[-4:]:
            role_lbl = "ユーザー" if m["role"] == "user" else "AIコーチ"
            content_short = (m["content"][:150] + "...") if len(m["content"]) > 150 else m["content"]
            m_lines.append(f"- {role_lbl}: {content_short}")
        parts.append("【直近のチャット相談の文脈】\n" + "\n".join(m_lines))

    return "\n\n".join(parts) if parts else "（過去の履歴データはまだありません）"


def build_system_prompt(profile: dict, metrics: dict) -> str:
    days_left = (profile["race_date"] - datetime.now().date()).days
    phase = determine_training_phase(days_left)
    pwr = round(profile["ftp"] / profile["weight"], 2) if profile["weight"] > 0 else 0

    act_summary = "データなし"
    pdc_summary = "データなし"
    wellness_summary = "HRV・睡眠データ未取得"
    ctl, atl, tsb, ramp_rate = "未取得", "未取得", "未取得", "未取得"

    if metrics:
        ctl, atl, tsb = metrics["ctl"], metrics["atl"], metrics["tsb"]
        ramp_rate = metrics.get("ramp_rate", 0.0)
        if not metrics["activities_df"].empty:
            act_summary = metrics["activities_df"].head(15).to_markdown(index=False)
        if "pdc" in metrics:
            p = metrics["pdc"]
            w = p["watts"]
            wkg = p["wkg"]
            pdc_summary = (
                f"- 過去1年ベストパワー: 15秒={w['15s']}W({wkg['15s']}W/kg), "
                f"1分={w['1m']}W({wkg['1m']}W/kg), 5分={w['5m']}W({wkg['5m']}W/kg), "
                f"20分={w['20m']}W({wkg['20m']}W/kg), 60分={w['60m']}W({wkg['60m']}W/kg)\n"
                f"- 推定eFTP: {p['eftp'] or '未算出'} W / 臨界パワー(CP): {p.get('cp') or '--'} W / 無酸素仕事容量(W'): {p.get('w_prime_kj') or '--'} kJ\n"
                f"- 脚質分類・弱点プロファイリング診断: {p['diagnosis']}"
            )
        if "wellness_readiness" in metrics and metrics["wellness_readiness"]:
            wr = metrics["wellness_readiness"]
            reasons_str = " / ".join(wr.get("reasons", []))
            wellness_summary = (
                f"- 総合レディネススコア: {wr.get('score')}/100点 ({wr.get('status_label')})\n"
                f"- 最新HRV: {wr.get('hrv') or '--'} ms (過去40日中央値: {wr.get('hrv_base') or '--'} ms)\n"
                f"- 最新睡眠スコア: {wr.get('sleep_score') or '--'} 点 (平均: {wr.get('sleep_score_base') or '--'} 点) / 睡眠時間: {wr.get('sleep_hours') or '--'} 時間\n"
                f"- 最新安静時心拍(RHR): {wr.get('rhr') or '--'} bpm (過去40日中央値: {wr.get('rhr_base') or '--'} bpm)\n"
                f"- コンディション判定理由: {reasons_str}\n"
                f"- AI強度ガイドライン: {wr.get('recommendation')}"
            )

    ext_status = profile.get("ext_status", "なし（通常通り）")
    ext_detail = profile.get("ext_detail", "").strip()
    ext_cap = profile.get("ext_intensity_cap", "制限なし（コンディションに応じる）")

    if ext_status != "なし（通常通り）" or ext_detail or ext_cap != "制限なし（コンディションに応じる）":
        external_factor_prompt = f"""
【🚨 最優先の安全制約：外的要因・事故・病気・身体制限事項 🚨】
- 現在の状態区分: **{ext_status}**
- 強度上限リミッター: **{ext_cap}**
- 詳細メモ・症状・制限: **{ext_detail or '特記事項なし'}**
※重要指示: 上記の外的要因（怪我・落車・病気・痛み・多忙等）が設定されている場合、通常のパワーカーブ強化よりも**「身体の安全・段階的復帰（Return to Play）・症状悪化の防止」を絶対最優先**してください。指定された強度上限リミッターや制限（例: ダンシング禁止、心拍制限、高トルク禁止など）を厳守したメニュー・アドバイスを提供してください。
"""
    else:
        external_factor_prompt = "【外的要因・怪我・病気の制限】\n- 特になし（通常通りトレーニング可能）"

    memory_context = build_memory_context_str()

    return f"""あなたはワールドツアーチーム帯同レベルのスポーツ科学知見（運動生理学・CP/W'モデル・HRV自律神経解析・スポーツ栄養学・バイオメカニクス）と実戦経験を持つ専属AIサイクリングコーチです。
以下のアスリートデータ、HRVと睡眠スコアに基づくレディネス診断、外的要因（怪我・病気等）、および過去のコーチング記憶を踏まえ、一貫性のある具体的かつ論理的なアドバイスや練習メニューを提供してください。

{external_factor_prompt}

【🫀 本日のHRV・睡眠スコア・自律神経レディネス診断】
{wellness_summary}

【Intervals.icu 最新トレーニング負荷 (PMC)】
- CTL (Fitness / 長期負荷): {ctl}
- ATL (Fatigue / 短期疲労): {atl}
- TSB (Form / バランス): {tsb}
- Ramp Rate (週あたりCTL変化率): {ramp_rate}

【アスリート基本プロファイル】
- 現在FTP: {profile['ftp']} W / 体重: {profile['weight']} kg (現在PWR: {pwr} W/kg)
- 目標FTP: {profile['target_ftp']} W (目標PWR: {round(profile['target_ftp']/profile['weight'], 2)} W/kg)
- 平日練習可能時間: {profile['weekday_hours']} 時間 / 休日練習可能時間: {profile['weekend_hours']} 時間

【目標レース情報】
- レース名: {profile['race_name']}
- 開催日: {profile['race_date']} (残り {days_left} 日 / 現在のフェーズ: {phase})
- レースカテゴリー・特性: {profile['race_type']}
- 目標・メモ: {profile['race_goal']}

【パワーデュレーションカーブ (PDC / 過去1年) & CP/W' 弱点診断】
{pdc_summary}

【直近40日間のライド履歴（EF・VI・仕事量含む）】
{act_summary}

【🧠 AIコーチの長期記憶（過去のやり取り・メニュー・ライド評価履歴）】
{memory_context}

【Intervals.icu ワークアウト記法の書式ルール】
Warmup
- 10m 50% 90rpm
Main Set 3x
- 8m 98%
- 4m 55%
Cooldown
- 10m 45%
"""


SARASHINA_API_URL = "https://api.sarashina.ai/v1/chat/completions"


def is_sarashina_model(model_name: str) -> bool:
    """モデル名が Sarashina AI 系かどうかを判定"""
    return "sarashina" in (model_name or "").lower()


def _contents_to_openai_messages(contents, sys_prompt: str) -> list[dict]:
    """文字列または Gemini Content 配列を OpenAI / Sarashina 互換の messages 配列へ変換"""
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, dict):
                role = "assistant" if item.get("role") in ["assistant", "model"] else "user"
                messages.append({"role": role, "content": str(item.get("content", ""))})
            else:
                # google.genai.types.Content オブジェクトの場合
                role_raw = getattr(item, "role", "user")
                role = "assistant" if role_raw in ["assistant", "model"] else "user"
                parts = getattr(item, "parts", []) or []
                text_pieces = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                messages.append({"role": role, "content": "\n".join(text_pieces)})
    else:
        messages.append({"role": "user", "content": str(contents)})
    return messages


def call_sarashina_chat(sarashina_key: str, model_name: str, contents, sys_prompt: str) -> str:
    """Sarashina API (OpenAI互換 Chat Completions) を呼び出して応答テキストを返す"""
    if not sarashina_key or not sarashina_key.strip():
        raise ValueError("Sarashina API Key が未設定です。サイドバーで Sarashina API Key を入力・保存してください。")

    headers = {
        "Authorization": f"Bearer {sarashina_key.strip()}",
        "Content-Type": "application/json",
    }
    messages = _contents_to_openai_messages(contents, sys_prompt)
    # Sarashina API は未知フィールドに対して厳格なため model と messages のみを送信
    payload = {
        "model": model_name.strip(),
        "messages": messages,
    }
    res = requests.post(SARASHINA_API_URL, headers=headers, json=payload, timeout=90)
    if res.status_code != 200:
        raise RuntimeError(f"Sarashina API Error ({res.status_code}): {res.text}")
    data = res.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Sarashina API から空の応答が返されました: {data}")
    return choices[0].get("message", {}).get("content", "")


def _extract_json_str(raw_text: str) -> str:
    """Markdownの ```json ... ``` ブロックや前後の説明文から純粋なJSON文字列を抽出"""
    s = raw_text.strip()
    m_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m_fence:
        return m_fence.group(1).strip()
    m_brace = re.search(r"(\{.*\})", s, re.DOTALL)
    if m_brace:
        return m_brace.group(1).strip()
    return s


def get_candidate_models(primary_model: str, gemini_key: str = "", sarashina_key: str = "") -> list[str]:
    """選択モデルを最優先にしつつ、クォータ制限時のフォールバック順序リスト（Sarashina & Gemini 両対応）を生成"""
    gemini_defaults = ["gemini-3.8-flash", "gemini-3.1-pro", "gemini-4-argon", "gemini-2.0-flash"]
    sarashina_defaults = ["sarashina3-mini", "sarashina3-nano"]

    candidates = []
    if primary_model:
        candidates.append(primary_model)

    if is_sarashina_model(primary_model):
        for m in sarashina_defaults:
            if m not in candidates and sarashina_key:
                candidates.append(m)
        for m in gemini_defaults:
            if m not in candidates and gemini_key:
                candidates.append(m)
    else:
        for m in gemini_defaults:
            if m not in candidates and gemini_key:
                candidates.append(m)
        for m in sarashina_defaults:
            if m not in candidates and sarashina_key:
                candidates.append(m)

    if not candidates:
        candidates = [primary_model or "gemini-3.8-flash"]
    return candidates


def stream_gemini_response(client_ai, model_name: str, contents, sys_prompt: str, sarashina_key: str = "", gemini_key: str = ""):
    """Gemini および Sarashina AI 対応：429/503/404発生時は利用可能な別モデルへ自動フォールバックして生成"""
    candidates = get_candidate_models(model_name, gemini_key=gemini_key, sarashina_key=sarashina_key)
    last_err = None

    for current_m in candidates:
        for attempt in range(2):
            try:
                if is_sarashina_model(current_m):
                    text_out = call_sarashina_chat(sarashina_key, current_m, contents, sys_prompt)
                    if current_m != model_name:
                        st.toast(f"🔄 {model_name} の制限に達したため、自動的に {current_m} (Sarashina AI) に切り替えました！")
                    yield text_out
                    return
                else:
                    if client_ai is None:
                        if gemini_key:
                            client_ai = genai.Client(api_key=gemini_key)
                        else:
                            raise ValueError("Gemini API Key が未設定です。")
                    response_stream = client_ai.models.generate_content_stream(
                        model=current_m,
                        contents=contents,
                        config=types.GenerateContentConfig(system_instruction=sys_prompt)
                    )
                    yielded = False
                    for chunk in response_stream:
                        if not yielded and current_m != model_name:
                            st.toast(f"🔄 {model_name} の制限に達したため、自動的に {current_m} に切り替えました！")
                        if chunk.text:
                            yield chunk.text
                            yielded = True
                    if yielded:
                        return
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_quota = any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded", "Rate limit"])
                is_not_found = any(k in err_str for k in ["404", "NOT_FOUND"])
                if is_quota or is_not_found:
                    break
                is_busy = any(k in err_str for k in ["503", "UNAVAILABLE", "high demand", "502", "504"])
                if is_busy and attempt < 1:
                    time.sleep(2)
                    continue
                else:
                    break

    if last_err:
        raise last_err


class _SimpleTextResponse:
    def __init__(self, text: str):
        self.text = text


def generate_content_with_retry(client_ai, model_name: str, contents, config, sarashina_key: str = "", gemini_key: str = "", sys_prompt: str = ""):
    """構造化JSON生成（Gemini & Sarashina AI 両対応・429自動フォールバック付き）"""
    candidates = get_candidate_models(model_name, gemini_key=gemini_key, sarashina_key=sarashina_key)
    last_err = None

    json_schema_instruction = """

【重要：出力フォーマット厳守】
必ず以下のJSONスキーマ構造のみを純粋なJSON文字列として出力してください（前後の説明文は不要です）：
{
  "weekly_summary": "今週の全体方針・HRV/睡眠や外的要因への配慮・パワーカーブ弱点克服のアプローチ解説",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_label": "月曜",
      "is_rest_day": false,
      "title": "ワークアウトのタイトル (休養日は 'Rest Day')",
      "duration_min": 60,
      "target_tss": 65,
      "rationale": "この日のメニューの狙いとアドバイス",
      "workout_code": "Warmup\\n- 10m 55%\\nMain Set 3x\\n- 8m 95%\\n- 4m 55%\\nCooldown\\n- 10m 50%"
    }
  ]
}
"""

    for current_m in candidates:
        for attempt in range(2):
            try:
                if is_sarashina_model(current_m):
                    prompt_with_json = str(contents) + json_schema_instruction
                    raw_res = call_sarashina_chat(sarashina_key, current_m, prompt_with_json, sys_prompt)
                    clean_json = _extract_json_str(raw_res)
                    # JSONとしてパースできるか検証
                    json.loads(clean_json)
                    if current_m != model_name:
                        st.toast(f"🔄 {model_name} の制限に達したため、自動的に {current_m} (Sarashina AI) に切り替えました！")
                    return _SimpleTextResponse(clean_json)
                else:
                    if client_ai is None:
                        if gemini_key:
                            client_ai = genai.Client(api_key=gemini_key)
                        else:
                            raise ValueError("Gemini API Key が未設定です。")
                    res = client_ai.models.generate_content(
                        model=current_m,
                        contents=contents,
                        config=config
                    )
                    if current_m != model_name:
                        st.toast(f"🔄 {model_name} の制限に達したため、自動的に {current_m} に切り替えました！")
                    return res
            except Exception as e:
                last_err = e
                err_str = str(e)
                is_quota = any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded", "Rate limit"])
                is_not_found = any(k in err_str for k in ["404", "NOT_FOUND"])
                if is_quota or is_not_found:
                    break
                is_busy = any(k in err_str for k in ["503", "UNAVAILABLE", "high demand", "502", "504"])
                if is_busy and attempt < 1:
                    time.sleep(2)
                    continue
                else:
                    break

    if last_err:
        raise last_err


def show_ai_error(prefix: str, e: Exception, current_model: str):
    err_str = str(e)
    st.error(f"{prefix}: {err_str}")
    if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded", "Rate limit"]):
        st.warning("⚠️ 利用可能なAIモデルの利用枠上限（レート制限）に達しました。サイドバーで別モデル（Sarashina AI / Gemini）へ切り替えるか、しばらく待ってから再試行してください。")
    elif any(k in err_str for k in ["503", "UNAVAILABLE", "high demand"]):
        st.warning("⚠️ AIサーバーでアクセス集中（一時的な混雑）が発生しています。数秒待って再試行してください。")
    elif any(k in err_str for k in ["401", "403", "Unauthorized", "Invalid API"]):
        st.warning("⚠️ APIキーが無効または未設定の可能性があります。サイドバーの「⚙️ API & AIモデル設定」でAPIキーをご確認ください。")
    elif any(k in err_str for k in ["404", "NOT_FOUND"]):
        st.warning("⚠️ 選択されたモデルが利用できない可能性があります。サイドバーのモデル設定をご確認ください。")


# ==========================================
# 5. Streamlit UI 構築（プロチーム・テレメトリーUI）
# ==========================================
st.set_page_config(
    page_title="AI Cycling Coach Pro",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 3.8rem !important; padding-bottom: 6.5rem !important; max-width: 1320px; }

    /* KPI メトリクスカード：純白背景＋濃紺文字を <label data-testid="stMetricLabel"> 含め全階層で完全保証 */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #94a3b8 !important;
        border-left: 5px solid #0284c7 !important;
        padding: 0.85rem 1.05rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12) !important;
        color: #0f172a !important;
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] label *,
    [data-testid="stMetricLabel"],
    label[data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span,
    [data-testid="stMetricLabel"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stMetricValue"] p,
    [data-testid="stMetricValue"] span,
    [data-testid="stMetricValue"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] {
        background-color: #dcfce7 !important;
        color: #14532d !important;
        -webkit-text-fill-color: #14532d !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        border-radius: 999px !important;
        padding: 0.15rem 0.55rem !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"],
    [data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stMetricDelta"] p,
    [data-testid="stMetricDelta"] span,
    [data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] svg {
        color: #14532d !important;
        -webkit-text-fill-color: #14532d !important;
        fill: #15803d !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    /* トップ・テレメトリーバナー：爽やかなスカイブルーグラデーション＋濃紺テキスト */
    .telemetry-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
        border: 1px solid #38bdf8 !important;
        border-left: 6px solid #0284c7 !important;
        border-radius: 0.85rem !important;
        padding: 1.0rem 1.35rem !important;
        margin-bottom: 1.0rem !important;
        color: #0f172a !important;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.12) !important;
    }
    .telemetry-banner h2,
    .telemetry-banner span,
    .telemetry-banner p,
    .telemetry-banner div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    .telemetry-sub {
        color: #1e293b !important;
        -webkit-text-fill-color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* ステータスピルバッジ：白背景＋ブルー枠＋濃青テキストでくっきり表示 */
    .status-pill {
        display: inline-block;
        padding: 0.24rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-top: 0.2rem;
        background: #ffffff !important;
        color: #0369a1 !important;
        -webkit-text-fill-color: #0369a1 !important;
        border: 1px solid #0284c7 !important;
        box-shadow: 0 1px 2px rgba(2, 132, 199, 0.1);
    }

    /* 情報・警告アラートボックス (st.info / st.warning / st.success)：ライトブルー背景＋濃紺文字を完全保証 */
    [data-testid="stAlert"] {
        background-color: #e0f2fe !important;
        border: 1px solid #38bdf8 !important;
        border-left: 5px solid #0284c7 !important;
        border-radius: 0.65rem !important;
        color: #0f172a !important;
    }
    [data-testid="stAlert"] *,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] li,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] strong,
    [data-testid="stAlert"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }

    /* エクスパンダー：白カード背景に対し、ダークモード時でも内部のタイトル・説明文・箇条書き・キャプションを全て濃紺(#0f172a)に固定 */
    [data-testid="stExpander"] {
        border: 1px solid #94a3b8 !important;
        border-radius: 0.75rem !important;
        background: #ffffff !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08) !important;
        margin-bottom: 0.85rem !important;
        overflow: hidden !important;
        color: #0f172a !important;
    }
    [data-testid="stExpander"] details summary {
        background: linear-gradient(90deg, #e0f2fe 0%, #f1f5f9 100%) !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 700 !important;
        padding: 0.7rem 1rem !important;
        border-bottom: 1px solid #bae6fd !important;
    }
    [data-testid="stExpander"] details summary *,
    [data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] details summary [data-testid="stMarkdownContainer"] *,
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary svg {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        fill: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }
    [data-testid="stExpanderDetails"] {
        background: #ffffff !important;
        color: #0f172a !important;
        padding-top: 0.8rem !important;
    }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] em,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] h6,
    [data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"],
    [data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] *,
    [data-testid="stExpanderDetails"] label,
    [data-testid="stExpanderDetails"] label * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] a {
        color: #0284c7 !important;
        -webkit-text-fill-color: #0284c7 !important;
        font-weight: 700 !important;
        text-decoration: underline !important;
    }
    [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] code {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        padding: 0.15rem 0.35rem !important;
        border-radius: 4px !important;
    }
    [data-testid="stExpanderDetails"] input,
    [data-testid="stExpanderDetails"] textarea {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border: 1px solid #94a3b8 !important;
    }

    /* 通常のキャプションもダークモードで薄くなりすぎないよう不透明度を確保 */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        opacity: 0.95 !important;
        font-size: 0.88rem !important;
    }

    /* タブバー（React Aria [data-testid="stTab"] & BaseWeb 両対応）：非選択タブも明瞭なピル背景＋濃紺文字、選択タブはブルー背景＋白文字 */
    button[data-baseweb="tab"],
    [data-testid="stTab"] {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.05rem !important;
        border-radius: 0.55rem 0.55rem 0 0 !important;
        margin-right: 0.3rem !important;
        border: 1px solid #cbd5e1 !important;
    }
    button[data-baseweb="tab"] *,
    [data-testid="stTab"] *,
    [data-testid="stTab"] [data-testid="stMarkdownContainer"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    [data-testid="stTab"][aria-selected="true"],
    [data-testid="stTab"][data-selected] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #0284c7 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] *,
    [data-testid="stTab"][aria-selected="true"] *,
    [data-testid="stTab"][data-selected] *,
    [data-testid="stTab"][aria-selected="true"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stTab"][data-selected] [data-testid="stMarkdownContainer"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* AIコーチ相談のチャット入力欄を常に画面最下部に固定表示 */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(1280px, 94vw) !important;
        z-index: 999 !important;
        background: #1e293b !important;
        padding: 0.65rem 1rem 0.9rem 1rem !important;
        border-top: 2px solid #0284c7 !important;
        box-shadow: 0 -6px 20px rgba(0, 0, 0, 0.3) !important;
        border-radius: 0.85rem 0.85rem 0 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- マルチユーザー & オーナー自動ログイン判定（ゲスト画面からオーナー画面への侵入を完全遮断） ---
owner_cfg = load_owner_config()
owner_master_token = compute_owner_token(owner_cfg.get("intervals_key", ""))
q_mode = st.query_params.get("mode", "")
q_uid = st.query_params.get("uid", "")
q_token = st.query_params.get("token", "")

# 1. URLに ?mode=guest が指定された場合、または一度でもゲストモードに入ったセッションは locked_as_guest = True にロック
if q_mode == "guest" or q_uid:
    st.session_state.locked_as_guest = True
    st.session_state.user_mode = "guest"
    if q_uid:
        st.session_state.guest_uid = q_uid

# 2. 正しいオーナートークン (?token=...) がURLにある場合のみロック解除を許可
if owner_master_token and q_token == owner_master_token:
    st.session_state.locked_as_guest = False
    st.session_state.user_mode = "owner"

if "user_mode" not in st.session_state:
    if st.session_state.get("locked_as_guest", False):
        st.session_state.user_mode = "guest"
    elif owner_cfg.get("public_default_guest", False) or not owner_cfg.get("intervals_key"):
        st.session_state.locked_as_guest = True
        st.session_state.user_mode = "guest"
    else:
        # オーナーが通常URLを開いた場合は今まで通り自動ログイン
        st.session_state.user_mode = "owner"

# もし locked_as_guest が True なのに user_mode が owner になっていたら強制的に guest に戻す
if st.session_state.get("locked_as_guest", False):
    st.session_state.user_mode = "guest"

is_owner_mode = (st.session_state.user_mode == "owner")

# 設定とAI記憶のロード（現在のユーザー専用領域から読み込み）
cfg = load_config()
mem = load_memory()

if "metrics" not in st.session_state:
    st.session_state.metrics = load_metrics_cache()
if "messages" not in st.session_state:
    st.session_state.messages = mem.get("messages", [])
if "generated_workout" not in st.session_state:
    st.session_state.generated_workout = mem.get("generated_workout")
if "daily_workout_history" not in st.session_state:
    st.session_state.daily_workout_history = mem.get("daily_workout_history", [])
if "weekly_plan" not in st.session_state:
    st.session_state.weekly_plan = mem.get("weekly_plan")
if "weekly_plan_history" not in st.session_state:
    st.session_state.weekly_plan_history = mem.get("weekly_plan_history", [])
if "ride_reviews" not in st.session_state:
    st.session_state.ride_reviews = mem.get("ride_reviews", {})

race_options = [
    "ヒルクライム (60分前後・L4/SST持続力重視)",
    "ショートヒルクライム (20〜40分・L4〜VO2max重視)",
    "ロードレース / サーキット (インターバル耐性・VO2max・スプリント)",
    "クリテリウム (高強度インターバル・無酸素反復能力)",
    "エンデューロ / ロングライド (有酸素耐久性・脂質代謝)"
]

# --- サイドバー：設定・外的要因・バックアップ ---
with st.sidebar:
    st.header("👤 アカウント状態")
    if is_owner_mode:
        st.success(f"👑 オーナー自動ログイン中 ({cfg.get('athlete_id') or '設定済'})")
        if st.button("👥 ゲスト画面（他ユーザー用）をプレビュー", use_container_width=True):
            st.session_state.locked_as_guest = True
            st.session_state.user_mode = "guest"
            st.session_state.guest_uid = ""
            st.session_state.guest_config = {}
            try:
                st.query_params["mode"] = "guest"
                if "uid" in st.query_params:
                    del st.query_params["uid"]
            except Exception:
                pass
            reload_user_session_data()
            st.rerun()

        with st.expander("🔗 他の人への共有URL ＆ セキュリティ設定"):
            st.caption("① 他の人に共有する招待URL（このURLからはオーナー画面には絶対に入れません）：")
            st.code("https://ai-cycling-coach-ps662nnyskxx8krt7w7rnj.streamlit.app/?mode=guest", language="text")
            if owner_master_token:
                st.caption("② あなた専用の暗号化トークン付き自動ログインURL（ブックマーク推奨）：")
                st.code(f"https://ai-cycling-coach-ps662nnyskxx8krt7w7rnj.streamlit.app/?token={owner_master_token}", language="text")
            pub_default = st.checkbox(
                "🔒 通常URL（パラメータなし）もゲスト入力画面にする",
                value=bool(cfg.get("public_default_guest", False)),
                help="ONにすると、パラメータなしのURLを開いた人も全員ゲスト画面になり、あなたは上の ?token=... 付きURL（または自分のIntervals API Key入力）でのみオーナー画面に入れます。"
            )
            if pub_default != bool(cfg.get("public_default_guest", False)):
                cfg["public_default_guest"] = pub_default
                save_config(cfg)
                st.success("セキュリティ設定を更新しました！")
    else:
        # ゲスト画面ではオーナー画面への切り替えボタン等を一切表示しない
        pub_default = bool(owner_cfg.get("public_default_guest", False))
        if cfg.get("intervals_key") and (cfg.get("gemini_key") or cfg.get("sarashina_key")):
            st.info(f"🚴 ログイン中 (ID: {cfg.get('athlete_id') or '自動取得'})")
            if st.session_state.get("guest_uid"):
                st.caption("💡 現在のURL（`?mode=guest&uid=...`）をブックマークすると次回から自動ログインできます。")
            if st.button("🚪 ログアウト / APIキー再入力", use_container_width=True):
                st.session_state.guest_uid = ""
                st.session_state.guest_config = {}
                try:
                    st.query_params["mode"] = "guest"
                    if "uid" in st.query_params:
                        del st.query_params["uid"]
                except Exception:
                    pass
                reload_user_session_data()
                st.rerun()
        else:
            st.warning("🔑 未ログイン（メイン画面でAPIキーを入力してください）")

    st.divider()
    st.header("⚙️ API & AIモデル設定")

    if is_owner_mode:
        athlete_id = cfg.get("athlete_id", "")
        intervals_key = cfg.get("intervals_key", "")
        gemini_key = cfg.get("gemini_key", "")
        sarashina_key = cfg.get("sarashina_key", "")
        st.caption(
            f"🔑 登録状態 — Gemini: **{'✅設定済' if gemini_key else '未設定'}** ／ Sarashina: **{'✅設定済' if sarashina_key else '未設定'}**"
        )
        with st.expander("🔑 オーナーAPIキーの登録・変更（非表示保護中）", expanded=(not sarashina_key)):
            st.caption("セキュリティ保護のため保存済みキーは非表示です。新規登録または変更する項目のみ入力し、下の「💾 設定を保存する」を押してください。")
            new_aid = st.text_input("Intervals Athlete ID", value=athlete_id, placeholder="例: i123456")
            new_ikey = st.text_input("新しい Intervals API Key", value="", type="password", placeholder="変更しない場合は空欄")
            new_gkey = st.text_input("新しい Gemini API Key", value="", type="password", placeholder="変更しない場合は空欄")
            new_skey = st.text_input("Sarashina AI API Key", value="", type="password", placeholder="hki_v1_... (未登録・変更時に入力)")
            if new_aid.strip():
                athlete_id = new_aid.strip()
            if new_ikey.strip():
                intervals_key = new_ikey.strip()
            if new_gkey.strip():
                gemini_key = new_gkey.strip()
            if new_skey.strip():
                sarashina_key = new_skey.strip()
    else:
        athlete_id = st.text_input("Intervals Athlete ID (空欄で自動取得)", value=cfg.get("athlete_id", ""), placeholder="例: i123456")
        intervals_key = st.text_input("Intervals API Key", value=cfg.get("intervals_key", ""), type="password")
        gemini_key = st.text_input("Gemini API Key (Gemini利用時)", value=cfg.get("gemini_key", ""), type="password")
        sarashina_key = st.text_input("Sarashina AI API Key (Sarashina利用時)", value=cfg.get("sarashina_key", ""), type="password", placeholder="hki_v1_...")

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.1-pro",
        "gemini-4-argon",
        "sarashina3-mini",
        "sarashina3-nano",
        "カスタム指定...",
    ]
    cur_model = cfg.get("model", "gemini-3.8-flash")
    idx_model = model_options.index(cur_model) if cur_model in model_options else (len(model_options) - 1 if cur_model else 0)
    selected_option = st.selectbox(
        "🤖 AIモデル選択 (Gemini / Sarashina AI)",
        model_options,
        index=idx_model,
        help="Google Gemini シリーズおよび SB Intuitions 国産LLM『Sarashina AI (sarashina3-mini / sarashina3-nano)』を選択できます。利用枠制限（429）時は設定済みの別モデルへ自動フォールバックします。"
    )
    if selected_option == "カスタム指定...":
        custom_model = st.text_input("モデルIDを入力 (例: sarashina3-mini 等)", value=cur_model if cur_model not in model_options else "sarashina3-mini")
        selected_model = custom_model.strip()
    else:
        selected_model = selected_option

    if is_sarashina_model(selected_model) and not sarashina_key:
        st.warning("⚠️ Sarashina AI が選択されています。上の「Sarashina AI API Key」を入力し、下の「💾 設定を保存する」を押してください。")
    elif not is_sarashina_model(selected_model) and not gemini_key:
        st.warning("⚠️ Gemini モデルが選択されています。Gemini API Key を入力して保存してください。")

    st.divider()
    st.subheader("🏥 外的要因・怪我・病気の設定")
    st.caption("事故や体調不良、部位の痛みなどを最優先で考慮してメニューを安全に調整します。")
    ext_status_options = [
        "なし（通常通り）",
        "🤒 病気・体調不良（風邪・発熱・病み上がり等）",
        "🤕 事故・落車・怪我からの回復中",
        "🦵 身体の痛み・違和感（膝・腰・首など）",
        "💼 仕事多忙・出張・強い精神的ストレス",
        "🔄 長期ブランクからの復帰期",
    ]
    cur_ext_status = cfg.get("ext_status", ext_status_options[0])
    idx_ext = ext_status_options.index(cur_ext_status) if cur_ext_status in ext_status_options else 0
    ext_status = st.selectbox("現在の状態区分", ext_status_options, index=idx_ext)

    ext_cap_options = [
        "制限なし（コンディションに応じる）",
        "L3（テンポ / SST下限）までに制限",
        "L2（有酸素持久走・Z2）までに制限",
        "L1（アクティブリカバリー・回復走）のみ",
        "完全休養（バイクなし・ケアのみ）",
    ]
    cur_ext_cap = cfg.get("ext_intensity_cap", ext_cap_options[0])
    idx_cap = ext_cap_options.index(cur_ext_cap) if cur_ext_cap in ext_cap_options else 0
    ext_intensity_cap = st.selectbox("強度上限リミッター", ext_cap_options, index=idx_cap)

    ext_detail = st.text_area(
        "症状・制限の詳細メモ",
        value=cfg.get("ext_detail", ""),
        placeholder="例: 3日前に落車し右肩打撲、ダンシング不可 / 風邪の病み上がり2日目で心拍140bpm以下に抑えたい 等"
    )

    st.divider()
    st.subheader("⚡ フィジカル & 目標レース")
    ftp = st.number_input("現在のFTP (W)", 100, 500, int(cfg.get("ftp", 280)), 5)
    weight = st.number_input("現在の体重 (kg)", 35.0, 120.0, float(cfg.get("weight", 60.0)), 0.5)
    target_ftp = st.number_input("目標FTP (W)", 100, 550, int(cfg.get("target_ftp", 300)), 5)

    race_name = st.text_input("目標レース名", value=cfg.get("race_name", "Mt.富士ヒルクライム"))
    try:
        default_rdate = datetime.strptime(cfg.get("race_date", ""), "%Y-%m-%d").date()
    except Exception:
        default_rdate = datetime.now().date() + timedelta(days=60)
    race_date = st.date_input("開催日", value=default_rdate)

    cur_race_type = cfg.get("race_type", race_options[0])
    idx_race = race_options.index(cur_race_type) if cur_race_type in race_options else 0
    race_type = st.selectbox("レース特性", race_options, index=idx_race)
    race_goal = st.text_area("目標タイム・課題メモ", value=cfg.get("race_goal", ""))

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        weekday_hours = st.number_input("平日枠(h/日)", 0.5, 5.0, float(cfg.get("weekday_hours", 1.5)), 0.5)
    with col_w2:
        weekend_hours = st.number_input("休日枠(h/日)", 1.0, 10.0, float(cfg.get("weekend_hours", 3.5)), 0.5)

    current_cfg_dict = {
        "athlete_id": athlete_id,
        "athlete_name": cfg.get("athlete_name", ""),
        "intervals_key": intervals_key,
        "gemini_key": gemini_key,
        "sarashina_key": sarashina_key,
        "model": selected_model,
        "ftp": ftp,
        "weight": weight,
        "target_ftp": target_ftp,
        "race_name": race_name,
        "race_date": race_date.strftime("%Y-%m-%d"),
        "race_type": race_type,
        "race_goal": race_goal,
        "weekday_hours": weekday_hours,
        "weekend_hours": weekend_hours,
        "ext_status": ext_status,
        "ext_detail": ext_detail,
        "ext_intensity_cap": ext_intensity_cap,
        "public_default_guest": pub_default,
    }

    if st.button("💾 設定を保存する", use_container_width=True):
        save_config(current_cfg_dict)
        st.success("✅ 設定を保存しました！")

    st.divider()
    with st.expander("🧠 記憶データのバックアップ / 管理"):
        st.caption("チャット履歴、過去のライド評価、生成したメニューの記憶を管理します。")
        backup_payload = json.dumps({
            "config": current_cfg_dict,
            "memory": {
                "messages": st.session_state.messages,
                "generated_workout": st.session_state.generated_workout,
                "daily_workout_history": st.session_state.daily_workout_history,
                "weekly_plan": st.session_state.weekly_plan,
                "weekly_plan_history": st.session_state.weekly_plan_history,
                "ride_reviews": st.session_state.ride_reviews,
            }
        }, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 全記憶・設定をバックアップ保存 (.json)",
            data=backup_payload,
            file_name=f"ai_coach_backup_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
        uploaded_backup = st.file_uploader("📤 バックアップから復元", type=["json"])
        if uploaded_backup is not None:
            if st.button("復元を実行する", use_container_width=True):
                try:
                    loaded_bk = json.loads(uploaded_backup.read().decode("utf-8"))
                    if "config" in loaded_bk:
                        save_config(loaded_bk["config"])
                    if "memory" in loaded_bk:
                        m_bk = loaded_bk["memory"]
                        st.session_state.messages = m_bk.get("messages", [])
                        st.session_state.generated_workout = m_bk.get("generated_workout")
                        st.session_state.daily_workout_history = m_bk.get("daily_workout_history", [])
                        st.session_state.weekly_plan = m_bk.get("weekly_plan")
                        st.session_state.weekly_plan_history = m_bk.get("weekly_plan_history", [])
                        st.session_state.ride_reviews = m_bk.get("ride_reviews", {})
                        save_memory()
                    st.success("✅ バックアップから記憶と設定を復元しました！ページを再読み込みしてください。")
                except Exception as e:
                    st.error(f"復元エラー: {e}")

# --- ゲスト（他の人）が未ログインの場合：オーナー画面への導線を一切持たない専用ログイン画面を表示 ---
if not is_owner_mode and (not intervals_key or (not gemini_key and not sarashina_key)):
    st.markdown("""
    <div class="telemetry-banner">
        <h2 style="margin:0; font-size:1.6rem; color:#0f172a; font-weight:800;">🚴‍♂️ AI Cycling Coach Pro</h2>
        <p style="margin:0.35rem 0 0 0; color:#1e293b; font-weight:600; font-size:0.95rem;">
            Intervals.icu の生理学データ（HRV・睡眠・CP/W'・Stravaセグメント）× Gemini / Sarashina AI によるパーソナル・ロードレースコーチングシステム
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("🔑 ご自身の **Intervals.icu API Key** と、**Gemini API Key または Sarashina AI API Key**（両方設定も可）を入力してログインしてください。入力されたデータ・記憶はあなた専用の暗号化領域に分離保存されます。")

    with st.expander("📖 APIキーの取得方法（初回のみ・約1分）", expanded=True):
        st.markdown("""
1. **Intervals.icu API Key の取得**
   - **[Intervals.icu の設定ページ (Settings)](https://intervals.icu/settings)** を開きます。
   - ページ下部にある **「Developer Settings」** セクションの **「API Key」**（「View」または生成ボタン）をコピーします。
   - ※ Athlete ID（`i123456` 等）は空欄のままでもAPIキーから自動認識されます。
2. **AI API Key の取得（Gemini または Sarashina AI のどちらか1つ以上でOK）**
   - **Google Gemini API Key**: **[Google AI Studio (API Keys)](https://aistudio.google.com/app/apikey)** で「Create API key」を押して発行。
   - **Sarashina AI API Key**: **SB Intuitions Sarashina API ポータル**から発行されたシークレットキー（`hki_v1_...`）をコピー。
        """)

    with st.form("guest_onboarding_form"):
        st.subheader("🚀 アカウント接続 & 目標セットアップ")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            in_intervals_key = st.text_input("1️⃣ Intervals.icu API Key (必須)", type="password", placeholder="Intervals.icu の API Key を貼り付け")
            in_athlete_id = st.text_input("Intervals Athlete ID (任意・空欄で自動取得)", placeholder="例: i123456（空欄でも自動検出されます）")
            auto_fetch_phys = st.checkbox("✅ 現在のFTPと体重を Intervals.icu から自動取得する", value=True)
        with f_col2:
            in_gemini_key = st.text_input("2️⃣-A Gemini API Key (Gemini利用時)", type="password", placeholder="AIza... から始まるキー")
            in_sarashina_key = st.text_input("2️⃣-B Sarashina AI API Key (Sarashina利用時)", type="password", placeholder="hki_v1_... から始まるキー")

        st.markdown("##### 🏆 目標レース・フィジカル設定（ログイン後もサイドバーで変更可能）")
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            in_ftp = st.number_input("現在のFTP (W) ※自動取得時は上書き", 100, 500, 250, 5)
            in_weight = st.number_input("現在の体重 (kg) ※自動取得時は上書き", 35.0, 120.0, 60.0, 0.5)
        with g_col2:
            in_target_ftp = st.number_input("目標FTP (W)", 100, 550, 280, 5)
            in_race_name = st.text_input("目標レース名 / イベント名", value="Mt.富士ヒルクライム")
        with g_col3:
            in_race_date = st.date_input("目標開催日", value=datetime.now().date() + timedelta(days=60))
            in_race_type = st.selectbox("レース特性", race_options)

        submitted_login = st.form_submit_button("⚡ ログインしてAIコーチングを開始", type="primary", use_container_width=True)

    if submitted_login:
        if not in_intervals_key.strip() or (not in_gemini_key.strip() and not in_sarashina_key.strip()):
            st.error("⚠️ Intervals.icu API Key と、Gemini API Key または Sarashina AI API Key の少なくとも一方を入力してください。")
        else:
            # もし入力された Intervals API Key がオーナー本人の秘密キーと完全一致した場合はオーナーモードとして認証
            if owner_cfg.get("intervals_key") and in_intervals_key.strip() == owner_cfg.get("intervals_key", "").strip():
                st.session_state.locked_as_guest = False
                st.session_state.user_mode = "owner"
                try:
                    st.query_params.clear()
                    if owner_master_token:
                        st.query_params["token"] = owner_master_token
                except Exception:
                    pass
                reload_user_session_data()
                st.rerun()

            try:
                with st.spinner("Intervals.icu に接続し、生理学データ・パワーカーブを解析中..."):
                    temp_client = IntervalsClient(in_athlete_id.strip(), in_intervals_key.strip())
                    prof = temp_client.get_athlete_profile()
                    resolved_id = prof.get("athlete_id") or in_athlete_id.strip() or "0"
                    resolved_name = prof.get("name") or ""
                    resolved_ftp = prof.get("ftp") if (auto_fetch_phys and prof.get("ftp")) else int(in_ftp)
                    resolved_weight = prof.get("weight") if (auto_fetch_phys and prof.get("weight")) else float(in_weight)

                    uid = compute_user_uid(in_intervals_key.strip(), resolved_id)
                    st.session_state.guest_uid = uid
                    reload_user_session_data()

                    default_guest_model = "sarashina3-mini" if (in_sarashina_key.strip() and not in_gemini_key.strip()) else selected_model
                    new_guest_cfg = current_cfg_dict.copy()
                    new_guest_cfg.update({
                        "athlete_id": resolved_id,
                        "athlete_name": resolved_name,
                        "intervals_key": in_intervals_key.strip(),
                        "gemini_key": in_gemini_key.strip(),
                        "sarashina_key": in_sarashina_key.strip(),
                        "model": default_guest_model,
                        "ftp": resolved_ftp,
                        "weight": resolved_weight,
                        "target_ftp": int(in_target_ftp),
                        "race_name": in_race_name,
                        "race_date": in_race_date.strftime("%Y-%m-%d"),
                        "race_type": in_race_type,
                    })
                    save_config(new_guest_cfg)

                    authed_client = IntervalsClient(resolved_id, in_intervals_key.strip())
                    wellness = authed_client.get_wellness(days=40)
                    activities = authed_client.get_activities(days=40)
                    pdc_raw = authed_client.get_power_curves(curves="1y")
                    st.session_state.metrics = parse_intervals_data(wellness, activities, pdc_raw, resolved_weight, resolved_ftp)
                    save_metrics_cache(st.session_state.metrics)

                st.success(f"✅ ログイン完了！（Athlete ID: {resolved_id} / FTP: {resolved_ftp}W / 体重: {resolved_weight}kg）")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 接続エラー: APIキーまたはAthlete IDをご確認ください。（詳細: {e}）")

    st.stop()

# ==========================================
# ログイン済みメインダッシュボード
# ==========================================
profile_data = {
    "ftp": ftp,
    "weight": weight,
    "target_ftp": target_ftp,
    "race_name": race_name,
    "race_date": race_date,
    "race_type": race_type,
    "race_goal": race_goal,
    "weekday_hours": weekday_hours,
    "weekend_hours": weekend_hours,
    "ext_status": ext_status,
    "ext_detail": ext_detail,
    "ext_intensity_cap": ext_intensity_cap,
}


def sync_intervals_now(show_msg=True):
    global athlete_id
    if not intervals_key:
        if show_msg:
            st.warning("左上の「＞」からサイドバーを開き、Intervals API Keyを入力・保存してください。")
        return
    try:
        with st.spinner("Intervals.icuから最新データ（HRV・睡眠・PMC・1年PDC・40日履歴）を取得中..."):
            client = IntervalsClient(athlete_id, intervals_key)
            if not athlete_id or athlete_id == "0":
                try:
                    prof = client.get_athlete_profile()
                    if prof.get("athlete_id"):
                        athlete_id = prof["athlete_id"]
                        current_cfg_dict["athlete_id"] = athlete_id
                        client = IntervalsClient(athlete_id, intervals_key)
                except Exception:
                    pass
            save_config(current_cfg_dict)
            wellness = client.get_wellness(days=40)
            activities = client.get_activities(days=40)
            pdc_raw = client.get_power_curves(curves="1y")
            st.session_state.metrics = parse_intervals_data(wellness, activities, pdc_raw, weight, ftp)
            save_metrics_cache(st.session_state.metrics)
        if show_msg:
            st.toast("✅ Intervals.icu 同期＆生理学テレメトリー解析完了！")
    except Exception as e:
        err_msg = str(e)
        resp_text = ""
        if hasattr(e, "response") and e.response is not None:
            resp_text = e.response.text
        if "Maintenance" in resp_text:
            st.warning(f"🚧 **Intervals.icu 公式サーバーが現在メンテナンス中です**（{resp_text.strip()}）。終了までしばらくお待ちください。")
        elif "503" in err_msg or "Service Unavailable" in err_msg:
            st.warning("🚧 **Intervals.icu 公式サーバーが現在メンテナンスまたは高負荷となっています**。終了までしばらくお待ちください。")
        else:
            st.error(f"同期エラー: {e}")

        if st.session_state.metrics is None:
            cached = load_metrics_cache()
            if cached:
                st.session_state.metrics = cached
                st.info(f"ℹ️ 前回の同期キャッシュデータ（{cached.get('cached_at', '以前')} 取得分）を自動適用しました。")

# 初回起動時（またはキャッシュに最新項目がない場合）に自動同期
if (
    st.session_state.metrics is None
    or "wellness_readiness" not in st.session_state.metrics
    or "pmc_history" not in st.session_state.metrics
    or "cp" not in st.session_state.metrics.get("pdc", {})
) and intervals_key:
    sync_intervals_now(show_msg=False)

# --- プロチーム・テレメトリーヘッダー ---
days_to_race = (race_date - datetime.now().date()).days
phase_str = determine_training_phase(days_to_race)
user_badge = f"👑 オーナー ({athlete_id})" if is_owner_mode else f"🚴 アスリート ({athlete_id or 'User'})"
phenotype_badge = st.session_state.metrics["pdc"].get("phenotype", "All-Rounder") if (st.session_state.metrics and "pdc" in st.session_state.metrics) else "All-Rounder"

hdr_col1, hdr_col2, hdr_col3 = st.columns([3.4, 1, 1])
with hdr_col1:
    st.markdown(f"""
    <div class="telemetry-banner">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <span style="font-size:1.4rem; font-weight:800; color:#0f172a;">🚴‍♂️ AI Cycling Coach Pro</span>
                <span style="margin-left:0.65rem; color:#0369a1; font-weight:700; font-size:0.92rem; background:#ffffff; padding:0.2rem 0.6rem; border-radius:6px; border:1px solid #bae6fd;">{user_badge}</span>
            </div>
            <div>
                <span class="status-pill">🏆 {race_name} まであと {days_to_race}日</span>
                <span class="status-pill">{phase_str}</span>
                <span class="status-pill">{phenotype_badge}</span>
                <span class="status-pill">🤖 {selected_model}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with hdr_col2:
    if st.button("🔄 最新データ同期", use_container_width=True):
        sync_intervals_now(show_msg=True)
with hdr_col3:
    if is_owner_mode:
        if st.button("👥 ゲスト画面へ切替", use_container_width=True, help="他の人がAPIを入力して利用するゲスト画面に切り替えます"):
            st.session_state.locked_as_guest = True
            st.session_state.user_mode = "guest"
            st.session_state.guest_uid = ""
            st.session_state.guest_config = {}
            try:
                st.query_params["mode"] = "guest"
                if "uid" in st.query_params:
                    del st.query_params["uid"]
            except Exception:
                pass
            reload_user_session_data()
            st.rerun()
    else:
        if st.button("🚪 ログアウト", use_container_width=True, help="ログアウトしてAPIキー入力画面に戻ります"):
            st.session_state.guest_uid = ""
            st.session_state.guest_config = {}
            try:
                st.query_params["mode"] = "guest"
                if "uid" in st.query_params:
                    del st.query_params["uid"]
            except Exception:
                pass
            reload_user_session_data()
            st.rerun()

# 外的要因（事故・病気・怪我等）が設定されている場合の警告バナー
if ext_status != "なし（通常通り）" or ext_detail.strip() or ext_intensity_cap != "制限なし（コンディションに応じる）":
    st.warning(
        f"🏥 **外的要因・セーフティモード適用中:** 【{ext_status}】 / **上限リミッター:** {ext_intensity_cap}"
        + (f"\n\n📝 **詳細メモ:** {ext_detail}" if ext_detail.strip() else "")
    )

current_pwr = round(ftp / weight, 2) if weight > 0 else 0
target_pwr = round(target_ftp / weight, 2) if weight > 0 else 0
gap_watts = target_ftp - ftp

# --- コア・テレメトリー KPI メトリクス (5カラム) ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("⚡ 現在 PWR (FTP)", f"{current_pwr} W/kg", f"{ftp}W → 目標{target_ftp}W ({gap_watts:+}W)")
if st.session_state.metrics:
    ctl_v = st.session_state.metrics["ctl"]
    atl_v = st.session_state.metrics["atl"]
    tsb_v = st.session_state.metrics["tsb"]
    ramp_v = st.session_state.metrics.get("ramp_rate", 0.0)
    ramp_status = "適正ビルド" if 2.0 <= ramp_v <= 6.0 else ("急上昇注意" if ramp_v > 6.0 else "維持/回復")
    m2.metric("📈 CTL (長期フィットネス)", ctl_v, f"Ramp: {ramp_v:+} /週 ({ramp_status})")
    atl_status = "高負荷期" if atl_v > ctl_v + 5 else ("回復傾向" if atl_v < ctl_v - 5 else "CTLと均衡")
    m3.metric("🔥 ATL (短期疲労)", atl_v, atl_status)
    tsb_status = "フレッシュ (好調)" if tsb_v > 5 else ("最適鍛錬ゾーン" if tsb_v >= -20 else "過疲労警戒")
    m4.metric("⚖️ TSB (フォーム/余力)", tsb_v, tsb_status)
    wr_quick = st.session_state.metrics.get("wellness_readiness", {})
    r_score = wr_quick.get("score", "--")
    r_lbl = wr_quick.get("status_label", "未取得").split(" (")[0]
    m5.metric("🫀 本日のReadiness", f"{r_score} / 100", r_lbl)
else:
    m2.metric("📈 CTL", "--")
    m3.metric("🔥 ATL", "--")
    m4.metric("⚖️ TSB", "--")
    m5.metric("🫀 Readiness", "--")

# --- エクスパンダー1: HRV・睡眠レディネス & 40日PMC推移 ---
if st.session_state.metrics and st.session_state.metrics.get("wellness_readiness"):
    wr = st.session_state.metrics["wellness_readiness"]
    with st.expander(f"🫀 自律神経(HRV)・睡眠レディネス診断 ＆ 40日PMC推移 — 【{wr['status_label']} / スコア: {wr['score']}点】", expanded=True):
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        hrv_delta = f"40日中央値: {wr['hrv_base']} ms" if wr.get("hrv_base") else None
        w_col1.metric("❤️ 最新 HRV (rMSSD)", f"{wr['hrv']} ms" if wr.get("hrv") else "--", hrv_delta)

        slp_delta = f"睡眠時間: {wr['sleep_hours']}h" if wr.get("sleep_hours") else None
        w_col2.metric("😴 睡眠スコア", f"{wr['sleep_score']} 点" if wr.get("sleep_score") else "--", slp_delta)

        rhr_delta = f"40日中央値: {wr['rhr_base']} bpm" if wr.get("rhr_base") else None
        w_col3.metric("💓 安静時心拍 (RHR)", f"{wr['rhr']} bpm" if wr.get("rhr") else "--", rhr_delta)

        pdc_quick = st.session_state.metrics.get("pdc", {})
        cp_v = pdc_quick.get("cp", 0)
        wp_v = pdc_quick.get("w_prime_kj", 0.0)
        w_col4.metric("🧬 臨界パワー CP / W'", f"{cp_v} W" if cp_v else "--", f"無酸素容量 W': {wp_v} kJ" if wp_v else None)

        st.info(f"**【AIスポーツ科学判定 & 本日の推奨強度】**\n\n{wr['recommendation']}\n\n- " + "\n- ".join(wr.get("reasons", [])))

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            w_hist = st.session_state.metrics.get("wellness_history", [])
            w_chart = build_wellness_chart(w_hist)
            if w_chart:
                st.caption("📈 直近14日間の HRV (紫・実線) ＆ 睡眠スコア (水色・破線) 推移")
                st.altair_chart(w_chart, use_container_width=True, theme=None)
        with ch_col2:
            pmc_hist = st.session_state.metrics.get("pmc_history", [])
            pmc_chart = build_pmc_chart(pmc_hist)
            if pmc_chart:
                st.caption("📊 直近40日間の PMC推移 — 水色線:CTL / 赤破線:ATL / 棒:TSB(フォーム)")
                st.altair_chart(pmc_chart, use_container_width=True, theme=None)

# --- エクスパンダー2: 過去1年のパワーカーブ (PDC) & CP/W'・5軸脚質・7ゾーン一覧 ---
if st.session_state.metrics and "pdc" in st.session_state.metrics:
    pdc = st.session_state.metrics["pdc"]
    with st.expander(f"📈 過去1年のパワーカーブ (PDC) ＆ CP/W' 脚質プロファイリング — 【{pdc.get('phenotype', '')}】", expanded=False):
        st.info(pdc["diagnosis"])

        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        pc1.metric("15秒 (Sprint)", f"{pdc['watts']['15s']} W", f"{pdc['wkg']['15s']} W/kg")
        pc2.metric("1分 (Anaerobic)", f"{pdc['watts']['1m']} W", f"{pdc['wkg']['1m']} W/kg")
        pc3.metric("5分 (VO2max)", f"{pdc['watts']['5m']} W", f"{pdc['wkg']['5m']} W/kg")
        pc4.metric("20分 (Threshold)", f"{pdc['watts']['20m']} W", f"{pdc['wkg']['20m']} W/kg")
        pc5.metric("60分 / eFTP", f"{pdc['watts']['60m']} W", f"eFTP: {pdc['eftp'] or '--'}W")
        pc6.metric("CP / W' (無酸素容量)", f"{pdc.get('cp') or '--'} W", f"W': {pdc.get('w_prime_kj') or '--'} kJ")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.caption("📈 1年ベスト・パワーデュレーションカーブ (W/kg)")
            pdc_chart = build_pdc_chart(pdc, weight, ftp, target_ftp)
            st.altair_chart(pdc_chart, use_container_width=True, theme=None)
        with p_col2:
            st.caption("🎯 5軸ライダー能力バランス (100 = 国内トップアマ/富士ヒルプラチナ基準)")
            prof_chart = build_rider_profile_chart(pdc)
            if prof_chart:
                st.altair_chart(prof_chart, use_container_width=True, theme=None)

        st.markdown("##### ⚡ あなたの現在のパワートレーニングゾーン表 (FTP: " + f"`{ftp}W` / `{current_pwr}W/kg` 基準)")
        zones_df = pd.DataFrame([
            {"ゾーン": "Z1 アクティブリカバリー", "%FTP": "〜55%", "ワット範囲 (W)": f"〜 {round(ftp*0.55)} W", "PWR (W/kg)": f"〜 {round(ftp*0.55/weight, 2)}", "主な生理学的適応": "血流促進・疲労物質除去"},
            {"ゾーン": "Z2 エンデュランス (有酸素)", "%FTP": "56〜75%", "ワット範囲 (W)": f"{round(ftp*0.56)} 〜 {round(ftp*0.75)} W", "PWR (W/kg)": f"{round(ftp*0.56/weight, 2)} 〜 {round(ftp*0.75/weight, 2)}", "主な生理学的適応": "ミトコンドリア増殖・毛細血管密度・脂質代謝向上"},
            {"ゾーン": "Z3 テンポ", "%FTP": "76〜87%", "ワット範囲 (W)": f"{round(ftp*0.76)} 〜 {round(ftp*0.87)} W", "PWR (W/kg)": f"{round(ftp*0.76/weight, 2)} 〜 {round(ftp*0.87/weight, 2)}", "主な生理学的適応": "筋グリコーゲン貯蔵量増加・巡航持久力"},
            {"ゾーン": "🔥 SST (スイートスポット)", "%FTP": "88〜94%", "ワット範囲 (W)": f"{round(ftp*0.88)} 〜 {round(ftp*0.94)} W", "PWR (W/kg)": f"{round(ftp*0.88/weight, 2)} 〜 {round(ftp*0.94/weight, 2)}", "主な生理学的適応": "疲労を抑えつつ最大の有酸素・閾値引き上げ効果"},
            {"ゾーン": "Z4 乳酸閾値 (LT / FTP)", "%FTP": "95〜105%", "ワット範囲 (W)": f"{round(ftp*0.95)} 〜 {round(ftp*1.05)} W", "PWR (W/kg)": f"{round(ftp*0.95/weight, 2)} 〜 {round(ftp*1.05/weight, 2)}", "主な生理学的適応": "乳酸処理能力(MSSL)向上・60分巡航出力の強化"},
            {"ゾーン": "Z5 VO2max (最大酸素摂取量)", "%FTP": "106〜120%", "ワット範囲 (W)": f"{round(ftp*1.06)} 〜 {round(ftp*1.20)} W", "PWR (W/kg)": f"{round(ftp*1.06/weight, 2)} 〜 {round(ftp*1.20/weight, 2)}", "主な生理学的適応": "心拍出量(一回拍出量)の最大化・有酸素天井の突破"},
            {"ゾーン": "Z6 無酸素容量 (W' 強化)", "%FTP": "121〜150%", "ワット範囲 (W)": f"{round(ftp*1.21)} 〜 {round(ftp*1.50)} W", "PWR (W/kg)": f"{round(ftp*1.21/weight, 2)} 〜 {round(ftp*1.50/weight, 2)}", "主な生理学的適応": "解糖系出力・緩衝能力・アタック耐性向上"},
        ])
        st.dataframe(zones_df, hide_index=True, use_container_width=True)

# --- 6タブ構成 ---
tab_daily, tab_weekly, tab_debrief, tab_simulator, tab_chat, tab_history = st.tabs([
    "🎯 今日のメニュー設計",
    "📅 週間プラン一括生成",
    "🔬 ライド科学診断・振り返り",
    "🏔️ レース＆ヒルクライム物理計算",
    "💬 AIコーチ相談",
    "📊 ライド履歴 & EFトレンド"
])


# ==========================================
# Tab 1: 今日のAIワークアウト生成
# ==========================================
with tab_daily:
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns([1.2, 1.2, 1.5, 1.5])
    with col_opt1:
        avail_minutes = st.slider("練習時間 (分)", 30, 240, 75, 15)
    with col_opt2:
        indoor_mode = st.selectbox("環境", ["インドア (MyWhoosh/Zwift等)", "アウトドア (実走)"])
    with col_opt3:
        rpe_feeling = st.selectbox(
            "体調・コンディショニング方針",
            [
                "🤖 HRV・睡眠スコアからAI自動最適化（推奨）",
                "普通（予定通り）",
                "脚が重い・疲労感あり",
                "絶好調（限界突破したい）",
                "リカバリー・回復走希望",
            ]
        )
    with col_opt4:
        tactical_focus = st.selectbox(
            "🎯 戦術フォーカス（プリセット）",
            [
                "🤖 AIおまかせ（弱点＆フェーズ最適）",
                "🏔️ ヒルクライム巡航 (SST / L4 持続時間延長)",
                "🚀 VO2max 天井突破 (3〜5分 L5インターバル)",
                "⚡ 30/15秒 マイクロバースト (ロンネスタッド式VO2max)",
                "🔥 オーバーアンダー (乳酸処理・アタック対応)",
                "🍃 Z2ファットバーン＋高ケイデンススキル",
                "🧘 アクティブリカバリー (完全疲労抜き)",
            ]
        )

    today_note = st.text_input(
        "今日の補足メモ・体調変化・要望（任意）",
        placeholder="例: 昨日少し寝不足、右膝に少し違和感あり、今日はケイデンス95rpm以上を意識したい 等"
    )

    if st.button("⚡ 今日の最適ワークアウトを生成", type="primary", use_container_width=True):
        if is_sarashina_model(selected_model) and not sarashina_key and not gemini_key:
            st.error("サイドバーで Sarashina AI API Key を設定・保存してください。")
        elif not gemini_key and not sarashina_key:
            st.error("サイドバーで Gemini API Key または Sarashina AI API Key を設定・保存してください。")
        else:
            sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
            user_req = f"""
以下の条件で、本日実施すべき最適なワークアウトを1つ作成してください。
- 使用可能時間: {avail_minutes} 分
- 環境: {indoor_mode}
- 体調・メニュー方針: {rpe_feeling}
- 戦術フォーカス指定: {tactical_focus}
- 本日の補足メモ: {today_note or '特になし'}
- 外的要因・制限設定: {ext_status} / 強度上限: {ext_intensity_cap} / 詳細: {ext_detail or 'なし'}

【出力構成】
1. **本日のコンディション判定とメニューの科学的狙い**（最新のHRV・睡眠スコア・安静時心拍・TSB・RampRate、CP/W'やパワーカーブ弱点、外的要因や過去の記憶をどう統合したか）
2. **メニュー詳細とペダリング・補給アドバイス**（各セクションの目標ワット数 `{ftp}W` 基準の実数値、指定ケイデンス、走行中の糖質・水分摂取のポイント）
3. **Intervals.icu カレンダー登録用のワークアウトコードブロック**（必ず ```workout から始めて記述すること。完全休養を推奨する場合はコードブロックは省略可）
"""
            try:
                client_ai = genai.Client(api_key=gemini_key) if gemini_key else None
                st.session_state.generated_workout = ""
                with st.spinner(f"AIコーチ（{selected_model}）が生理学データ・記憶を踏まえてメニューを設計中..."):
                    response_text = st.write_stream(
                        stream_gemini_response(
                            client_ai, selected_model, user_req, sys_prompt,
                            sarashina_key=sarashina_key, gemini_key=gemini_key
                        )
                    )
                    st.session_state.generated_workout = response_text

                    m_code = re.search(r"```workout\n(.*?)```", response_text, re.DOTALL)
                    w_code_extracted = m_code.group(1).strip() if m_code else ""
                    st.session_state.daily_workout_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "title": f"{tactical_focus[:12]} ({avail_minutes}min)",
                        "condition": f"{rpe_feeling} / {today_note}",
                        "text": response_text,
                        "workout_code": w_code_extracted,
                    })
                    save_memory()
            except Exception as e:
                show_ai_error("AI生成エラー", e, selected_model)

    if st.session_state.generated_workout:
        match = re.search(r"```workout\n(.*?)```", st.session_state.generated_workout, re.DOTALL)
        if match:
            workout_code = match.group(1).strip()
            steps, df_chart = parse_intervals_workout(workout_code, ftp)
            w_stats = calculate_workout_stats(steps, ftp)

            if w_stats:
                st.markdown("##### 📊 生成ワークアウトの科学スペック & 推奨補給量")
                ws1, ws2, ws3, ws4, ws5, ws6 = st.columns(6)
                ws1.metric("⏱️ 総時間", f"{w_stats['total_min']} 分")
                ws2.metric("⚡ 推定 NP / 平均", f"{w_stats['np_watts']} W", f"平均 {w_stats['avg_watts']} W")
                ws3.metric("🔥 強度係数 (IF)", f"{w_stats['if']}")
                ws4.metric("📈 推定 TSS", f"{w_stats['tss']}")
                ws5.metric("🔋 消費仕事量", f"{w_stats['kj']} kJ", f"約 {w_stats['kj']} kcal")
                ws6.metric("🍬 推奨糖質 / 水分", f"糖質 {w_stats['carbs_g']}g", f"水分 {w_stats['water_ml']}ml")

            ch_w1, ch_w2 = st.columns([2.2, 1])
            with ch_w1:
                chart = build_workout_chart(df_chart, ftp)
                if chart:
                    st.altair_chart(chart, use_container_width=True, theme=None)
            with ch_w2:
                if w_stats and w_stats.get("zone_mins"):
                    tiz_chart = build_tiz_bar_chart(w_stats["zone_mins"])
                    if tiz_chart:
                        st.altair_chart(tiz_chart, use_container_width=True, theme=None)

            col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
            with col_p1:
                w_name = st.text_input("登録名", value=f"AI Coach ({avail_minutes}min)")
            with col_p2:
                w_date = st.date_input("登録日", value=datetime.now().date())
            with col_p3:
                zwo_xml = generate_zwo_xml(w_name, steps, ftp, description=f"AI Coach Plan for {w_date}")
                st.download_button(
                    label="📥 MyWhoosh / Zwift (.zwo) DL",
                    data=zwo_xml,
                    file_name=f"{w_name.replace(' ', '_')}_{w_date}.zwo",
                    mime="application/xml",
                    use_container_width=True,
                )

            if st.button("🚀 Intervals.icu カレンダーに登録", use_container_width=True):
                try:
                    client = IntervalsClient(athlete_id, intervals_key)
                    client.post_workout(w_name, workout_code, w_date.strftime("%Y-%m-%d"))
                    st.success(f"✅ {w_date} のカレンダーに『{w_name}』を登録しました！")
                except Exception as e:
                    st.error(f"登録エラー: {e}")

        st.divider()
        with st.container():
            st.markdown(st.session_state.generated_workout)

    if st.session_state.daily_workout_history:
        st.divider()
        with st.expander(f"📚 過去に生成したメニュー履歴 ({len(st.session_state.daily_workout_history)}件)"):
            for idx, item in enumerate(reversed(st.session_state.daily_workout_history)):
                st.markdown(f"**{item.get('date')} — {item.get('title')}** （{item.get('condition')}）")
                if st.button("このメニューを画面に呼び出す", key=f"restore_daily_{idx}"):
                    st.session_state.generated_workout = item.get("text")
                    save_memory()
                    st.rerun()
                st.divider()


# ==========================================
# Tab 2: 週間プラン一括生成 & カレンダー同期
# ==========================================
with tab_weekly:
    st.subheader("📅 1週間分の練習メニュー一括設計 ＆ カレンダー同期")
    st.caption("HRV・睡眠トレンド、外的要因（怪我・病気等）、CP/W' パワーカーブ弱点を統合した7日間の最適プランを作成・記憶します。")

    col_wk1, col_wk2 = st.columns(2)
    with col_wk1:
        start_plan_date = st.date_input("プラン開始日（月曜日など）", value=datetime.now().date())
    with col_wk2:
        weekly_note = st.text_input("今週の予定・体調・要望メモ", value="水曜と金曜は休養日にしたい、日曜は実走メイン")

    if st.button("🗓️ 7日分の週間メニューを一括生成", type="primary", use_container_width=True):
        if is_sarashina_model(selected_model) and not sarashina_key and not gemini_key:
            st.error("サイドバーで Sarashina AI API Key を設定・保存してください。")
        elif not gemini_key and not sarashina_key:
            st.error("サイドバーで Gemini API Key または Sarashina AI API Key を設定・保存してください。")
        else:
            sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
            date_list_str = ", ".join([
                (start_plan_date + timedelta(days=i)).strftime("%Y-%m-%d(%a)")
                for i in range(7)
            ])
            weekly_req = f"""
開始日 {start_plan_date} からの7日間（対象日: {date_list_str}）の週間トレーニングプランを作成してください。
- ユーザーの要望メモ: {weekly_note}
- 平日の上限時間: {weekday_hours}時間 / 休日の上限時間: {weekend_hours}時間
- 現在のHRV・睡眠スコアによるレディネス状態、TSB、および外的要因（{ext_status} / 上限:{ext_intensity_cap} / {ext_detail}）を厳密に考慮し、ポラライズドまたはピラミッド型のメリハリある7日間にすること。
- 練習日の `workout_code` には、Intervals.icuのカレンダーでそのままグラフ化される正確なワークアウト記法（Warmup / Main Set / Cooldown、`- 10m 90%` 等の形式）を記述すること。
"""
            try:
                with st.spinner(f"AIコーチ（{selected_model}）が7日間のメニューを構造化生成中..."):
                    client_ai = genai.Client(api_key=gemini_key) if gemini_key else None
                    response = generate_content_with_retry(
                        client_ai=client_ai,
                        model_name=selected_model,
                        contents=weekly_req,
                        config=types.GenerateContentConfig(
                            system_instruction=sys_prompt,
                            response_mime_type="application/json",
                            response_schema=WeeklyPlanResponse,
                        ),
                        sarashina_key=sarashina_key,
                        gemini_key=gemini_key,
                        sys_prompt=sys_prompt,
                    )
                    plan_obj = json.loads(response.text)
                    st.session_state.weekly_plan = plan_obj
                    st.session_state.weekly_plan_history.append({
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "start_date": str(start_plan_date),
                        "plan": plan_obj
                    })
                    save_memory()
            except Exception as e:
                show_ai_error("週間プラン生成エラー", e, selected_model)

    if st.session_state.weekly_plan:
        plan = st.session_state.weekly_plan
        total_wk_tss = sum(d.get("target_tss", 0) for d in plan.get("days", []))
        total_wk_min = sum(d.get("duration_min", 0) for d in plan.get("days", []))
        st.success(f"✨ 週間プラン表示中 — 週間合計予定時間: **{round(total_wk_min/60, 1)}時間 ({total_wk_min}分)** ／ 週間合計目標TSS: **{total_wk_tss} TSS**")
        st.markdown(f"**【今週のコーチング方針】**\n\n{plan['weekly_summary']}")

        for idx, day_item in enumerate(plan["days"]):
            badge = "🛌 休養日" if day_item["is_rest_day"] else f"🚴‍♂️ {day_item['duration_min']}分 / 目標TSS: {day_item['target_tss']}"
            with st.expander(f"**{day_item['date']} ({day_item['day_label']})** — {day_item['title']} 【{badge}】"):
                st.write(day_item["rationale"])
                if not day_item["is_rest_day"] and day_item["workout_code"]:
                    st.code(day_item["workout_code"], language="text")
                    steps_w, df_w = parse_intervals_workout(day_item["workout_code"], ftp)
                    ch_w = build_workout_chart(df_w, ftp)
                    if ch_w:
                        st.altair_chart(ch_w, use_container_width=True, theme=None)
                    zwo_day = generate_zwo_xml(day_item["title"], steps_w, ftp)
                    st.download_button(
                        f"📥 {day_item['date']} 用 (.zwo) DL",
                        data=zwo_day,
                        file_name=f"{day_item['date']}_{day_item['title'].replace(' ', '_')}.zwo",
                        mime="application/xml",
                        key=f"dl_wk_{idx}"
                    )

        st.divider()
        if st.button("🚀 7日分のワークアウトをIntervals.icuカレンダーへ一括登録する", type="primary", use_container_width=True):
            if not intervals_key:
                st.error("Intervals API Keyを設定してください。")
            else:
                client = IntervalsClient(athlete_id, intervals_key)
                posted_count = 0
                progress = st.progress(0)
                days_list = plan["days"]
                for idx, d in enumerate(days_list):
                    if not d["is_rest_day"] and d["workout_code"].strip():
                        try:
                            client.post_workout(
                                name=d["title"],
                                description=d["workout_code"].strip(),
                                target_date=d["date"]
                            )
                            posted_count += 1
                        except Exception as e:
                            st.warning(f"{d['date']} の登録に失敗: {e}")
                    progress.progress((idx + 1) / len(days_list))
                st.balloons()
                st.success(f"🎉 完了！合計 {posted_count} 件のワークアウトをIntervals.icuカレンダーへ一括登録しました！")


# ==========================================
# Tab 3: ライド科学診断・振り返り (Post-Ride Debrief & 記憶保存)
# ==========================================
with tab_debrief:
    st.subheader("🔬 直近ライドのスポーツ科学採点 ＆ AIコーチング診断")
    st.caption("NP/平均W比(VI)、有酸素効率(EF = NP/HR)、グリコーゲン消費量、Stravaセグメントを自動解析し、評価を長期記憶します。")

    if st.session_state.metrics and st.session_state.metrics.get("activities"):
        acts = st.session_state.metrics["activities"]
        ride_options = []
        for a in acts:
            aid = str(a.get("id", ""))
            reviewed_mark = "✅評価済 " if aid in st.session_state.ride_reviews else ""
            ride_options.append(
                f"{reviewed_mark}{a['日付']} - {a['ワークアウト名']} ({a['時間(分)']}分, TSS:{a['TSS']}, NP:{a['NP(W)']}W, EF:{a.get('EF', '--')})"
            )

        selected_ride_idx = st.selectbox("分析するライドを選択", range(len(ride_options)), format_func=lambda i: ride_options[i])
        selected_ride = acts[selected_ride_idx]
        act_id = str(selected_ride.get("id", ""))

        r_ready_score = st.session_state.metrics.get("wellness_readiness", {}).get("score", 75)
        sci_score, nutri_info, sci_notes = evaluate_ride_science_score(selected_ride, r_ready_score)

        col_r1, col_r2, col_r3, col_r4, col_r5, col_r6 = st.columns(6)
        col_r1.metric("🏅 ライド科学スコア", f"{sci_score} / 100", "自動採点")
        col_r2.metric("⏱️ 時間 / 距離", f"{selected_ride['時間(分)']} 分", f"{selected_ride.get('距離(km)', '--')} km")
        col_r3.metric("📈 TSS / IF", f"{selected_ride['TSS']}", f"IF: {selected_ride['IF']}")
        col_r4.metric("⚡ NP / 平均パワー", f"{selected_ride['NP(W)']} W", f"平均: {selected_ride['平均(W)']} W (VI: {selected_ride.get('VI', '--')})")
        col_r5.metric("💚 有酸素効率 (EF)", f"{selected_ride.get('EF', '--')}", f"平均心拍: {selected_ride['心拍(bpm)']} bpm")
        col_r6.metric("🍙 推奨リカバリー補給", f"糖質 {nutri_info['immediate_carbs_g']}g", f"タンパク質 {nutri_info['protein_g']}g / 総消費 {nutri_info['kj']}kJ")

        st.info(
            f"**🔬 自動テレメトリー所見:** {' ／ '.join(sci_notes)}\n\n"
            f"💡 **ゴールデンタイム栄養補給目安（ライド後30分以内）:** 推定グリコーゲン消費量 **{nutri_info['burned_carbs_g']}g** に対し、"
            f"まずは吸収の早い糖質 **{nutri_info['immediate_carbs_g']}g**（おにぎり約{max(1, round(nutri_info['immediate_carbs_g']/40))}〜{max(2, round(nutri_info['immediate_carbs_g']/35))}個分またはマルトデキストリン）＋ タンパク質 **{nutri_info['protein_g']}g** を摂取すると翌日のHRV回復が加速します。"
        )

        current_segments = []
        if act_id and intervals_key:
            cache_key = f"segments_{act_id}"
            if cache_key not in st.session_state:
                client = IntervalsClient(athlete_id, intervals_key)
                st.session_state[cache_key] = client.get_activity_segments(act_id)
            current_segments = st.session_state[cache_key]

        if current_segments:
            with st.expander(f"📍 このライドの Strava セグメントログ ({len(current_segments)}件)", expanded=True):
                df_seg = pd.DataFrame(current_segments)
                if weight > 0:
                    df_seg["W/kg"] = df_seg["平均パワー(W)"].apply(
                        lambda w: round(w / weight, 2) if isinstance(w, (int, float)) and w > 0 else "--"
                    )
                    cols = ["セグメント名", "タイム", "距離(km)", "平均パワー(W)", "W/kg", "最大パワー(W)", "平均心拍(bpm)"]
                    df_seg = df_seg[[c for c in cols if c in df_seg.columns]]
                st.dataframe(df_seg, use_container_width=True)

        saved_review_obj = st.session_state.ride_reviews.get(act_id)
        default_memo = saved_review_obj.get("memo", "") if saved_review_obj else ""

        user_ride_feedback = st.text_input(
            "ライド時の体感メモ・補足（任意・記憶されます）",
            value=default_memo,
            placeholder="例: 後半脚がタレた、補給が足りなかった、向かい風が強かった、ケイデンスを意識した等"
        )

        btn_label = "🔄 このライドを再評価して記憶を更新する" if saved_review_obj else "🔍 このライドをAIコーチに詳しくレビューしてもらう"
        if st.button(btn_label, type="primary", use_container_width=True):
            if is_sarashina_model(selected_model) and not sarashina_key and not gemini_key:
                st.error("サイドバーで Sarashina AI API Key を設定・保存してください。")
            elif not gemini_key and not sarashina_key:
                st.error("サイドバーで Gemini API Key または Sarashina AI API Key を設定・保存してください。")
            else:
                sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)

                seg_summary = "セグメントなし（平坦またはインドア）"
                if current_segments:
                    seg_summary = "\n".join([
                        f"- {s['セグメント名']}: タイム {s['タイム']}, 距離 {s['距離(km)']}km, 平均 {s['平均パワー(W)']}W, 最大 {s['最大パワー(W)']}W, 平均心拍 {s['平均心拍(bpm)']}bpm"
                        for s in current_segments[:8]
                    ])

                debrief_req = f"""
以下のライド実績について、プロのサイクリングコーチとして多角的な分析とアドバイスを行ってください。

【対象ライドデータ】
- 日付: {selected_ride['日付']} / ワークアウト名: {selected_ride['ワークアウト名']}
- 時間: {selected_ride['時間(分)']} 分 / 距離: {selected_ride.get('距離(km)', '--')} km / 仕事量: {selected_ride.get('仕事量(kJ)', '--')} kJ
- TSS: {selected_ride['TSS']} / IF: {selected_ride['IF']} / 自動科学スコア: {sci_score}/100点
- NP: {selected_ride['NP(W)']} W / 平均パワー: {selected_ride['平均(W)']} W (FTP: {ftp}W / 変動指数VI: {selected_ride.get('VI', '--')})
- 平均心拍: {selected_ride['心拍(bpm)']} bpm / 有酸素効率(EF): {selected_ride.get('EF', '--')} W/bpm
- アスリートの体感メモ: {user_ride_feedback or '特になし'}

【Strava 通過セグメント実績】
{seg_summary}

【分析・フィードバック項目】
1. **総括と負荷評価**: 現在のHRV・睡眠コンディションやTSB、外的要因（怪我・病気等）に照らし合わせたTSS/IFの適切さ
2. **ペーシング(VI)と有酸素効率(EF)の深掘り**: NPと平均Wの乖離（VI={selected_ride.get('VI', '--')}）、パワーに対する心拍効率（EF={selected_ride.get('EF', '--')}、過去のライドとの比較）
3. **Stravaセグメント＆CP/W'分析**: 登坂やスプリント各セグメントでの踏み込み・ペーシング・FTP/CP比評価
4. **今日のGoodポイント**: 褒めるべき点や成長が見える数値
5. **リカバリー＆次回への戦術アドバイス**: 具体的な栄養補給（糖質{nutri_info['immediate_carbs_g']}g＋タンパク質{nutri_info['protein_g']}g）、睡眠・HRV回復に向けたケア、明日以降のメニュー調整
"""
                try:
                    client_ai = genai.Client(api_key=gemini_key) if gemini_key else None
                    st.divider()
                    st.subheader("📋 AIコーチからのフィードバック")
                    with st.spinner(f"AIコーチ（{selected_model}）がライドデータを科学診断中..."):
                        review_text = st.write_stream(
                            stream_gemini_response(
                                client_ai, selected_model, debrief_req, sys_prompt,
                                sarashina_key=sarashina_key, gemini_key=gemini_key
                            )
                        )
                        st.session_state.ride_reviews[act_id] = {
                            "date": selected_ride["日付"],
                            "name": selected_ride["ワークアウト名"],
                            "memo": user_ride_feedback,
                            "review": review_text,
                            "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        save_memory()
                except Exception as e:
                    show_ai_error("ライド診断エラー", e, selected_model)
        elif saved_review_obj:
            st.divider()
            st.subheader(f"📋 保存済みのAIコーチ評価（記録日時: {saved_review_obj.get('reviewed_at', '--')}）")
            if saved_review_obj.get("memo"):
                st.caption(f"📝 記録された体感メモ: {saved_review_obj['memo']}")
            st.markdown(saved_review_obj["review"])
    else:
        st.info("ライド履歴がありません。上部の「🔄 最新データ同期」を押してデータを取得してください。")


# ==========================================
# Tab 4: レース＆ヒルクライム物理シミュレーター・補給戦略
# ==========================================
with tab_simulator:
    st.subheader("🏔️ ヒルクライム＆目標レース 物理タイムシミュレーター ＆ ドラフティング・補給戦略")
    st.caption("ニュートン力学（重力・転がり抵抗・標高別空気密度・ドラフティング空気抵抗軽減・駆動ロス）と高地低酸素モデルを用い、予想タイムや「ドラフティング効果」「軽量化 vs パワー向上」を精密計算します。")

    climb_presets = {
        "🗻 Mt.富士ヒルクライム (計測24.0km / 標高差1,255m / 平均勾配5.2% / 平均標高1,750m)": {"dist": 24.0, "elev": 1255, "alt": 1750},
        "🏔️ 乗鞍ヒルクライム (20.5km / 標高差1,260m / 平均勾配6.1% / 平均標高2,100m)": {"dist": 20.5, "elev": 1260, "alt": 2100},
        "⛰️ 赤城山ヒルクライム (18.4km / 標高差1,213m / 平均勾配6.6% / 平均標高1,000m)": {"dist": 18.4, "elev": 1213, "alt": 1000},
        "🌲 ヤビツ峠・表ヤビツ (11.8km / 標高差660m / 平均勾配5.6% / 平均標高450m)": {"dist": 11.8, "elev": 660, "alt": 450},
        "🐸 筑波山・不動峠 (3.8km / 標高差265m / 平均勾配7.0% / 平均標高280m)": {"dist": 3.8, "elev": 265, "alt": 280},
        "🛠️ カスタムコース（距離・獲得標高を自由入力）": {"dist": 15.0, "elev": 800, "alt": 800},
    }

    draft_presets = {
        "👤 単独走行（ソロ・空気抵抗軽減 0%）": 0,
        "🔄 2〜3人先頭交代ローテーション（平均 12%軽減）": 12,
        "🚴‍♂️🚴‍♂️ 前走者の真後ろ・ツキイチ追走（22%軽減）": 22,
        "🚄 中規模トレイン・5〜8人パック中盤（32%軽減）": 32,
        "🐝 大集団ペロトン内部・選抜/ゴールドトレイン（40%軽減）": 40,
    }

    top_p1, top_p2 = st.columns([1.4, 1.2])
    with top_p1:
        selected_preset = st.selectbox("📍 コースプリセット選択", list(climb_presets.keys()))
        preset_vals = climb_presets[selected_preset]
    with top_p2:
        selected_draft_label = st.selectbox("💨 ドラフティング・集団位置プリセット", list(draft_presets.keys()), index=0)
        default_draft_pct = draft_presets[selected_draft_label]

    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        sim_dist = st.number_input("距離 (km)", 1.0, 200.0, float(preset_vals["dist"]), 0.5)
        sim_elev = st.number_input("獲得標高 (m)", 50, 5000, int(preset_vals["elev"]), 25)
    with sim_c2:
        sim_alt = st.number_input("平均標高 (m) ※空気密度＆高地減衰", 0, 3500, int(preset_vals["alt"]), 100)
        sim_bike_kg = st.number_input("機材総重量 (バイク+装備+ボトル kg)", 5.5, 18.0, 7.8, 0.1)
    with sim_c3:
        sim_rider_kg = st.number_input("シミュレーション体重 (kg)", 35.0, 120.0, float(weight), 0.5)
        sim_intensity_pct = st.slider("レース巡航強度 (%FTP)", 75, 110, 95, 1)
    with sim_c4:
        sim_ftp_base = st.number_input("基準FTP (W)", 100, 500, int(ftp), 5)
        sim_draft_pct = st.slider("💨 空気抵抗軽減率 (ドラフティング %)", 0, 45, int(default_draft_pct), 1)
        apply_alt_penalty = st.checkbox("高地（低酸素）による有酸素出力減衰を適用", value=(preset_vals["alt"] >= 1200))

    raw_target_watts = round(sim_ftp_base * (sim_intensity_pct / 100.0))
    prelim = solve_hillclimb_physics(sim_dist, sim_elev, sim_rider_kg, sim_bike_kg, raw_target_watts, sim_alt, draft_saving_pct=0.0)
    alt_drop_pct = prelim["altitude_penalty_pct"] if apply_alt_penalty else 0.0
    effective_watts = round(raw_target_watts * (1.0 - alt_drop_pct / 100.0))

    # 単独走行(0%) と 現在のドラフティング設定(sim_draft_pct%) の両方を計算
    solo_res = solve_hillclimb_physics(sim_dist, sim_elev, sim_rider_kg, sim_bike_kg, effective_watts, sim_alt, draft_saving_pct=0.0)
    sim_res = solve_hillclimb_physics(sim_dist, sim_elev, sim_rider_kg, sim_bike_kg, effective_watts, sim_alt, draft_saving_pct=sim_draft_pct)

    def _fmt_time(sec_float: float) -> str:
        t_s = int(round(sec_float))
        hh, rem = divmod(t_s, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh}:{mm:02d}:{ss:02d}" if hh > 0 else f"{mm}分{ss:02d}秒"

    def _fmt_diff_sec(diff_s: int) -> str:
        if diff_s == 0:
            return "±0秒 (単独基準)"
        sign = "-" if diff_s < 0 else "+"
        abs_s = abs(diff_s)
        mm, ss = divmod(abs_s, 60)
        return f"{sign}{mm}分{ss:02d}秒 ({diff_s:+}秒)" if mm > 0 else f"{diff_s:+}秒"

    total_sec = int(round(sim_res["time_sec"]))
    solo_sec = int(round(solo_res["time_sec"]))
    draft_time_diff_sec = total_sec - solo_sec
    time_formatted = _fmt_time(sim_res["time_sec"])

    # 単独走行と同じタイム(速度 solo_res['v_ms'])で走るために必要なパワーと温存ワット数
    solo_v = solo_res["v_ms"]
    req_watts_at_solo_speed = round(
        ((solo_res["f_grav_roll"] + 0.5 * solo_res["rho"] * sim_res["eff_cda"] * (solo_v ** 2)) * solo_v) / (1.0 - 0.025)
    )
    saved_watts_vs_solo = max(0, effective_watts - req_watts_at_solo_speed)

    # 富士ヒルリング判定
    ring_badge = ""
    if "富士ヒル" in selected_preset:
        if total_sec < 3600:
            ring_badge = "💎 プラチナリング圏内 (<60分)"
        elif total_sec < 3900:
            ring_badge = "🥇 ゴールドリング圏内 (<65分)"
        elif total_sec < 4500:
            ring_badge = "🥈 シルバーリング圏内 (<75分)"
        elif total_sec < 5400:
            ring_badge = "🥉 ブロンズリング圏内 (<90分)"
        else:
            ring_badge = "🟦 ブルーリング圏内"

    res_c1, res_c2, res_c3, res_c4, res_c5 = st.columns(5)
    res_c1.metric("🏁 予想ゴールタイム", time_formatted, ring_badge or f"平均勾配 {sim_res['grade_pct']}%")
    res_c2.metric(
        "💨 ドラフティング恩恵",
        f"{abs(draft_time_diff_sec)}秒 短縮" if draft_time_diff_sec < 0 else "単独走行 (0%)",
        f"単独比 {_fmt_diff_sec(draft_time_diff_sec)} / {saved_watts_vs_solo}W温存相当" if sim_draft_pct > 0 else f"実効CdA: {sim_res['eff_cda']}"
    )
    res_c3.metric("⚡ 実効巡航パワー", f"{effective_watts} W", f"平地換算 {raw_target_watts}W (高地 -{alt_drop_pct}%)" if alt_drop_pct > 0 else f"{sim_intensity_pct}% FTP")
    res_c4.metric("⚖️ 実効 PWR", f"{round(effective_watts / sim_rider_kg, 2)} W/kg", f"総重量 {round(sim_rider_kg + sim_bike_kg, 1)} kg")
    res_c5.metric("🚀 平均速度 / VAM", f"{sim_res['speed_kmh']} km/h", f"VAM: {sim_res['vam']} m/h")

    st.markdown("##### 💨 ドラフティング（集団・トレイン活用）によるタイム差 ＆ 温存パワー比較表")
    st.caption(
        f"同じ実効パワー（**{effective_watts}W**）で走った場合に**どれだけタイムが縮まるか**、および単独タイム（**{_fmt_time(solo_res['time_sec'])}**）と同じペースで登る場合に**何ワット脚を温存できるか**の比較です。"
    )
    draft_table_rows = []
    for d_label, d_pct in [
        ("👤 単独走行（ソロ・基準）", 0),
        ("🔄 2〜3人ローテーション（先頭交代）", 12),
        ("🚴‍♂️🚴‍♂️ 前走者の真後ろ（ツキイチ追走）", 22),
        ("🚄 中規模トレイン中盤（5〜8人パック）", 32),
        ("🐝 大集団ペロトン内部（選抜・大パック）", 40),
    ]:
        r_d = solve_hillclimb_physics(sim_dist, sim_elev, sim_rider_kg, sim_bike_kg, effective_watts, sim_alt, draft_saving_pct=d_pct)
        diff_s = int(round(r_d["time_sec"])) - solo_sec
        req_w_same_speed = round(
            ((solo_res["f_grav_roll"] + 0.5 * solo_res["rho"] * r_d["eff_cda"] * (solo_v ** 2)) * solo_v) / (1.0 - 0.025)
        )
        save_w = max(0, effective_watts - req_w_same_speed)
        draft_table_rows.append({
            "走行形態・パック位置": d_label,
            "空気抵抗軽減率": f"-{d_pct}% (実効CdA: {r_d['eff_cda']})",
            f"{effective_watts}W での予想タイム": _fmt_time(r_d["time_sec"]),
            "単独比タイム短縮": _fmt_diff_sec(diff_s),
            "平均速度": f"{r_d['speed_kmh']} km/h",
            "単独タイム維持の必要出力 (温存W)": f"{req_w_same_speed} W (-{save_w}W 温存)" if d_pct > 0 else f"{effective_watts} W (基準)",
        })
    st.dataframe(pd.DataFrame(draft_table_rows), hide_index=True, use_container_width=True)

    st.markdown("##### 📊 「軽量化（減量・機材軽量化）」vs「パワー向上」によるタイム短縮効果マトリクス")
    st.caption(f"現在の設定（ドラフティング -{sim_draft_pct}% / 予想タイム {time_formatted}）を基準に、重量を減らした場合とパワーを上げた場合の比較表です。")
    matrix_rows = []
    for w_delta in [0.0, -0.5, -1.0, -1.5, -2.0]:
        row_dict = {"重量変化 (体重+機材)": f"{w_delta:+.1f} kg ({round(sim_rider_kg + w_delta, 1)}kg)" if w_delta != 0 else f"±0.0 kg (現在: {sim_rider_kg}kg)"}
        for p_delta in [0, 5, 10, 15, 20]:
            r_m = solve_hillclimb_physics(
                sim_dist, sim_elev, max(35.0, sim_rider_kg + w_delta), sim_bike_kg,
                effective_watts + p_delta, sim_alt, draft_saving_pct=sim_draft_pct
            )
            sec_diff = int(round(r_m["time_sec"] - sim_res["time_sec"]))
            t_s = int(round(r_m["time_sec"]))
            mm, ss = divmod(t_s, 60)
            diff_str = f"({sec_diff:+}秒)" if sec_diff != 0 else "(基準)"
            row_dict[f"パワー +{p_delta}W ({effective_watts+p_delta}W)"] = f"{mm}分{ss:02d}秒 {diff_str}"
        matrix_rows.append(row_dict)
    st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

    # レース当日の栄養・カフェイン・ペーシング戦略カード
    race_hours = sim_res["time_sec"] / 3600.0
    race_kj = round(effective_watts * sim_res["time_sec"] / 1000.0)
    caffeine_mg = round(sim_rider_kg * 3.5)
    ingame_carbs_g = round(65 * race_hours) if race_hours >= 0.9 else 30

    with st.expander("🧪 スポーツ科学に基づくレース当日ペーシング＆補給・エルゴジェニック戦略", expanded=True):
        st.markdown(f"""
- **⏱️ スタート前ローディング**:
  - **スタート3時間前**: 消化の良い糖質 **{round(sim_rider_kg * 2.0)}g**（白米・餅・うどん・バナナ等、脂質と食物繊維は控える）
  - **スタート45〜60分前**: **カフェイン {caffeine_mg} mg**（体重1kgあたり約3.5mg：RPE軽減＆中枢神経動員向上）＋ エネルギージェル1本（糖質30g）
- **🚴‍♂️ レース中の補給量（予想タイム {time_formatted} / 推定消費 {race_kj} kJ）**:
  - **走行中の目標糖質摂取**: 合計 **{ingame_carbs_g} g**（マルトデキストリン＋フルクトース配合ドリンクまたはジェルを20〜25分ごとに分割摂取）
- **🎯 科学的ペーシング＆ドラフティング鉄則（無酸素容量 W' = `{st.session_state.metrics['pdc'].get('w_prime_kj', 15.0) if st.session_state.metrics else 15.0} kJ` を温存する走り方）**:
  - **序盤（最初の10分間）**: アドレナリンでオーバーペースになりやすいため、上限を **{round(effective_watts * 1.02)}W** 以下に厳格ロック。同じ脚力帯のトレインを見つけたらツキイチ〜中盤に入り、単独比 **{saved_watts_vs_solo if sim_draft_pct > 0 else 12}W前後** の空気抵抗節約を狙いましょう。
  - **中盤（平坦・緩斜面区間）**: 勾配が緩む区間（富士ヒルの1合目下・奥庭平坦など）は速度が上がり空気抵抗が2乗で増えるため、絶対に単独で風を受けずパック内で脚を休めること。
  - **終盤（ラスト3km〜ゴール）**: パック内で温存した無酸素容量（W'）を解放し、徐々に **{round(effective_watts * 1.08)}W → 全開スプリント** へビルドアップ！
        """)


# ==========================================
# Tab 5: AIコーチ・チャット相談（永続記憶付き・入力欄最下部固定）
# ==========================================
with tab_chat:
    col_ch1, col_ch2 = st.columns([4, 1])
    with col_ch1:
        st.caption("💬 過去の会話履歴・ライド評価・HRV/睡眠状態・CP/W'・怪我や病気の設定をすべて記憶した状態で相談できます。（入力欄は常に画面最下部に固定されています）")
    with col_ch2:
        if st.button("🗑️ 会話履歴をクリア", use_container_width=True):
            st.session_state.messages = []
            save_memory()
            st.rerun()

    # メッセージ表示用コンテナを st.chat_input の上に配置することで、新規発言時も常に入力欄が最下部に位置する
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("例: 今日のHRVと睡眠スコア的に走っても大丈夫？ / 富士ヒルで65分を切るためにあと何を伸ばすべき？"):
        if is_sarashina_model(selected_model) and not sarashina_key and not gemini_key:
            st.error("サイドバーで Sarashina AI API Key を設定・保存してください。")
        elif not gemini_key and not sarashina_key:
            st.error("サイドバーで Gemini API Key または Sarashina AI API Key を設定・保存してください。")
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            save_memory()
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
                        client_ai = genai.Client(api_key=gemini_key) if gemini_key else None
                        history_contents = [
                            types.Content(
                                role="user" if m["role"] == "user" else "model",
                                parts=[types.Part.from_text(text=m["content"])]
                            )
                            for m in st.session_state.messages[-20:]
                        ]
                        reply = st.write_stream(
                            stream_gemini_response(
                                client_ai, selected_model, history_contents, sys_prompt,
                                sarashina_key=sarashina_key, gemini_key=gemini_key
                            )
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                        save_memory()
                    except Exception as e:
                        show_ai_error("チャット応答エラー", e, selected_model)

    with chat_container:
        st.markdown('<div style="height: 80px;"></div>', unsafe_allow_html=True)


# ==========================================
# Tab 6: 直近40日のアクティビティ履歴 & 有酸素効率(EF)トレンド
# ==========================================
with tab_history:
    if st.session_state.metrics and not st.session_state.metrics["activities_df"].empty:
        acts_raw_list = st.session_state.metrics.get("activities", [])
        ef_chart = build_ef_trend_chart(acts_raw_list)
        if ef_chart:
            st.subheader("📈 直近40日間の有酸素効率 (EF = NP / 平均心拍) ＆ TSS 推移")
            st.caption("緑の実線（EF）が右肩上がりであれば、同じ心拍数でより高いパワーを出せる（有酸素エンジンが強化されている）証拠です。")
            st.altair_chart(ef_chart, use_container_width=True, theme=None)
            st.divider()

        st.subheader("📋 直近40日間のライド履歴一覧（EF・VI・仕事量付き）")
        st.dataframe(st.session_state.metrics["activities_df"], use_container_width=True)
    else:
        st.info("データがありません。上部の「🔄 最新データ同期」を押してください。")

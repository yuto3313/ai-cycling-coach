import streamlit as st
import requests
import pandas as pd
import json
import os
import re
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
# 0. 設定・キャッシュ・AI記憶の自動保存マネージャー（マルチユーザー＆オーナー自動ログイン対応）
# ==========================================
DEFAULT_CONFIG = {
    "athlete_id": "",
    "athlete_name": "",
    "intervals_key": "",
    "gemini_key": "",
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
    # 公開設定（Trueにすると通常URLアクセス時はゲストAPI入力画面をデフォルトにする）
    "public_default_guest": False,
}

DEFAULT_MEMORY = {
    "messages": [],               # チャット履歴 [{"role": "user"|"assistant", "content": "...", "timestamp": "..."}]
    "generated_workout": None,    # 最新の「今日のメニュー」テキスト
    "daily_workout_history": [],  # 過去のメニュー生成履歴 [{"date": "...", "title": "...", "condition": "...", "text": "...", "workout_code": "..."}]
    "weekly_plan": None,          # 最新の週間プラン dict
    "weekly_plan_history": [],    # 過去の週間プラン履歴 [{"created_at": "...", "plan": {...}}]
    "ride_reviews": {},           # アクティビティID別の評価履歴 {act_id: {"date": "...", "name": "...", "memo": "...", "review": "...", "reviewed_at": "..."}}
}


def compute_user_uid(intervals_key: str, athlete_id: str = "") -> str:
    """他ユーザーのAPIキーから固有の識別ハッシュ（16文字）を生成（本人のみアクセス可能）"""
    if not intervals_key or not intervals_key.strip():
        return ""
    raw = f"icu_user:{intervals_key.strip()}"
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
        for k in ["athlete_id", "intervals_key", "gemini_key"]:
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
                # セッション内で未入力の項目があればファイルから復元
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
            "weight": metrics.get("weight"),
            "wellness_readiness": metrics.get("wellness_readiness", {}),
            "wellness_history": metrics.get("wellness_history", []),
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

                m, ss = divmod(dur, 60)
                time_str = f"{m}:{ss:02d}"

                results.append({
                    "セグメント名": s.get("name", "名称不明"),
                    "タイム": time_str,
                    "距離(km)": round(dist_m / 1000, 2),
                    "平均パワー(W)": avg_w,
                    "最大パワー(W)": max_w,
                    "平均心拍(bpm)": avg_hr if avg_hr > 0 else "--",
                })
            return results
        except Exception:
            return []


# ==========================================
# 3. ワークアウト解析・ZWO生成・Altairチャート
# ==========================================
def parse_intervals_workout(workout_text: str, ftp: int):
    lines = workout_text.strip().split("\n")
    steps = []
    current_section = "Main"
    repeat_count = 1
    repeat_block = []
    in_repeat = False

    time_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(m|s|h|min|sec)", re.IGNORECASE)
    pct_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
    watt_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*w\b", re.IGNORECASE)
    cadence_pattern = re.compile(r"(\d+)\s*rpm", re.IGNORECASE)

    def flush_repeat():
        nonlocal in_repeat, repeat_block, repeat_count
        if in_repeat and repeat_block:
            for _ in range(repeat_count):
                for b in repeat_block:
                    steps.append(dict(b))
            repeat_block = []
            in_repeat = False

    def parse_line(line_str):
        t_match = time_pattern.search(line_str)
        if not t_match:
            return None
        val, unit = float(t_match.group(1)), t_match.group(2).lower()
        if unit in ["s", "sec"]:
            duration_sec = int(val)
        elif unit in ["h"]:
            duration_sec = int(val * 3600)
        else:
            duration_sec = int(val * 60)

        p_pct = None
        pct_match = pct_pattern.search(line_str)
        if pct_match:
            p_pct = float(pct_match.group(1)) / 100.0
        else:
            w_match = watt_pattern.search(line_str)
            if w_match and ftp > 0:
                p_pct = float(w_match.group(1)) / ftp

        if p_pct is None:
            p_pct = 0.50

        cad = None
        cad_match = cadence_pattern.search(line_str)
        if cad_match:
            cad = int(cad_match.group(1))

        return {"duration": duration_sec, "power": round(p_pct, 3), "cadence": cad}

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue

        if re.search(r"^warmup", cleaned, re.IGNORECASE):
            flush_repeat()
            current_section = "Warmup"
            continue
        elif re.search(r"^cooldown", cleaned, re.IGNORECASE):
            flush_repeat()
            current_section = "Cooldown"
            continue
        elif match_rep := re.search(r"(\d+)\s*x", cleaned, re.IGNORECASE):
            flush_repeat()
            current_section = "Main"
            repeat_count = int(match_rep.group(1))
            in_repeat = True
            repeat_block = []
            continue

        parsed = parse_line(cleaned)
        if parsed:
            parsed["section"] = current_section
            if in_repeat:
                repeat_block.append(parsed)
            else:
                steps.append(parsed)

    flush_repeat()

    timeline = []
    current_time_sec = 0

    def get_zone(p):
        if p < 0.56: return ("Z1 回復", "#60a5fa")
        elif p < 0.76: return ("Z2 耐久", "#34d399")
        elif p < 0.91: return ("Z3 テンポ", "#fbbf24")
        elif p < 1.06: return ("Z4 閾値/SST", "#f97316")
        elif p < 1.21: return ("Z5 VO2max", "#ef4444")
        else: return ("Z6 無酸素", "#a855f7")

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
            "power_pct": int(round(p * 100)),
            "power_w": watts,
            "zone": zone_label,
            "color": color,
            "cadence": str(s["cadence"]) if s["cadence"] else "--",
            "section": s["section"]
        })
        current_time_sec += dur

    return steps, pd.DataFrame(timeline)


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


def build_workout_chart(df: pd.DataFrame, ftp: int):
    if df.empty:
        return None
    
    max_w = max(ftp * 1.25, df["power_w"].max() * 1.1)

    chart = alt.Chart(df).mark_rect(stroke="#1e293b", strokeWidth=0.5).encode(
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
        color="#ffffff", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(
        y="ftp:Q",
        tooltip=[alt.Tooltip("label:N", title="基準ライン")]
    )

    return (chart + ftp_line).properties(height=230)


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

    line = base.mark_line(color="#38bdf8", strokeWidth=3, point=True)
    area = base.mark_area(opacity=0.2, color="#38bdf8")

    ref_df = pd.DataFrame([
        {"level": f"目標FTP ({tar_pwr} W/kg)", "val": tar_pwr, "color": "#f87171"},
        {"level": f"現在FTP ({cur_pwr} W/kg)", "val": cur_pwr, "color": "#4ade80"},
    ])
    rule = alt.Chart(ref_df).mark_rule(strokeDash=[4, 4], strokeWidth=2).encode(
        y="val:Q",
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("level:N", title="基準")]
    )

    return (area + line + rule).properties(height=240)


def build_wellness_chart(wellness_history: list):
    """直近のHRVと睡眠スコアの推移チャート"""
    if not wellness_history:
        return None
    df_w = pd.DataFrame(wellness_history)
    if df_w.empty or ("hrv" not in df_w.columns and "sleep_score" not in df_w.columns):
        return None

    base = alt.Chart(df_w).encode(
        x=alt.X("date:N", title="日付")
    )

    hrv_line = base.mark_line(color="#a855f7", strokeWidth=2.5, point=True).encode(
        y=alt.Y("hrv:Q", title="HRV (ms) / 睡眠スコア"),
        tooltip=[
            alt.Tooltip("date:N", title="日付"),
            alt.Tooltip("hrv:Q", title="HRV (ms)"),
            alt.Tooltip("sleep_score:Q", title="睡眠スコア"),
            alt.Tooltip("sleep_hours:Q", title="睡眠時間 (h)"),
            alt.Tooltip("rhr:Q", title="安静時心拍 (bpm)"),
        ]
    )

    sleep_line = base.mark_line(color="#38bdf8", strokeDash=[4, 2], strokeWidth=2, point=True).encode(
        y=alt.Y("sleep_score:Q"),
        tooltip=[
            alt.Tooltip("date:N", title="日付"),
            alt.Tooltip("sleep_score:Q", title="睡眠スコア"),
            alt.Tooltip("sleep_hours:Q", title="睡眠時間 (h)"),
            alt.Tooltip("hrv:Q", title="HRV (ms)"),
        ]
    )

    return (hrv_line + sleep_line).properties(height=200)


# ==========================================
# 4. データ解析 & HRV/睡眠レディネス & パワーカーブ診断
# ==========================================
def analyze_wellness_readiness(wellness_raw: list, latest_tsb: float) -> tuple[dict, list]:
    """
    Intervals.icu の wellness データから HRV・睡眠スコア・睡眠時間・安静時心拍を解析し、
    本日のレディネススコア(0-100)と推奨メニュー強度を算出する。
    """
    history = []
    hrv_vals = []
    rhr_vals = []
    sleep_score_vals = []
    sleep_hour_vals = []

    for w in wellness_raw:
        d_str = w.get("id", "")
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
                "date": d_str[5:] if len(d_str) >= 10 else d_str,
                "full_date": d_str,
                "hrv": round(float(hrv), 1) if hrv is not None else None,
                "rhr": int(rhr) if rhr is not None else None,
                "sleep_score": int(round(float(s_score))) if s_score is not None else None,
                "sleep_hours": round(s_hours, 1) if s_hours is not None else None,
                "sleep_quality": s_qual,
            })

    # 最新の有効レコードを取得（当日または前日）
    latest_rec = history[-1] if history else {}
    latest_hrv = latest_rec.get("hrv")
    latest_rhr = latest_rec.get("rhr")
    latest_sleep_score = latest_rec.get("sleep_score")
    latest_sleep_hours = latest_rec.get("sleep_hours")
    latest_date = latest_rec.get("full_date", "--")

    # 過去40日のベースライン（中央値・平均値）
    hrv_base = round(statistics.median(hrv_vals), 1) if len(hrv_vals) >= 3 else None
    rhr_base = round(statistics.median(rhr_vals), 1) if len(rhr_vals) >= 3 else None
    sleep_score_base = round(statistics.mean(sleep_score_vals), 1) if len(sleep_score_vals) >= 3 else None

    # レディネススコア計算 (基準75点)
    score = 75
    reasons = []

    # 1. HRV 評価
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

    # 2. 睡眠スコア & 睡眠時間評価
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

    # 3. 安静時心拍 (RHR) 評価
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

    # 4. TSB 加味
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
    return readiness_info, history[-14:]  # 直近14日分の推移をチャート用に返す


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

    readiness_info, wellness_hist = analyze_wellness_readiness(wellness_raw, latest_tsb)

    act_list = []
    eftp_from_act = None
    for act in activities_raw:
        if not act.get("type") or "Ride" not in act.get("type", ""):
            continue
        if act.get("icu_pm_ftp"):
            eftp_from_act = act.get("icu_pm_ftp")
        moving_time_min = round((act.get("moving_time") or 0) / 60)
        act_list.append({
            "id": act.get("id"),
            "日付": act.get("start_date_local", "")[:10],
            "ワークアウト名": act.get("name", "Ride"),
            "時間(分)": moving_time_min,
            "距離(km)": round((act.get("distance") or 0) / 1000, 1),
            "TSS": round(act.get("icu_training_load") or 0),
            "NP(W)": round(act.get("icu_weighted_avg_watts") or act.get("weighted_average_watts") or 0),
            "平均(W)": round(act.get("icu_average_watts") or act.get("average_watts") or 0),
            "IF": round((act.get("icu_intensity") or 0) / 100, 2) if act.get("icu_intensity") else 0.0,
            "心拍(bpm)": round(act.get("average_heartrate") or 0),
        })

    pdc_info = extract_pdc_metrics(pdc_raw, weight, ftp)
    if not pdc_info["eftp"] and eftp_from_act:
        pdc_info["eftp"] = eftp_from_act

    return {
        "ctl": latest_ctl,
        "atl": latest_atl,
        "tsb": latest_tsb,
        "weight": latest_weight,
        "wellness_readiness": readiness_info,
        "wellness_history": wellness_hist,
        "activities": act_list,
        "activities_df": pd.DataFrame(act_list).drop(columns=["id"], errors="ignore") if act_list else pd.DataFrame(),
        "pdc": pdc_info
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

    # 1. 過去のライド評価履歴（直近3件）
    reviews = st.session_state.get("ride_reviews", {})
    if reviews:
        rev_items = list(reviews.values())[-3:]
        rev_lines = []
        for r in rev_items:
            short_rev = (r.get("review", "")[:250] + "...") if len(r.get("review", "")) > 250 else r.get("review", "")
            rev_lines.append(f"- [{r.get('date')}] {r.get('name')} (体感メモ: {r.get('memo') or 'なし'}): {short_rev}")
        parts.append("【過去のライド振り返り・評価記憶（直近3件）】\n" + "\n".join(rev_lines))

    # 2. 直近生成した日次メニュー履歴（直近3件）
    w_hist = st.session_state.get("daily_workout_history", [])
    if w_hist:
        w_lines = [f"- [{w.get('date')}] {w.get('title')} (条件: {w.get('condition', '')})" for w in w_hist[-3:]]
        parts.append("【直近提案した日次ワークアウト履歴】\n" + "\n".join(w_lines))

    # 3. 現在の週間プラン方針
    wk_plan = st.session_state.get("weekly_plan")
    if wk_plan and isinstance(wk_plan, dict):
        summary = wk_plan.get("weekly_summary", "")[:200]
        parts.append(f"【現在進行中の週間プラン方針】\n{summary}")

    # 4. チャットでの直近の相談履歴（直近4メッセージ）
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
    ctl, atl, tsb = "未取得", "未取得", "未取得"

    if metrics:
        ctl, atl, tsb = metrics["ctl"], metrics["atl"], metrics["tsb"]
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
                f"- 推定eFTP: {p['eftp'] or '未算出'} W\n"
                f"- 弱点プロファイリング診断: {p['diagnosis']}"
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

    # 外的要因（事故・病気・怪我・制限）の構築
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

    return f"""あなたはプロレベルの科学的知見（運動生理学・自律神経HRV分析・スポーツ医学）を持つ専属AIサイクリングコーチです。
以下のアスリートデータ、HRVと睡眠スコアに基づくレディネス診断、外的要因（怪我・病気等）、および過去のコーチング記憶を踏まえ、一貫性のある具体的かつ論理的なアドバイスや練習メニューを提供してください。

{external_factor_prompt}

【🫀 本日のHRV・睡眠スコア・自律神経レディネス診断】
{wellness_summary}

【Intervals.icu 最新トレーニング負荷】
- CTL (Fitness / 長期負荷): {ctl}
- ATL (Fatigue / 短期疲労): {atl}
- TSB (Form / バランス): {tsb}

【アスリート基本プロファイル】
- 現在FTP: {profile['ftp']} W / 体重: {profile['weight']} kg (現在PWR: {pwr} W/kg)
- 目標FTP: {profile['target_ftp']} W (目標PWR: {round(profile['target_ftp']/profile['weight'], 2)} W/kg)
- 平日練習可能時間: {profile['weekday_hours']} 時間 / 休日練習可能時間: {profile['weekend_hours']} 時間

【目標レース情報】
- レース名: {profile['race_name']}
- 開催日: {profile['race_date']} (残り {days_left} 日 / 現在のフェーズ: {phase})
- レースカテゴリー・特性: {profile['race_type']}
- 目標・メモ: {profile['race_goal']}

【パワーデュレーションカーブ (PDC / 過去1年) & 弱点診断】
{pdc_summary}

【直近40日間のライド履歴（抜粋）】
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


def get_candidate_models(primary_model: str) -> list[str]:
    """クォータ制限時のフォールバック順序リストを生成"""
    defaults = ["gemini-3.8-flash", "gemini-3.1-pro", "gemini-4-argon", "gemini-2.0-flash"]
    candidates = [primary_model] if primary_model else []
    for m in defaults:
        if m not in candidates:
            candidates.append(m)
    return candidates


def stream_gemini_response(client_ai, model_name: str, contents, sys_prompt: str):
    """
    指定モデルが429（クォータ上限）や503、404の場合、
    制限のない代替モデルへ自動的にフォールバックしてストリーミング生成を行う。
    """
    candidates = get_candidate_models(model_name)
    last_err = None

    for current_m in candidates:
        for attempt in range(2):
            try:
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
                is_quota = any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded"])
                is_not_found = any(k in err_str for k in ["404", "NOT_FOUND"])
                if is_quota or is_not_found:
                    break
                is_busy = any(k in err_str for k in ["503", "UNAVAILABLE", "high demand"])
                if is_busy and attempt < 1:
                    time.sleep(2)
                    continue
                else:
                    break

    if last_err:
        raise last_err


def generate_content_with_retry(client_ai, model_name: str, contents, config):
    """
    通常生成（429クォータ上限時の自動モデルフォールバック対応）
    """
    candidates = get_candidate_models(model_name)
    last_err = None

    for current_m in candidates:
        for attempt in range(2):
            try:
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
                is_quota = any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded"])
                is_not_found = any(k in err_str for k in ["404", "NOT_FOUND"])
                if is_quota or is_not_found:
                    break
                is_busy = any(k in err_str for k in ["503", "UNAVAILABLE", "high demand"])
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
    if any(k in err_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded"]):
        st.warning("⚠️ 利用可能な全AIモデル（3.8 Flash / 3.1 Pro / 4 Argon 等）の無料利用枠上限に達しました。Google AI Studio のリセット時刻をお待ちいただくか、有料プランの紐付けをご検討ください。")
    elif any(k in err_str for k in ["503", "UNAVAILABLE", "high demand"]):
        st.warning("⚠️ Googleのサーバーでアクセス集中（一時的な混雑）が発生しています。数秒待って再試行してください。")
    elif any(k in err_str for k in ["404", "NOT_FOUND"]):
        st.warning("⚠️ 選択されたモデルが利用できない可能性があります。サイドバーのモデル設定をご確認ください。")


# ==========================================
# 5. Streamlit UI 構築
# ==========================================
st.set_page_config(
    page_title="AI Cycling Coach",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚴‍♂️ AI Cycling Coach")

# --- マルチユーザー & オーナー自動ログイン判定 ---
owner_cfg = load_owner_config()
q_mode = st.query_params.get("mode", "")
q_uid = st.query_params.get("uid", "")

if "user_mode" not in st.session_state:
    if q_mode == "guest" or q_uid:
        st.session_state.user_mode = "guest"
        if q_uid:
            st.session_state.guest_uid = q_uid
    elif q_mode == "owner":
        st.session_state.user_mode = "owner"
    else:
        # オーナーが「通常URLをゲスト画面にする」をONにしている場合以外は、今まで通りオーナー自動ログイン
        if owner_cfg.get("public_default_guest", False) or not owner_cfg.get("intervals_key"):
            st.session_state.user_mode = "guest"
        else:
            st.session_state.user_mode = "owner"

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

# --- サイドバー：ユーザー切替・設定入力・外的要因・自動保存 ---
with st.sidebar:
    st.header("👤 アカウント & モード切替")
    if is_owner_mode:
        st.success(f"👑 オーナー自動ログイン中 ({cfg.get('athlete_id') or '設定済'})")
        if st.button("👥 他の人が使う（別アカウント / API入力）", use_container_width=True):
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

        with st.expander("🔗 他の人へのアプリ共有・公開設定"):
            st.caption("友人に共有する際は、URLの末尾に `?mode=guest` を付けて送ると、最初からAPI入力画面が開きます。")
            st.code("https://ai-cycling-coach-ps662nnyskxx8krt7w7rnj.streamlit.app/?mode=guest", language="text")
            pub_default = st.checkbox(
                "🔒 通常URL（パラメータなし）もゲストAPI入力画面にする",
                value=bool(cfg.get("public_default_guest", False)),
                help="ONにすると、通常のURLを開いた時は全員『API入力画面』になり、あなた自身は ?mode=owner のブックマークURLで自動ログインできるようになります。"
            )
            if pub_default != bool(cfg.get("public_default_guest", False)):
                cfg["public_default_guest"] = pub_default
                save_config(cfg)
                st.success("公開デフォルト設定を更新しました！")
                if pub_default:
                    st.info("💡 あなた専用の自動ログインURL: 末尾に `?mode=owner` を付けてブックマークしてください。")
    else:
        pub_default = bool(owner_cfg.get("public_default_guest", False))
        if cfg.get("intervals_key") and cfg.get("gemini_key"):
            st.info(f"🚴 ゲスト利用中 (ID: {cfg.get('athlete_id') or '自動取得'})")
            if st.session_state.get("guest_uid"):
                st.caption("💡 現在のURL（`?mode=guest&uid=...`）をブックマークすると、次回もあなたの設定と記憶で自動ログインできます。")
            if st.button("🚪 ログアウト / 別のAPIキーを入力", use_container_width=True):
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
            st.warning("🔑 未ログイン（APIキーを入力してください）")

        if owner_cfg.get("intervals_key"):
            if not pub_default:
                if st.button("👑 オーナー（自分）の自動ログインに戻る", use_container_width=True):
                    st.session_state.user_mode = "owner"
                    try:
                        st.query_params["mode"] = "owner"
                        if "uid" in st.query_params:
                            del st.query_params["uid"]
                    except Exception:
                        pass
                    reload_user_session_data()
                    st.rerun()
            else:
                with st.expander("👑 オーナー（管理者）ログイン"):
                    owner_pin = st.text_input("オーナー Athlete ID を入力", placeholder="例: i123456")
                    if st.button("オーナーとして自動ログイン", use_container_width=True):
                        if owner_pin.strip() == owner_cfg.get("athlete_id", "").strip():
                            st.session_state.user_mode = "owner"
                            try:
                                st.query_params["mode"] = "owner"
                                if "uid" in st.query_params:
                                    del st.query_params["uid"]
                            except Exception:
                                pass
                            reload_user_session_data()
                            st.rerun()
                        else:
                            st.error("Athlete ID が一致しません。")

    st.divider()
    st.header("⚙️ API & モデル設定")

    if is_owner_mode:
        # オーナーモードでは既存のAPIキー文字列を画面のinput valueに直接出さず保護する
        athlete_id = cfg.get("athlete_id", "")
        intervals_key = cfg.get("intervals_key", "")
        gemini_key = cfg.get("gemini_key", "")
        with st.expander("🔑 オーナーAPIキーの変更（通常は変更不要）"):
            st.caption("セキュリティ保護のため現在のキーは非表示です。変更する場合のみ入力してください。")
            new_aid = st.text_input("Intervals Athlete ID", value=athlete_id, placeholder="例: i123456")
            new_ikey = st.text_input("新しい Intervals API Key", value="", type="password", placeholder="変更しない場合は空欄")
            new_gkey = st.text_input("新しい Gemini API Key", value="", type="password", placeholder="変更しない場合は空欄")
            if new_aid.strip():
                athlete_id = new_aid.strip()
            if new_ikey.strip():
                intervals_key = new_ikey.strip()
            if new_gkey.strip():
                gemini_key = new_gkey.strip()
    else:
        athlete_id = st.text_input("Intervals Athlete ID (空欄でも自動取得)", value=cfg.get("athlete_id", ""), placeholder="例: i123456")
        intervals_key = st.text_input("Intervals API Key", value=cfg.get("intervals_key", ""), type="password")
        gemini_key = st.text_input("Gemini API Key", value=cfg.get("gemini_key", ""), type="password")

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.1-pro",
        "gemini-4-argon",
        "カスタム指定...",
    ]
    cur_model = cfg.get("model", "gemini-3.8-flash")
    idx_model = model_options.index(cur_model) if cur_model in model_options else (len(model_options) - 1 if cur_model else 0)
    selected_option = st.selectbox(
        "🤖 AIモデル選択",
        model_options,
        index=idx_model,
        help="利用枠制限（429）に達した場合は自動で別モデルへ切り替わります。"
    )
    if selected_option == "カスタム指定...":
        custom_model = st.text_input("モデルIDを入力", value=cur_model if cur_model not in model_options else "gemini-3.8-flash")
        selected_model = custom_model.strip()
    else:
        selected_model = selected_option

    st.divider()
    st.subheader("🏥 外的要因・怪我・病気の設定")
    st.caption("事故や体調不良、部位の痛みなどを考慮してメニューを安全に調整します。")
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

    race_options = [
        "ヒルクライム (60分前後・L4/SST持続力重視)",
        "ショートヒルクライム (20〜40分・L4〜VO2max重視)",
        "ロードレース / サーキット (インターバル耐性・VO2max・スプリント)",
        "クリテリウム (高強度インターバル・無酸素反復能力)",
        "エンデューロ / ロングライド (有酸素耐久性・脂質代謝)"
    ]
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

# --- ゲスト（他の人）が未ログインの場合：メイン画面に分かりやすいAPI入力＆初期設定フォームを表示 ---
if not is_owner_mode and (not intervals_key or not gemini_key):
    st.info("👋 **AI Cycling Coach へようこそ！** ご自身の Intervals.icu API Key と Gemini API Key を入力して利用を開始できます。（入力したデータは他のユーザーとは完全に分離して処理されます）")

    with st.expander("📖 APIキーの取得方法（初回のみ・約1分）", expanded=True):
        st.markdown("""
1. **Intervals.icu API Key の取得**
   - **[Intervals.icu の設定ページ (Settings)](https://intervals.icu/settings)** を開きます。
   - ページ下部にある **「Developer Settings」** セクションの **「API Key」**（「View」または生成ボタン）をコピーします。
   - ※ Athlete ID（`i123456` 等）は空欄のままでもAPIキーから自動認識されます。
2. **Google Gemini API Key の取得（無料枠で利用可能）**
   - **[Google AI Studio (API Keys)](https://aistudio.google.com/app/apikey)** にGoogleアカウントでログインします。
   - **「Create API key」** を押して発行されたキーをコピーします。
        """)

    with st.form("guest_onboarding_form"):
        st.subheader("🔑 APIキーと基本目標の入力")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            in_intervals_key = st.text_input("1️⃣ Intervals.icu API Key (必須)", type="password", placeholder="Intervals.icu の API Key を貼り付け")
            in_athlete_id = st.text_input("Intervals Athlete ID (任意・空欄で自動取得)", placeholder="例: i123456（空欄でもOK）")
        with f_col2:
            in_gemini_key = st.text_input("2️⃣ Gemini API Key (必須)", type="password", placeholder="AIza... または AQ... から始まるキー")
            auto_fetch_phys = st.checkbox("✅ 現在のFTPと体重を Intervals.icu から自動取得する", value=True)

        st.markdown("##### 🏆 目標レース・フィジカル設定（後からサイドバーでいつでも変更可能）")
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            in_ftp = st.number_input("現在のFTP (W) ※自動取得時は上書きされます", 100, 500, 250, 5)
            in_weight = st.number_input("現在の体重 (kg) ※自動取得時は上書きされます", 35.0, 120.0, 60.0, 0.5)
        with g_col2:
            in_target_ftp = st.number_input("目標FTP (W)", 100, 550, 280, 5)
            in_race_name = st.text_input("目標レース名 / イベント名", value="Mt.富士ヒルクライム")
        with g_col3:
            in_race_date = st.date_input("目標開催日", value=datetime.now().date() + timedelta(days=60))
            in_race_type = st.selectbox("レース特性", race_options)

        submitted_login = st.form_submit_button("🚀 このAPIキーでログインしてデータ同期を開始", type="primary", use_container_width=True)

    if submitted_login:
        if not in_intervals_key.strip() or not in_gemini_key.strip():
            st.error("⚠️ Intervals.icu API Key と Gemini API Key の両方を入力してください。")
        else:
            try:
                with st.spinner("Intervals.icu に接続し、プロフィールと最新データを取得中..."):
                    temp_client = IntervalsClient(in_athlete_id.strip(), in_intervals_key.strip())
                    prof = temp_client.get_athlete_profile()
                    resolved_id = prof.get("athlete_id") or in_athlete_id.strip() or "0"
                    resolved_name = prof.get("name") or ""
                    resolved_ftp = prof.get("ftp") if (auto_fetch_phys and prof.get("ftp")) else int(in_ftp)
                    resolved_weight = prof.get("weight") if (auto_fetch_phys and prof.get("weight")) else float(in_weight)

                    # このユーザー固有のUIDをセットし、以前に保存された記憶があればロード
                    uid = compute_user_uid(in_intervals_key.strip(), resolved_id)
                    st.session_state.guest_uid = uid
                    reload_user_session_data()

                    new_guest_cfg = current_cfg_dict.copy()
                    new_guest_cfg.update({
                        "athlete_id": resolved_id,
                        "athlete_name": resolved_name,
                        "intervals_key": in_intervals_key.strip(),
                        "gemini_key": in_gemini_key.strip(),
                        "ftp": resolved_ftp,
                        "weight": resolved_weight,
                        "target_ftp": int(in_target_ftp),
                        "race_name": in_race_name,
                        "race_date": in_race_date.strftime("%Y-%m-%d"),
                        "race_type": in_race_type,
                    })
                    save_config(new_guest_cfg)

                    # データ同期実行
                    authed_client = IntervalsClient(resolved_id, in_intervals_key.strip())
                    wellness = authed_client.get_wellness(days=40)
                    activities = authed_client.get_activities(days=40)
                    pdc_raw = authed_client.get_power_curves(curves="1y")
                    st.session_state.metrics = parse_intervals_data(wellness, activities, pdc_raw, resolved_weight, resolved_ftp)
                    save_metrics_cache(st.session_state.metrics)

                st.success(f"✅ ログイン成功！（Athlete ID: {resolved_id} / FTP: {resolved_ftp}W / 体重: {resolved_weight}kg）")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 接続エラー: APIキーまたはAthlete IDをご確認ください。（詳細: {e}）")

    # オーナー自身がゲスト画面を確認している場合にワンクリックで戻れるボタン
    if owner_cfg.get("intervals_key") and not bool(owner_cfg.get("public_default_guest", False)):
        st.divider()
        if st.button("👑 オーナー（自分）の自動ログイン画面に戻る"):
            st.session_state.user_mode = "owner"
            try:
                st.query_params["mode"] = "owner"
                if "uid" in st.query_params:
                    del st.query_params["uid"]
            except Exception:
                pass
            reload_user_session_data()
            st.rerun()

    st.stop()

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
        with st.spinner("Intervals.icuから最新データ（HRV・睡眠・PDC・履歴）を取得中..."):
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
            st.toast("✅ Intervals.icu 同期＆HRV/睡眠/パワーカーブ解析完了！")
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

# 初回起動時（またはキャッシュにwellness_readinessがない場合）に自動同期
if (st.session_state.metrics is None or "wellness_readiness" not in st.session_state.metrics) and intervals_key:
    sync_intervals_now(show_msg=False)

# --- メイン画面上部：ワンタップ同期ボタン & ユーザー切替ボタン & コンディション指標 ---
top_col1, top_col2, top_col3 = st.columns([3.2, 1, 1])
with top_col1:
    days_to_race = (race_date - datetime.now().date()).days
    user_badge = f"👑 オーナー ({athlete_id})" if is_owner_mode else f"👤 ゲスト ({athlete_id or 'User'})"
    st.markdown(f"**🏆 目標: {race_name}** （残り **{days_to_race}日** / {determine_training_phase(days_to_race)}） ｜ {user_badge} ｜ 🤖 `{selected_model}`")
with top_col2:
    if st.button("🔄 最新データを再同期", use_container_width=True):
        sync_intervals_now(show_msg=True)
with top_col3:
    if is_owner_mode:
        if st.button("👥 他ユーザー利用へ切替", use_container_width=True, help="他の人が自分のAPIキーを入力して使う画面へ切り替えます"):
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
        if st.button("🚪 別API入力 / 切替", use_container_width=True, help="ログアウトしてAPIキー入力画面に戻ります"):
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
        f"🏥 **外的要因・特別配慮モード適用中:** 【{ext_status}】 / **上限:** {ext_intensity_cap}"
        + (f"\n\n📝 **詳細メモ:** {ext_detail}" if ext_detail.strip() else "")
    )

current_pwr = round(ftp / weight, 2)
target_pwr = round(target_ftp / weight, 2)

# メトリック行1: パワー & 負荷指標
m1, m2, m3, m4 = st.columns(4)
m1.metric("現在 PWR", f"{current_pwr} W/kg", f"目標 {target_pwr} W/kg")
if st.session_state.metrics:
    m2.metric("CTL (Fitness)", st.session_state.metrics["ctl"])
    m3.metric("ATL (Fatigue)", st.session_state.metrics["atl"])
    tsb_val = st.session_state.metrics["tsb"]
    tsb_status = "フレッシュ" if tsb_val > 5 else ("適正負荷" if tsb_val >= -15 else "過疲労注意")
    m4.metric("TSB (Form)", tsb_val, tsb_status)
else:
    m2.metric("CTL", "--")
    m3.metric("ATL", "--")
    m4.metric("TSB", "--")

# メトリック行2 & エクスパンダー: HRV・睡眠スコア・レディネス診断
if st.session_state.metrics and st.session_state.metrics.get("wellness_readiness"):
    wr = st.session_state.metrics["wellness_readiness"]
    with st.expander(f"🫀 本日のHRV・睡眠コンディション診断 — 【{wr['status_label']} / スコア: {wr['score']}点】", expanded=True):
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        hrv_delta = f"基準中央値: {wr['hrv_base']} ms" if wr.get("hrv_base") else None
        w_col1.metric("❤️ 最新 HRV", f"{wr['hrv']} ms" if wr.get("hrv") else "--", hrv_delta)

        slp_delta = f"睡眠時間: {wr['sleep_hours']}h" if wr.get("sleep_hours") else None
        w_col2.metric("😴 睡眠スコア", f"{wr['sleep_score']} 点" if wr.get("sleep_score") else "--", slp_delta)

        rhr_delta = f"基準中央値: {wr['rhr_base']} bpm" if wr.get("rhr_base") else None
        w_col3.metric("💓 安静時心拍 (RHR)", f"{wr['rhr']} bpm" if wr.get("rhr") else "--", rhr_delta)

        w_col4.metric("⚡ レディネススコア", f"{wr['score']} / 100", wr["status_label"].split(" ")[0])

        st.info(f"**【AIコンディション判定 & 推奨強度】**\n\n{wr['recommendation']}\n\n- " + "\n- ".join(wr.get("reasons", [])))

        w_hist = st.session_state.metrics.get("wellness_history", [])
        w_chart = build_wellness_chart(w_hist)
        if w_chart:
            st.caption("📈 直近14日間の HRV (紫・実線) ＆ 睡眠スコア (水色・破線) 推移")
            st.altair_chart(w_chart, use_container_width=True)

# パワーカーブ弱点診断バナー ＆ ビジュアルチャート
if st.session_state.metrics and "pdc" in st.session_state.metrics:
    pdc = st.session_state.metrics["pdc"]
    with st.expander("📈 過去1年のパワーカーブ (PDC) & AI弱点プロファイリング結果", expanded=False):
        st.info(pdc["diagnosis"])
        pdc_chart = build_pdc_chart(pdc, weight, ftp, target_ftp)
        st.altair_chart(pdc_chart, use_container_width=True)

        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("15秒 (Sprint)", f"{pdc['watts']['15s']} W", f"{pdc['wkg']['15s']} W/kg")
        pc2.metric("1分 (Anaerobic)", f"{pdc['watts']['1m']} W", f"{pdc['wkg']['1m']} W/kg")
        pc3.metric("5分 (VO2max)", f"{pdc['watts']['5m']} W", f"{pdc['wkg']['5m']} W/kg")
        pc4.metric("20分 (Threshold)", f"{pdc['watts']['20m']} W", f"{pdc['wkg']['20m']} W/kg")
        pc5.metric("60分 / eFTP", f"{pdc['watts']['60m']} W", f"eFTP: {pdc['eftp'] or '--'}W")

# --- タブ構成 ---
tab_daily, tab_weekly, tab_debrief, tab_chat, tab_history = st.tabs([
    "🎯 今日のメニュー",
    "📅 週間プラン一括生成",
    "🚴‍♂️ ライド振り返り・診断",
    "💬 AIコーチ相談",
    "📊 ライド履歴"
])


# ==========================================
# Tab 1: 今日のAIワークアウト生成
# ==========================================
with tab_daily:
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        avail_minutes = st.slider("今日の練習時間 (分)", 30, 240, 75, 15)
    with col_opt2:
        indoor_mode = st.selectbox("環境", ["インドア (MyWhoosh/Zwift等)", "アウトドア (実走)"])
    with col_opt3:
        rpe_feeling = st.selectbox(
            "体調・メニュー方針",
            [
                "🤖 HRV・睡眠スコアからAI自動最適化（推奨）",
                "普通（予定通り）",
                "脚が重い・疲労感あり",
                "絶好調（追い込みたい）",
                "リカバリー・回復走希望",
            ]
        )

    today_note = st.text_input(
        "今日の補足メモ・体調変化・外的要因（任意）",
        placeholder="例: 昨日少し寝不足、右膝に少し違和感あり、今日はケイデンス意識したい 等"
    )

    if st.button("⚡ 今日の最適ワークアウトを生成", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("サイドバーで Gemini API Key を設定・保存してください。")
        else:
            sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
            user_req = f"""
以下の条件で、本日実施すべき最適なワークアウトを1つ作成してください。
- 使用可能時間: {avail_minutes} 分
- 環境: {indoor_mode}
- 体調・メニュー方針: {rpe_feeling}
- 本日の補足メモ: {today_note or '特になし'}
- 外的要因・制限設定: {ext_status} / 強度上限: {ext_intensity_cap} / 詳細: {ext_detail or 'なし'}

【出力構成】
1. **本日のコンディション判定とメニューの狙い**（最新のHRV・睡眠スコア・安静時心拍・TSB、外的要因や過去の記憶、パワーカーブ弱点をどう考慮したか）
2. **メニュー詳細とアドバイス**（各セクションの目標ワット数 `{ftp}W` 基準の実数値、ケイデンス、注意点）
3. **Intervals.icu カレンダー登録用のワークアウトコードブロック**（必ず ```workout から始めて記述すること。完全休養を推奨する場合はコードブロックは省略可）
"""
            try:
                client_ai = genai.Client(api_key=gemini_key)
                st.session_state.generated_workout = ""
                with st.spinner(f"AIコーチ（{selected_model}）がHRV・睡眠・記憶を踏まえてメニューを設計中..."):
                    response_text = st.write_stream(
                        stream_gemini_response(client_ai, selected_model, user_req, sys_prompt)
                    )
                    st.session_state.generated_workout = response_text

                    # 履歴に保存
                    m_code = re.search(r"```workout\n(.*?)```", response_text, re.DOTALL)
                    w_code_extracted = m_code.group(1).strip() if m_code else ""
                    st.session_state.daily_workout_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "title": f"AI Workout ({avail_minutes}min / {rpe_feeling[:10]})",
                        "condition": f"{rpe_feeling} / {today_note}",
                        "text": response_text,
                        "workout_code": w_code_extracted,
                    })
                    save_memory()
            except Exception as e:
                show_ai_error("AI生成エラー", e, selected_model)

    # 生成済みワークアウトの表示（リロード後も記憶から表示）
    if st.session_state.generated_workout:
        with st.container():
            st.markdown(st.session_state.generated_workout)

        match = re.search(r"```workout\n(.*?)```", st.session_state.generated_workout, re.DOTALL)
        if match:
            workout_code = match.group(1).strip()
            steps, df_chart = parse_intervals_workout(workout_code, ftp)

            st.divider()
            st.subheader("📊 ワークアウト・パワープロファイル")
            chart = build_workout_chart(df_chart, ftp)
            if chart:
                st.altair_chart(chart, use_container_width=True)

            col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
            with col_p1:
                w_name = st.text_input("登録名", value=f"AI Coach ({avail_minutes}min)")
            with col_p2:
                w_date = st.date_input("登録日", value=datetime.now().date())
            with col_p3:
                zwo_xml = generate_zwo_xml(w_name, steps, ftp, description=f"AI Coach Plan for {w_date}")
                st.download_button(
                    label="📥 MyWhoosh用 (.zwo) DL",
                    data=zwo_xml,
                    file_name=f"{w_name.replace(' ', '_')}_{w_date}.zwo",
                    mime="application/xml",
                    use_container_width=True,
                    help="ダウンロードした .zwo ファイルは workout.mywhoosh.com にアップロードすると MyWhoosh 内で走れます！"
                )

            if st.button("🚀 Intervals.icu カレンダーに登録", use_container_width=True):
                try:
                    client = IntervalsClient(athlete_id, intervals_key)
                    client.post_workout(w_name, workout_code, w_date.strftime("%Y-%m-%d"))
                    st.success(f"✅ {w_date} のカレンダーに『{w_name}』を登録しました！")
                except Exception as e:
                    st.error(f"登録エラー: {e}")

            with st.expander("💡 MyWhoosh でこのワークアウトを走る手順"):
                st.markdown("""
1. 右上の **「📥 MyWhoosh用 (.zwo) DL」** を押し、`.zwo` ファイルを保存します。
2. ブラウザで **[MyWhoosh Workout Builder](https://workout.mywhoosh.com)** にアクセスしてログインします。
3. **「Import」** を選んで保存した `.zwo` ファイルをアップロードします。
4. MyWhooshアプリ（PC/タブレット/スマホ）を開き、**「Workouts」>「Custom Workouts」** を開くとメニューが表示されます！
                """)

    # 過去に生成した今日のメニュー履歴アーカイブ
    if st.session_state.daily_workout_history:
        st.divider()
        with st.expander(f"📚 過去に生成したメニュー履歴 ({len(st.session_state.daily_workout_history)}件)"):
            for idx, item in enumerate(reversed(st.session_state.daily_workout_history)):
                st.markdown(f"**{item.get('date')} — {item.get('title')}**")
                if st.button(f"このメニューを画面に呼び出す", key=f"restore_daily_{idx}"):
                    st.session_state.generated_workout = item.get("text")
                    save_memory()
                    st.rerun()
                st.divider()


# ==========================================
# Tab 2: 週間プラン一括生成 & カレンダー同期
# ==========================================
with tab_weekly:
    st.subheader("📅 1週間分の練習メニュー一括設計 ＆ カレンダー同期")
    st.caption("HRV・睡眠トレンド、外的要因（怪我・病気等）、パワーカーブ弱点を統合した7日間の最適プランを作成・記憶します。")

    col_wk1, col_wk2 = st.columns(2)
    with col_wk1:
        start_plan_date = st.date_input("プラン開始日（月曜日など）", value=datetime.now().date())
    with col_wk2:
        weekly_note = st.text_input("今週の予定・体調・要望メモ", value="水曜と金曜は休養日にしたい、日曜は実走メイン")

    if st.button("🗓️ 7日分の週間メニューを一括生成", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("サイドバーで Gemini API Key を設定・保存してください。")
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
- 現在のHRV・睡眠スコアによるレディネス状態、TSB、および外的要因（{ext_status} / 上限:{ext_intensity_cap} / {ext_detail}）を厳密に考慮し、無理のないメリハリある7日間にすること。
- 練習日の `workout_code` には、Intervals.icuのカレンダーでそのままグラフ化される正確なワークアウト記法（Warmup / Main Set / Cooldown、`- 10m 90%` 等の形式）を記述すること。
"""
            try:
                with st.spinner(f"AIコーチ（{selected_model}）が7日間のメニューを構造化生成中..."):
                    client_ai = genai.Client(api_key=gemini_key)
                    response = generate_content_with_retry(
                        client_ai=client_ai,
                        model_name=selected_model,
                        contents=weekly_req,
                        config=types.GenerateContentConfig(
                            system_instruction=sys_prompt,
                            response_mime_type="application/json",
                            response_schema=WeeklyPlanResponse,
                        )
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
        st.success("✨ 保存されている週間プランを表示中")
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
                        st.altair_chart(ch_w, use_container_width=True)
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
# Tab 3: ライド振り返り・自動診断 (Post-Ride Debrief & 記憶保存)
# ==========================================
with tab_debrief:
    st.subheader("🚴‍♂️ 直近ライドの振り返り・自動コーチング診断（評価記憶機能付き）")
    st.caption("一度評価したライドは自動記憶され、次回以降のメニュー作成やチャット相談にも活かされます。")

    if st.session_state.metrics and st.session_state.metrics.get("activities"):
        acts = st.session_state.metrics["activities"]
        ride_options = []
        for a in acts:
            aid = str(a.get("id", ""))
            reviewed_mark = "✅評価済 " if aid in st.session_state.ride_reviews else ""
            ride_options.append(
                f"{reviewed_mark}{a['日付']} - {a['ワークアウト名']} ({a['時間(分)']}分, TSS:{a['TSS']}, NP:{a['NP(W)']}W)"
            )

        selected_ride_idx = st.selectbox("振り返るライドを選択", range(len(ride_options)), format_func=lambda i: ride_options[i])
        selected_ride = acts[selected_ride_idx]
        act_id = str(selected_ride.get("id", ""))

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("時間 / 距離", f"{selected_ride['時間(分)']} 分", f"{selected_ride.get('距離(km)', '--')} km")
        col_r2.metric("TSS / IF", f"{selected_ride['TSS']}", f"IF: {selected_ride['IF']}")
        col_r3.metric("NP / 平均パワー", f"{selected_ride['NP(W)']} W", f"平均: {selected_ride['平均(W)']} W")
        col_r4.metric("平均心拍", f"{selected_ride['心拍(bpm)']} bpm" if selected_ride['心拍(bpm)'] > 0 else "--")

        # Stravaセグメントログの取得と表示
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

        # 過去の保存済み評価があれば初期値にセット
        saved_review_obj = st.session_state.ride_reviews.get(act_id)
        default_memo = saved_review_obj.get("memo", "") if saved_review_obj else ""

        user_ride_feedback = st.text_input(
            "ライド時の体感メモ・補足（任意・記憶されます）",
            value=default_memo,
            placeholder="例: 後半脚がタレた、補給が足りなかった、病み上がりで心拍が高めだった等"
        )

        btn_label = "🔄 このライドを再評価して記憶を更新する" if saved_review_obj else "🔍 このライドをAIコーチにレビューしてもらう"
        if st.button(btn_label, type="primary", use_container_width=True):
            if not gemini_key:
                st.error("サイドバーで Gemini API Key を設定・保存してください。")
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
- 日付: {selected_ride['日付']}
- ワークアウト名: {selected_ride['ワークアウト名']}
- 時間: {selected_ride['時間(分)']} 分 / 距離: {selected_ride.get('距離(km)', '--')} km
- TSS: {selected_ride['TSS']} / IF: {selected_ride['IF']}
- NP: {selected_ride['NP(W)']} W / 平均パワー: {selected_ride['平均(W)']} W (FTP: {ftp}W)
- 平均心拍: {selected_ride['心拍(bpm)']} bpm
- アスリートの体感メモ: {user_ride_feedback or '特になし'}

【Strava 通過セグメント実績】
{seg_summary}

【分析・フィードバック項目】
1. **総括と負荷評価**: 現在のHRV・睡眠コンディションやTSB、外的要因（怪我・病気等）に照らし合わせたTSS/IFの適切さ
2. **強度と効率の分析**: NPと平均Wの乖離（変動性VI）、パワーに対する心拍の妥当性（心拍ドリフトや疲労の兆候、過去のライド評価との比較）
3. **Stravaセグメント分析**: 登坂やスプリント各セグメントでの踏み込み・ペーシング・FTP比評価
4. **今日のGoodポイント**: 褒めるべき点や成果
5. **リカバリー＆次回へのアドバイス**: 今夜の栄養（糖質・タンパク質補給）、睡眠・HRV回復に向けたケア、明日以降の調整
"""
                try:
                    client_ai = genai.Client(api_key=gemini_key)
                    st.divider()
                    st.subheader("📋 AIコーチからのフィードバック")
                    with st.spinner(f"AIコーチ（{selected_model}）がデータを診断中..."):
                        review_text = st.write_stream(
                            stream_gemini_response(client_ai, selected_model, debrief_req, sys_prompt)
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
        st.info("ライド履歴がありません。上部の「🔄 最新データを再同期」を押してデータを取得してください。")


# ==========================================
# Tab 4: AIコーチ・チャット相談（永続記憶付き）
# ==========================================
with tab_chat:
    col_ch1, col_ch2 = st.columns([4, 1])
    with col_ch1:
        st.caption("💬 過去の会話履歴・ライド評価・HRV/睡眠状態・怪我や病気の設定をすべて記憶した状態で相談できます。")
    with col_ch2:
        if st.button("🗑️ 会話履歴をクリア", use_container_width=True):
            st.session_state.messages = []
            save_memory()
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("例: 今日のHRVと睡眠スコア的に走っても大丈夫？ / 落車後の復帰メニューを相談したい"):
        if not gemini_key:
            st.error("サイドバーで Gemini API Key を設定・保存してください。")
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            save_memory()
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
                    client_ai = genai.Client(api_key=gemini_key)
                    history_contents = [
                        types.Content(
                            role="user" if m["role"] == "user" else "model",
                            parts=[types.Part.from_text(text=m["content"])]
                        )
                        for m in st.session_state.messages[-20:]  # 直近20メッセージを送信
                    ]
                    reply = st.write_stream(
                        stream_gemini_response(client_ai, selected_model, history_contents, sys_prompt)
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    save_memory()
                except Exception as e:
                    show_ai_error("チャット応答エラー", e, selected_model)


# ==========================================
# Tab 5: 直近のアクティビティ履歴
# ==========================================
with tab_history:
    if st.session_state.metrics and not st.session_state.metrics["activities_df"].empty:
        st.dataframe(st.session_state.metrics["activities_df"], use_container_width=True)
    else:
        st.info("データがありません。上部の「🔄 最新データを再同期」を押してください。")

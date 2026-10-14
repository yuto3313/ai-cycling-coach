import streamlit as st
import requests
import pandas as pd
import json
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import altair as alt

CONFIG_FILE = "config.json"

# ==========================================
# 0. 設定の自動保存・読み込みマネージャー
# ==========================================
DEFAULT_CONFIG = {
    "athlete_id": "",
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
    except Exception as e:
        st.warning(f"設定ファイルの保存に失敗しました（クラウド環境の場合は正常です）: {e}")


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

    def get_wellness(self, days: int = 14) -> list:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/wellness"
        res = requests.get(url, auth=self.auth, params={"oldest": start_date, "newest": end_date}, timeout=10)
        res.raise_for_status()
        return res.json()

    def get_activities(self, days: int = 14) -> list:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"{self.BASE_URL}/athlete/{self.athlete_id}/activities"
        res = requests.get(url, auth=self.auth, params={"oldest": start_date, "newest": end_date}, timeout=10)
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
# 3. ワークアウト解析・ZWO生成・Altairチャート
# ==========================================
def parse_intervals_workout(workout_text: str, ftp: int):
    """
    Intervals.icu のワークアウト記法をパースして、
    グラフ描画用のDataFrameとZWO用ステップ構造を返す。
    """
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
    """
    MyWhoosh / Zwift 互換の .zwo (XML) ファイル文字列を生成
    """
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
    """
    ワークアウトのパワー推移をAltairで描画
    """
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
    """
    パワーカーブ (MMP) と目標PWRの比較グラフ
    """
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


# ==========================================
# 4. データ解析 & パワーカーブ弱点診断ロジック
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
    w_1m = extracted["1m"]
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
            "TSS": round(act.get("icu_training_load") or 0),
            "NP(W)": round(act.get("icu_weighted_avg_watts") or 0),
            "平均(W)": round(act.get("average_watts") or 0),
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


def build_system_prompt(profile: dict, metrics: dict) -> str:
    days_left = (profile["race_date"] - datetime.now().date()).days
    phase = determine_training_phase(days_left)
    pwr = round(profile["ftp"] / profile["weight"], 2) if profile["weight"] > 0 else 0

    act_summary = "データなし"
    pdc_summary = "データなし"
    ctl, atl, tsb = "未取得", "未取得", "未取得"

    if metrics:
        ctl, atl, tsb = metrics["ctl"], metrics["atl"], metrics["tsb"]
        if not metrics["activities_df"].empty:
            act_summary = metrics["activities_df"].to_markdown(index=False)
        if "pdc" in metrics:
            p = metrics["pdc"]
            w = p["watts"]
            wkg = p["wkg"]
            pdc_summary = (
                f"- 直近42日ベストパワー: 15秒={w['15s']}W({wkg['15s']}W/kg), "
                f"1分={w['1m']}W({wkg['1m']}W/kg), 5分={w['5m']}W({wkg['5m']}W/kg), "
                f"20分={w['20m']}W({wkg['20m']}W/kg), 60分={w['60m']}W({wkg['60m']}W/kg)\n"
                f"- 推定eFTP: {p['eftp'] or '未算出'} W\n"
                f"- 弱点プロファイリング診断: {p['diagnosis']}"
            )

    return f"""あなたはプロレベルの科学的知見を持つAIサイクリングコーチです。
以下のアスリートデータ、Intervals.icuの最新メトリクス、およびパワーデュレーションカーブ(PDC)の弱点診断に基づき、具体的かつ論理的なアドバイスや練習メニューを提供してください。

【アスリート基本プロファイル】
- 現在FTP: {profile['ftp']} W / 体重: {profile['weight']} kg (現在PWR: {pwr} W/kg)
- 目標FTP: {profile['target_ftp']} W (目標PWR: {round(profile['target_ftp']/profile['weight'], 2)} W/kg)
- 平日練習可能時間: {profile['weekday_hours']} 時間 / 休日練習可能時間: {profile['weekend_hours']} 時間

【目標レース情報】
- レース名: {profile['race_name']}
- 開催日: {profile['race_date']} (残り {days_left} 日 / 現在のフェーズ: {phase})
- レースカテゴリー・特性: {profile['race_type']}
- 目標・メモ: {profile['race_goal']}

【Intervals.icu 最新コンディション】
- CTL (Fitness / 長期負荷): {ctl}
- ATL (Fatigue / 短期疲労): {atl}
- TSB (Form / コンディション): {tsb}

【パワーデュレーションカーブ (PDC / MMP) & 弱点診断】
{pdc_summary}

【直近14日間のライド履歴】
{act_summary}

【Intervals.icu ワークアウト記法の書式ルール】
Warmup
- 10m 50% 90rpm
Main Set 3x
- 8m 98%
- 4m 55%
Cooldown
- 10m 45%
"""


def stream_gemini_response(client_ai, model_name: str, contents, sys_prompt: str):
    """
    Geminiのストリーミング応答ジェネレータ
    """
    response_stream = client_ai.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=sys_prompt)
    )
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


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

cfg = load_config()

if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated_workout" not in st.session_state:
    st.session_state.generated_workout = None
if "weekly_plan" not in st.session_state:
    st.session_state.weekly_plan = None
if "ride_review_result" not in st.session_state:
    st.session_state.ride_review_result = None

# --- サイドバー：設定入力 ＆ 自動保存 ---
with st.sidebar:
    st.header("⚙️ 設定 & 自動保存")
    st.caption("一度保存すれば次回から入力不要です。")

    athlete_id = st.text_input("Intervals Athlete ID", value=cfg.get("athlete_id", ""), placeholder="例: i123456")
    intervals_key = st.text_input("Intervals API Key", value=cfg.get("intervals_key", ""), type="password")
    gemini_key = st.text_input("Gemini API Key", value=cfg.get("gemini_key", ""), type="password")

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.1-pro",
        "gemini-4-argon",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "カスタム指定...",
    ]
    cur_model = cfg.get("model", "gemini-3.8-flash")
    idx_model = model_options.index(cur_model) if cur_model in model_options else (len(model_options) - 1 if cur_model else 0)
    selected_option = st.selectbox(
        "🤖 AIモデル選択",
        model_options,
        index=idx_model,
        help="Gemini 3.8 Flash, 3.1 Pro, 4 Argon 等の最新推論モデルを選択できます。"
    )
    if selected_option == "カスタム指定...":
        custom_model = st.text_input("モデルIDを入力", value=cur_model if cur_model not in model_options else "gemini-3.8-flash")
        selected_model = custom_model.strip()
    else:
        selected_model = selected_option

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
    }

    if st.button("💾 設定を保存する", use_container_width=True):
        save_config(current_cfg_dict)
        st.success("設定を config.json に保存しました！")

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
}

def sync_intervals_now(show_msg=True):
    if not athlete_id or not intervals_key:
        if show_msg:
            st.warning("左上の「＞」からサイドバーを開き、Athlete IDとAPI Keyを入力・保存してください。")
        return
    try:
        with st.spinner("Intervals.icuから最新データ＆パワーカーブを取得中..."):
            save_config(current_cfg_dict)
            client = IntervalsClient(athlete_id, intervals_key)
            wellness = client.get_wellness(days=14)
            activities = client.get_activities(days=14)
            pdc_raw = client.get_power_curves(curves="42d")
            st.session_state.metrics = parse_intervals_data(wellness, activities, pdc_raw, weight, ftp)
        if show_msg:
            st.toast("✅ Intervals.icu 同期＆パワーカーブ解析完了！")
    except Exception as e:
        st.error(f"同期エラー: {e}")

if st.session_state.metrics is None and athlete_id and intervals_key:
    sync_intervals_now(show_msg=False)

# --- メイン画面上部：ワンタップ同期ボタン & コンディション指標 ---
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    days_to_race = (race_date - datetime.now().date()).days
    st.markdown(f"**🏆 目標: {race_name}** （残り **{days_to_race}日** / {determine_training_phase(days_to_race)}） ｜ 🤖 `{selected_model}`")
with top_col2:
    if st.button("🔄 最新データを再同期", use_container_width=True):
        sync_intervals_now(show_msg=True)

current_pwr = round(ftp / weight, 2)
target_pwr = round(target_ftp / weight, 2)

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

# パワーカーブ弱点診断バナー ＆ ビジュアルチャート
if st.session_state.metrics and "pdc" in st.session_state.metrics:
    pdc = st.session_state.metrics["pdc"]
    with st.expander("📈 直近42日のパワーカーブ (PDC) & AI弱点プロファイリング結果", expanded=True):
        st.info(pdc["diagnosis"])
        
        # PDCチャート描画
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
        rpe_feeling = st.selectbox("体調・脚の感覚", ["普通（予定通り）", "脚が重い・疲労感あり", "絶好調（追い込みたい）", "リカバリー希望"])

    if st.button("⚡ 今日の最適ワークアウトを生成", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("サイドバーで Gemini API Key を設定・保存してください。")
        else:
            sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
            user_req = f"""
以下の条件で、本日実施すべき最適なワークアウトを1つ作成してください。
- 使用可能時間: {avail_minutes} 分 / 環境: {indoor_mode} / 体調: {rpe_feeling}
【出力構成】
1. 今日のメニューの狙い（現在のTSB、パワーカーブの弱点診断、目標レースまでの残り日数を踏まえた理由）
2. メニュー詳細（各セクションの目標ワット数 `{ftp}W` 基準の実数値とケイデンス）
3. Intervals.icu カレンダー登録用のワークアウトコードブロック（必ず ```workout から始めて記述すること）
"""
            client_ai = genai.Client(api_key=gemini_key)
            st.session_state.generated_workout = ""
            with st.spinner("AIコーチがメニューを設計中..."):
                response_text = st.write_stream(
                    stream_gemini_response(client_ai, selected_model, user_req, sys_prompt)
                )
                st.session_state.generated_workout = response_text

    # 生成済みワークアウトの表示とグラフ化・ダウンロード
    if st.session_state.generated_workout:
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
                # MyWhoosh / Zwift 用の .zwo XML 生成
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


# ==========================================
# Tab 2: 週間プラン一括生成 & カレンダー同期
# ==========================================
with tab_weekly:
    st.subheader("📅 1週間分の練習メニュー一括設計 ＆ カレンダー同期")
    st.caption("パワーカーブの弱点克服と平日の練習時間枠・休日のロング枠を組み合わせた7日間の最適プランを作成します。")

    col_wk1, col_wk2 = st.columns(2)
    with col_wk1:
        start_plan_date = st.date_input("プラン開始日（月曜日など）", value=datetime.now().date())
    with col_wk2:
        weekly_note = st.text_input("今週の予定・要望メモ", value="水曜と金曜は休養日にしたい、日曜は実走メイン")

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
- 現在のTSBとパワーデュレーションカーブの弱点診断（5分対20分のバランス等）を反映し、メリハリのある7日間にすること。
- 練習日の `workout_code` には、Intervals.icuのカレンダーでそのままグラフ化される正確なワークアウト記法（Warmup / Main Set / Cooldown、`- 10m 90%` 等の形式）を記述すること。
"""
            with st.spinner(f"AIコーチ（{selected_model}）が7日間のメニューを構造化生成中..."):
                client_ai = genai.Client(api_key=gemini_key)
                response = client_ai.models.generate_content(
                    model=selected_model,
                    contents=weekly_req,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        response_mime_type="application/json",
                        response_schema=WeeklyPlanResponse,
                    )
                )
                st.session_state.weekly_plan = json.loads(response.text)

    if st.session_state.weekly_plan:
        plan = st.session_state.weekly_plan
        st.success("✨ 週間プランが生成されました！")
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
            if not athlete_id or not intervals_key:
                st.error("Athlete IDとAPI Keyを設定してください。")
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
# Tab 3: ライド振り返り・自動診断 (Post-Ride Debrief)
# ==========================================
with tab_debrief:
    st.subheader("🚴‍♂️ 直近ライドの振り返り・自動コーチング診断")
    st.caption("走った後のデータをAIが分析し、負荷の妥当性・心拍効率・リカバリーアドバイスをフィードバックします。")

    if st.session_state.metrics and st.session_state.metrics.get("activities"):
        acts = st.session_state.metrics["activities"]
        ride_options = [
            f"{a['日付']} - {a['ワークアウト名']} ({a['時間(分)']}分, TSS:{a['TSS']}, NP:{a['NP(W)']}W)"
            for a in acts
        ]
        selected_ride_idx = st.selectbox("振り返るライドを選択", range(len(ride_options)), format_func=lambda i: ride_options[i])
        selected_ride = acts[selected_ride_idx]

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("時間", f"{selected_ride['時間(分)']} 分")
        col_r2.metric("TSS / IF", f"{selected_ride['TSS']}", f"IF: {selected_ride['IF']}")
        col_r3.metric("NP / 平均パワー", f"{selected_ride['NP(W)']} W", f"平均: {selected_ride['平均(W)']} W")
        col_r4.metric("平均心拍", f"{selected_ride['心拍(bpm)']} bpm" if selected_ride['心拍(bpm)'] > 0 else "--")

        user_ride_feedback = st.text_input("ライド時の体感メモ・補足（任意）", placeholder="例: 後半脚がタレた、補給が足りなかった、最後まで踏み切れた等")

        if st.button("🔍 このライドをAIコーチにレビューしてもらう", type="primary", use_container_width=True):
            if not gemini_key:
                st.error("サイドバーで Gemini API Key を設定・保存してください。")
            else:
                sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
                debrief_req = f"""
以下のライド実績について、プロのサイクリングコーチとして多角的な分析とアドバイスを行ってください。

【対象ライドデータ】
- 日付: {selected_ride['日付']}
- ワークアウト名: {selected_ride['ワークアウト名']}
- 時間: {selected_ride['時間(分)']} 分
- TSS: {selected_ride['TSS']} / IF: {selected_ride['IF']}
- NP: {selected_ride['NP(W)']} W / 平均パワー: {selected_ride['平均(W)']} W (FTP: {ftp}W)
- 平均心拍: {selected_ride['心拍(bpm)']} bpm
- アスリートの体感メモ: {user_ride_feedback or '特になし'}

【分析・フィードバック項目】
1. **総括と負荷評価**: 現在のTSBやフェーズに照らし合わせたTSS/IFの適切さ
2. **強度と効率の分析**: NPと平均Wの乖離（変動性VI）、パワーに対する心拍の妥当性（心拍ドリフトや疲労の兆候）
3. **今日のGoodポイント**: 褒めるべき点や成果
4. **リカバリー＆次回へのアドバイス**: 今夜の栄養（糖質・タンパク質補給）、ストレッチ、明日以降の調整
"""
                client_ai = genai.Client(api_key=gemini_key)
                st.divider()
                st.subheader("📋 AIコーチからのフィードバック")
                with st.spinner("AIコーチがデータを診断中..."):
                    review_text = st.write_stream(
                        stream_gemini_response(client_ai, selected_model, debrief_req, sys_prompt)
                    )
                    st.session_state.ride_review_result = review_text
    else:
        st.info("ライド履歴がありません。上部の「🔄 最新データを再同期」を押してデータを取得してください。")


# ==========================================
# Tab 4: AIコーチ・チャット相談
# ==========================================
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("例: 5分パワーに対して20分パワーを伸ばすコツは？ / 今週の疲労感が強い時の対処法"):
        if not gemini_key:
            st.error("サイドバーで Gemini API Key を設定・保存してください。")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
                client_ai = genai.Client(api_key=gemini_key)
                history_contents = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    )
                    for m in st.session_state.messages
                ]
                reply = st.write_stream(
                    stream_gemini_response(client_ai, selected_model, history_contents, sys_prompt)
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})


# ==========================================
# Tab 5: 直近のアクティビティ履歴
# ==========================================
with tab_history:
    if st.session_state.metrics and not st.session_state.metrics["activities_df"].empty:
        st.dataframe(st.session_state.metrics["activities_df"], use_container_width=True)
    else:
        st.info("データがありません。上部の「🔄 最新データを再同期」を押してください。")

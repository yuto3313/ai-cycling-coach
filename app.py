import streamlit as st
import requests
import pandas as pd
import json
import os
import re
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
        """個別ライドのインターバル(ラップ)詳細・ゾーン滞在時間を取得"""
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

    # 新しい日付順（降順）にソート
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
            "デカップリング(%)": decoupling if decoupling is not None else "-",
        })

        raw_rides.append({
            "id": act.get("id"),
            "date": act.get("start_date_local", "")[:10],
            "name": act.get("name", "Ride"),
            "duration_min": moving_time_min,
            "distance_km": dist_km,
            "elevation_m": elev_m,
            "tss": round(act.get("icu_training_load") or 0),
            "np": np_w,
            "avg_w": avg_w,
            "max_w": round(act.get("max_watts") or 0),
            "if": round((act.get("icu_intensity") or 0) / 100, 2) if act.get("icu_intensity") else 0.0,
            "vi": vi,
            "avg_hr": round(act.get("average_heartrate") or 0),
            "max_hr": round(act.get("max_heartrate") or 0),
            "cadence": round(act.get("average_cadence") or 0),
            "decoupling": decoupling,
            "ef": ef,
            "zone_times": act.get("icu_zone_times"),
            "feel": act.get("feel"),
            "rpe": act.get("icu_rpe"),
        })

    pdc_info = extract_pdc_metrics(pdc_raw, weight, ftp)
    if not pdc_info["eftp"] and eftp_from_act:
        pdc_info["eftp"] = eftp_from_act

    return {
        "ctl": latest_ctl,
        "atl": latest_atl,
        "tsb": latest_tsb,
        "weight": latest_weight,
        "activities_df": pd.DataFrame(act_list),
        "raw_rides": raw_rides,
        "pdc": pdc_info
    }


def parse_activity_intervals(detail_json: dict, ftp: int) -> pd.DataFrame:
    """個別アクティビティのラップ(インターバル)一覧をDataFrame化"""
    intervals = detail_json.get("icu_intervals") or []
    rows = []
    for idx, iv in enumerate(intervals, start=1):
        dur_sec = iv.get("moving_time") or iv.get("elapsed_time") or 0
        if dur_sec < 10:
            continue
        mins, secs = divmod(int(dur_sec), 60)
        avg_w = round(iv.get("average_watts") or 0)
        pct_ftp = round((avg_w / ftp) * 100) if ftp > 0 else 0
        rows.append({
            "#": idx,
            "種別": iv.get("type", ""),
            "ラベル": iv.get("label") or f"Lap {idx}",
            "時間": f"{mins}:{secs:02d}",
            "平均パワー(W)": avg_w,
            "FTP比(%)": f"{pct_ftp}%",
            "平均心拍(bpm)": round(iv.get("average_heartrate") or 0),
            "最大心拍(bpm)": round(iv.get("max_heartrate") or 0),
            "ケイデンス(rpm)": round(iv.get("average_cadence") or 0),
        })
    return pd.DataFrame(rows)


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
            act_summary = metrics["activities_df"].head(15).to_markdown(index=False)
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

    return f"""あなたはプロレベルの科学的知見を持つAIサイクリングコーチ（AImoth型システム）です。
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

【直近のライド履歴（最新順）】
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


# ==========================================
# 4. Streamlit UI 構築
# ==========================================
icon_setting = "🚴‍♂️"
for fname in ["icon.png", "icon.jpg", "icon.jpeg", "icon.png.jpg", "icon.png.jpeg"]:
    if os.path.exists(fname):
        icon_setting = fname
        break
st.set_page_config(
    page_title="AI Cycling Coach",
    page_icon=icon_setting,
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
if "activity_reviews" not in st.session_state:
    st.session_state.activity_reviews = {}
if "activity_details_cache" not in st.session_state:
    st.session_state.activity_details_cache = {}

# --- サイドバー：設定入力 ＆ 自動保存 ---
with st.sidebar:
    st.header("⚙️ 設定 & 自動保存")
    athlete_id = st.text_input("Intervals Athlete ID", value=cfg["athlete_id"], placeholder="例: i123456")
    intervals_key = st.text_input("Intervals API Key", value=cfg["intervals_key"], type="password")
    gemini_key = st.text_input("Gemini API Key", value=cfg["gemini_key"], type="password")

    st.divider()
    st.subheader("⚡ フィジカル & 目標レース")
    ftp = st.number_input("現在のFTP (W)", 100, 500, int(cfg["ftp"]), 5)
    weight = st.number_input("現在の体重 (kg)", 35.0, 120.0, float(cfg["weight"]), 0.5)
    target_ftp = st.number_input("目標FTP (W)", 100, 550, int(cfg["target_ftp"]), 5)

    race_name = st.text_input("目標レース名", value=cfg["race_name"])
    try:
        default_rdate = datetime.strptime(cfg["race_date"], "%Y-%m-%d").date()
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
    idx_race = race_options.index(cfg["race_type"]) if cfg["race_type"] in race_options else 0
    race_type = st.selectbox("レース特性", race_options, index=idx_race)
    race_goal = st.text_area("目標タイム・課題メモ", value=cfg["race_goal"])

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        weekday_hours = st.number_input("平日枠(h/日)", 0.5, 5.0, float(cfg["weekday_hours"]), 0.5)
    with col_w2:
        weekend_hours = st.number_input("休日枠(h/日)", 1.0, 10.0, float(cfg["weekend_hours"]), 0.5)

    current_cfg_dict = {
        "athlete_id": athlete_id,
        "intervals_key": intervals_key,
        "gemini_key": gemini_key,
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
        st.success("設定を保存しました！")

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

# --- データ同期関数 ---
def sync_intervals_now(show_msg=True):
    if not athlete_id or not intervals_key:
        if show_msg:
            st.warning("左上の「＞」からサイドバーを開き、Athlete IDとAPI Keyを入力してください。")
        return
    try:
        with st.spinner("Intervals.icuから最新データ＆パワーカーブを取得中..."):
            save_config(current_cfg_dict)
            client = IntervalsClient(athlete_id, intervals_key)
            wellness = client.get_wellness(days=30)
            activities = client.get_activities(days=30)
            pdc_raw = client.get_power_curves(curves="42d")
            st.session_state.metrics = parse_intervals_data(wellness, activities, pdc_raw, weight, ftp)
        if show_msg:
            st.toast("✅ Intervals.icu 同期完了！")
    except Exception as e:
        st.error(f"同期エラー: {e}")

if st.session_state.metrics is None and athlete_id and intervals_key:
    sync_intervals_now(show_msg=False)

# --- メイン画面上部：ダッシュボード指標 ---
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    days_to_race = (race_date - datetime.now().date()).days
    st.markdown(f"**🏆 目標: {race_name}** （残り **{days_to_race}日** / {determine_training_phase(days_to_race)}）")
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

if st.session_state.metrics and "pdc" in st.session_state.metrics:
    pdc = st.session_state.metrics["pdc"]
    with st.expander("📈 直近42日のパワーカーブ (PDC) & 弱点プロファイリング", expanded=False):
        st.info(pdc["diagnosis"])
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("15秒", f"{pdc['watts']['15s']} W", f"{pdc['wkg']['15s']} W/kg")
        pc2.metric("1分", f"{pdc['watts']['1m']} W", f"{pdc['wkg']['1m']} W/kg")
        pc3.metric("5分", f"{pdc['watts']['5m']} W", f"{pdc['wkg']['5m']} W/kg")
        pc4.metric("20分", f"{pdc['watts']['20m']} W", f"{pdc['wkg']['20m']} W/kg")
        pc5.metric("60分 / eFTP", f"{pdc['watts']['60m']} W", f"eFTP: {pdc['eftp'] or '--'}W")

# --- タブ構成 ---
tab_review, tab_daily, tab_weekly, tab_chat = st.tabs([
    "🔍 直近ライドAI評価",
    "🎯 今日のメニュー生成",
    "📅 週間プラン一括生成",
    "💬 AIコーチ相談"
])

# ==========================================
# Tab 1: 直近ライドAI評価（新機能！）
# ==========================================
with tab_review:
    st.subheader("🔍 直近アクティビティのAI徹底評価 & ラップ解析")
    if not st.session_state.metrics or not st.session_state.metrics.get("raw_rides"):
        st.info("ライドデータがありません。右上の「🔄 最新データを再同期」を押してください。")
    else:
        rides = st.session_state.metrics["raw_rides"]
        ride_options = [
            f"{r['date']} | {r['name']} ({r['duration_min']}分 / TSS:{r['tss']} / NP:{r['np']}W)"
            for r in rides
        ]
        selected_idx = st.selectbox("評価・分析するライドを選択（一番上が最新ライドです）", range(len(ride_options)), format_func=lambda i: ride_options[i])
        target_ride = rides[selected_idx]
        act_id = target_ride["id"]

        # 選択したライドのサマリーメトリクス表示
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("NP / 平均パワー", f"{target_ride['np']} W", f"平均 {target_ride['avg_w']} W")
        r2.metric("IF / TSS", f"{target_ride['if']}", f"TSS: {target_ride['tss']}")
        r3.metric("時間 / 距離", f"{target_ride['duration_min']} 分", f"{target_ride['distance_km']} km / 獲得{target_ride['elevation_m']}m")
        r4.metric("平均 / 最大心拍", f"{target_ride['avg_hr']} bpm", f"Max {target_ride['max_hr']} bpm")
        dec_val = f"{target_ride['decoupling']}%" if target_ride['decoupling'] is not None else "データなし"
        dec_status = "優秀 (<5%)" if (target_ride['decoupling'] is not None and target_ride['decoupling'] < 5.0) else ("タレ注意 (>5%)" if target_ride['decoupling'] is not None else "")
        r5.metric("心拍デカップリング", dec_val, dec_status)
        r6.metric("ケイデンス / VI", f"{target_ride['cadence']} rpm", f"VI: {target_ride['vi']} / EF: {target_ride['ef'] or '--'}")

        # 個別ライドの詳細(ラップ/インターバル)を取得
        df_laps = pd.DataFrame()
        if act_id and athlete_id and intervals_key:
            if act_id not in st.session_state.activity_details_cache:
                try:
                    client = IntervalsClient(athlete_id, intervals_key)
                    st.session_state.activity_details_cache[act_id] = client.get_activity_detail(act_id)
                except Exception:
                    st.session_state.activity_details_cache[act_id] = {}
            detail_data = st.session_state.activity_details_cache.get(act_id, {})
            if detail_data:
                df_laps = parse_activity_intervals(detail_data, ftp)

        if not df_laps.empty:
            with st.expander("⏱️ インターバル / ラップ詳細データを表示", expanded=False):
                st.dataframe(df_laps, use_container_width=True, hide_index=True)

        user_comment = st.text_input("ライドの主観メモ（任意：脚の張り、補給、機材変更、きつかった場面など）", placeholder="例: 3本目から脚が重くなった、向かい風が強かった など")

        if st.button("🤖 このライドをAIコーチに採点・評価してもらう", type="primary", use_container_width=True):
            if not gemini_key:
                st.error("サイドバーで Gemini API Key を設定・保存してください。")
            else:
                sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
                laps_md = df_laps.to_markdown(index=False) if not df_laps.empty else "ラップ詳細なし（全体スタッツから評価してください）"
                review_prompt = f"""
以下の直近のアクティビティデータをプロコーチの視点で徹底評価してください。

【評価対象アクティビティ】
- 日付: {target_ride['date']}
- タイトル: {target_ride['name']}
- 移動時間: {target_ride['duration_min']}分 / 距離: {target_ride['distance_km']}km / 獲得標高: {target_ride['elevation_m']}m
- NP: {target_ride['np']}W (現在FTP {ftp}W に対して IF {target_ride['if']}) / 平均パワー: {target_ride['avg_w']}W / 最大パワー: {target_ride['max_w']}W
- TSS: {target_ride['tss']} / VI (変動指数): {target_ride['vi']} / EF (効率指数): {target_ride['ef']}
- 平均心拍: {target_ride['avg_hr']}bpm / 最大心拍: {target_ride['max_hr']}bpm / 平均ケイデンス: {target_ride['cadence']}rpm
- パワー心拍デカップリング (Pw:Hr): {dec_val}
- アスリート本人の主観メモ: {user_comment or '特になし'}

【インターバル / ラップ推移データ】
{laps_md}

【出力フォーマット】
1. 🏆 **総合評価スコア（100点満点 & S/A/B/C判定）と総評**: このライドが現在のフェーズおよび目標レース（{race_name}）に向けてどのような効果があったか。
2. 📊 **ペーシング・ラップ推移・心拍デカップリングの深掘り分析**:
   - インターバルの各本数の維持具合、VI（ペーシングの一定さ）、ケイデンス、および心拍デカップリング（後半の有酸素タレ具合）から読み取れる具体的な強みと課題。
3. 💡 **次回に向けた改善アクション & 疲労ケア**:
   - 現在のTSB（{st.session_state.metrics['tsb']}）も踏まえ、明日・明後日にどう過ごすべきか、次回の同系統メニューでは何W・何分を狙うべきか具体的に提案してください。
"""
                with st.spinner("AIコーチがラップ推移と心拍・パワーデータを精密解析中..."):
                    client_ai = genai.Client(api_key=gemini_key)
                    response = client_ai.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=review_prompt,
                        config=types.GenerateContentConfig(system_instruction=sys_prompt)
                    )
                    st.session_state.activity_reviews[act_id] = response.text

        if act_id in st.session_state.activity_reviews:
            st.divider()
            st.markdown(st.session_state.activity_reviews[act_id])

        st.divider()
        with st.expander("📋 過去30日間のライド一覧表を見る", expanded=False):
            st.dataframe(st.session_state.metrics["activities_df"], use_container_width=True, hide_index=True)

# ==========================================
# Tab 2: 今日のAIワークアウト生成
# ==========================================
with tab_daily:
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        avail_minutes = st.slider("今日の練習時間 (分)", 30, 240, 75, 15)
    with col_opt2:
        indoor_mode = st.selectbox("環境", ["インドア (Zwift等)", "アウトドア (実走)"])
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
1. 今日のメニューの狙い（現在のTSB、直近ライドの疲労、パワーカーブの弱点診断、目標レースまでの残り日数を踏まえた理由）
2. メニュー詳細（各セクションの目標ワット数 `{ftp}W` 基準の実数値とケイデンス）
3. Intervals.icu カレンダー登録用のワークアウトコードブロック（必ず ```workout から始めて記述すること）
"""
            with st.spinner("AIコーチがメニューを設計中..."):
                client_ai = genai.Client(api_key=gemini_key)
                response = client_ai.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_req,
                    config=types.GenerateContentConfig(system_instruction=sys_prompt)
                )
                st.session_state.generated_workout = response.text

    if st.session_state.generated_workout:
        st.markdown(st.session_state.generated_workout)
        match = re.search(r"```workout\n(.*?)```", st.session_state.generated_workout, re.DOTALL)
        if match:
            workout_code = match.group(1).strip()
            st.divider()
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                w_name = st.text_input("登録名", value=f"AI Coach ({avail_minutes}min)")
            with col_p2:
                w_date = st.date_input("登録日", value=datetime.now().date())

            if st.button("🚀 Intervals.icu カレンダーに登録", use_container_width=True):
                try:
                    client = IntervalsClient(athlete_id, intervals_key)
                    client.post_workout(w_name, workout_code, w_date.strftime("%Y-%m-%d"))
                    st.success(f"✅ {w_date} のカレンダーに『{w_name}』を登録しました！")
                except Exception as e:
                    st.error(f"登録エラー: {e}")

# ==========================================
# Tab 3: 週間プラン（7日分）一括生成 & カレンダー一括登録
# ==========================================
with tab_weekly:
    st.subheader("📅 1週間分の練習メニュー一括設計 ＆ カレンダー同期")
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
            with st.spinner("7日間のメニューとワークアウトコードを構造化生成中（約15〜20秒）..."):
                client_ai = genai.Client(api_key=gemini_key)
                response = client_ai.models.generate_content(
                    model="gemini-2.5-flash",
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

        for day_item in plan["days"]:
            badge = "🛌 休養日" if day_item["is_rest_day"] else f"🚴‍♂️ {day_item['duration_min']}分 / 目標TSS: {day_item['target_tss']}"
            with st.expander(f"**{day_item['date']} ({day_item['day_label']})** — {day_item['title']} 【{badge}】"):
                st.write(day_item["rationale"])
                if not day_item["is_rest_day"] and day_item["workout_code"]:
                    st.code(day_item["workout_code"], language="text")

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
# Tab 4: AIコーチ・チャット相談
# ==========================================
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("例: 5分パワーに対して20分パワーを伸ばすコツは？ / 昨日のライドをどう思う？"):
        if not gemini_key:
            st.error("サイドバーで Gemini API Key を設定・保存してください。")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("分析中..."):
                    sys_prompt = build_system_prompt(profile_data, st.session_state.metrics)
                    client_ai = genai.Client(api_key=gemini_key)
                    history_contents = [
                        types.Content(
                            role="user" if m["role"] == "user" else "model",
                            parts=[types.Part.from_text(text=m["content"])]
                        )
                        for m in st.session_state.messages
                    ]
                    response = client_ai.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=history_contents,
                        config=types.GenerateContentConfig(system_instruction=sys_prompt)
                    )
                    reply = response.text
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

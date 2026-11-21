"""
テニスコート ひび割れ検出・変化追跡アプリ（改良版）
usage: streamlit run tenniscourt_app.py

解析ロジックは crack_core.py にまとめてあります（旧版は tenniscourt_app_old.py）。
"""

import re
from datetime import date, datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import crack_core as cc

# ── 日本語フォント設定（japanize_matplotlib 不要）─────────
_JP_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
]
for _fp in _JP_FONT_CANDIDATES:
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break

# ── 設定 ─────────────────────────────────────────────
ALGO_VERSION = "v2"
DATA_DIR     = Path("crack_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_CSV  = DATA_DIR / "history.csv"
IMAGES_DIR   = DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

HISTORY_COLUMNS = [
    "date", "court", "count", "total_px", "total_area", "density_len", "density_area",
    "mean_width", "length_cm", "density_cm_per_m2", "algo", "image_path", "note",
]

# 感度 → ヒステリシス閾値（雑音レベルの何倍で「ひび」とみなすか）
SENSITIVITY = {
    "低（誤検出を減らす）": (4.5, 3.0),
    "標準": (cc.DEFAULT.th_high, cc.DEFAULT.th_low),
    "高（細いひびも拾う）": (3.0, 1.8),
}


# ── ファイル名から撮影日・地点を推定（例: 10_31(2).jpg → 10/31, 地点2）──
def parse_filename(name: str, year: int | None = None):
    m = re.match(r"^(\d{1,2})_(\d{1,2})(?:\s*\((\d+)\))?", Path(name).stem)
    if not m:
        return None, None
    y = year or date.today().year
    try:
        d = date(y, int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None, None
    return d, f"地点{m.group(3) or 1}"


# ── ひび割れ解析 ───────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=32)
def analyze(image_bytes: bytes, th_high: float, th_low: float, frame_cm: float | None) -> dict:
    p = cc.Params(th_high=th_high, th_low=th_low)
    res = cc.detect_cracks(image_bytes, p, frame_size_cm=frame_cm or None)
    res.pop("response", None)
    return res


def render_figure(res: dict) -> plt.Figure:
    img = cv2.cvtColor(res["img"], cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    title = f'ひび {res["count"]} 本  |  総延長 {res["total_px"]:,} px'
    if "length_cm" in res:
        title += f'（約 {res["length_cm"]:.0f} cm）'
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(img)
    axes[0].set_title("元画像（解析サイズに正規化）")

    overlay = img.copy()
    sk = cv2.dilate(res["skeleton"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    overlay[sk] = [255, 40, 40]
    edge = cv2.morphologyEx(res["mask"], cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    overlay[edge] = [0, 200, 255]
    axes[1].imshow(overlay)
    axes[1].set_title("検出されたひび（赤）／解析範囲（水色）")

    labels = res["labels"]
    cmap = matplotlib.colormaps.get_cmap("tab20")
    color = np.zeros((*labels.shape, 3), np.uint8)
    for i in range(1, labels.max() + 1):
        color[labels == i] = (np.array(cmap((i - 1) % 20)[:3]) * 255).astype(np.uint8)
    mixed = img.copy()
    on = labels > 0
    mixed[on] = (0.3 * img[on] + 0.7 * color[on]).astype(np.uint8)
    axes[2].imshow(mixed)
    axes[2].set_title(f"ひび領域の色分け（{res['count']} 本）")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ── 履歴 CSV ──────────────────────────────────────────
def load_history() -> pd.DataFrame:
    if HISTORY_CSV.exists():
        df = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
        for c in HISTORY_COLUMNS:
            if c not in df.columns:
                df[c] = "v1" if c == "algo" else np.nan
        return df
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_record(court: str, shot_date: date, res: dict, image_bytes: bytes, note: str,
                  fname_hint: str = "") -> pd.DataFrame:
    df = load_history()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = IMAGES_DIR / f"{court}_{shot_date:%Y%m%d}_{stamp}{Path(fname_hint).suffix or '.jpg'}"
    img_path.write_bytes(image_bytes)
    row = {
        "date": pd.Timestamp(shot_date), "court": court,
        "count": res["count"], "total_px": res["total_px"], "total_area": res["total_area"],
        "density_len": res["density_len"], "density_area": res["density_area"],
        "mean_width": res["mean_width"],
        "length_cm": res.get("length_cm", np.nan),
        "density_cm_per_m2": res.get("density_cm_per_m2", np.nan),
        "algo": ALGO_VERSION, "image_path": str(img_path), "note": note,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)
    return df


# ── 推移グラフ ─────────────────────────────────────────
def render_trend(df: pd.DataFrame, court: str):
    sub = df[(df["court"] == court) & (df["algo"] == ALGO_VERSION)].sort_values("date")
    if len(sub) < 2:
        st.info("推移グラフを表示するには、同じコート・同じ解析方式の記録が2件以上必要です。")
        return
    use_cm = sub["density_cm_per_m2"].notna().all()
    items = [
        ("count", "ひびの本数", "本"),
        ("density_cm_per_m2", "延長密度", "cm/m²") if use_cm else ("density_len", "延長密度", "px/100万px²"),
        ("density_area", "面積率", "%"),
        ("mean_width", "平均幅", "px"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"コート「{court}」の変化推移", fontsize=14)
    for ax, (col, title, unit) in zip(axes.flat, items):
        ax.plot(sub["date"], sub[col], marker="o", linewidth=2)
        ax.set_title(f"{title} [{unit}]")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if "note" in sub and sub["note"].astype(str).str.contains("雨|濡").any():
        st.caption("※ 雨上がり・濡れた状態の写真はひびが見えにくく、値が小さく出ます。")


# ── 比較ビュー ─────────────────────────────────────────
def render_comparison(sub: pd.DataFrame, idx_a: int, idx_b: int):
    row_a, row_b = sub.iloc[idx_a], sub.iloc[idx_b]
    cols = st.columns(2)
    for col, row, label in zip(cols, [row_a, row_b], ["比較元", "比較先"]):
        with col:
            st.markdown(f"**{label}**  {pd.Timestamp(row['date']):%Y-%m-%d}")
            if Path(str(row["image_path"])).exists():
                st.image(str(row["image_path"]), use_container_width=True)
            st.caption(f"本数 {int(row['count'])} / 延長 {int(row['total_px']):,} px / 面積率 {row['density_area']:.2f}%")
    d_cnt = int(row_b["count"]) - int(row_a["count"])
    d_px = int(row_b["total_px"]) - int(row_a["total_px"])
    sign = lambda v: "+" if v > 0 else ""
    st.markdown(f"**変化量：** ひび {sign(d_cnt)}{d_cnt} 本 ／ 延長 {sign(d_px)}{d_px:,} px")
    if row_a.get("algo") != row_b.get("algo"):
        st.warning("解析方式（旧版/改良版）が異なる記録どうしの比較です。数値は直接比較できません。")


# ══ パスワード認証 ═════════════════════════════════════
def check_password() -> bool:
    try:
        correct = st.secrets["password"]
    except (KeyError, FileNotFoundError):
        return True  # secrets 未設定ならスキップ（ローカル開発用）

    def _verify():
        st.session_state["auth"] = st.session_state.get("pw_input") == correct

    if st.session_state.get("auth"):
        return True
    st.set_page_config(page_title="ログイン", layout="centered")
    st.title("🔒 ログイン")
    st.text_input("パスワード", type="password", key="pw_input", on_change=_verify)
    if "auth" in st.session_state and not st.session_state["auth"]:
        st.error("パスワードが違います")
    st.stop()


check_password()

# ══ Streamlit UI ══════════════════════════════════════
st.set_page_config(page_title="テニスコート ひび割れ監視", layout="wide")
st.title("🎾 テニスコート ひび割れ監視システム")

with st.sidebar:
    st.header("⚙️ 解析設定")
    sens = st.radio("検出感度", list(SENSITIVITY), index=1,
                    help="同じコートの推移を見るときは、毎回同じ感度で解析してください。")
    th_high, th_low = SENSITIVITY[sens]
    frame_cm = st.number_input("計測枠（緑のひも）の一辺 [cm]", min_value=0.0, value=0.0, step=5.0,
                               help="入力すると、長さを cm・密度を cm/m² で表示します（0 なら px 表示）。")
    st.caption("撮影のコツ：真上から・枠全体が写るように・できれば同じ時間帯（影の向きをそろえる）")

tab_analyze, tab_bulk, tab_history, tab_trend, tab_compare = st.tabs(
    ["📸 解析", "🗂️ 一括登録", "📋 履歴一覧", "📈 推移グラフ", "🔍 比較"])

# ── 解析タブ ──────────────────────────────────────────
with tab_analyze:
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    guess_date, guess_court = parse_filename(uploaded.name) if uploaded else (None, None)
    with col2:
        df_all = load_history()
        courts = sorted(df_all["court"].dropna().unique().tolist()) if not df_all.empty else []
        court_input = st.text_input("コート名", value=guess_court or "",
                                    placeholder="例：地点1、A面", help="同じ名前で記録すると推移が追えます")
        if courts:
            selected = st.selectbox("または既存コートを選択", ["（新規入力）"] + courts)
            if selected != "（新規入力）":
                court_input = selected
        shot_date = st.date_input("撮影日", value=guess_date or date.today(),
                                  help="ファイル名が「10_31.jpg」の形式なら自動で入ります")
        note = st.text_input("メモ（任意）", placeholder="雨後、施工直後 など")

    if uploaded:
        image_bytes = uploaded.getvalue()
        with st.spinner("解析中…"):
            res = analyze(image_bytes, th_high, th_low, frame_cm)

        st.subheader("解析結果")
        fig = render_figure(res)
        st.pyplot(fig)
        plt.close(fig)

        c = st.columns(4)
        c[0].metric("ひびの本数", f"{res['count']} 本")
        if "length_cm" in res:
            c[1].metric("総延長", f"{res['length_cm']:.0f} cm")
            c[2].metric("延長密度", f"{res['density_cm_per_m2']:.0f} cm/m²")
            c[3].metric("平均幅", f"{res['mean_width_mm']:.1f} mm")
        else:
            c[1].metric("総延長", f"{res['total_px']:,} px")
            c[2].metric("延長密度", f"{res['density_len']:.0f} px/100万px²")
            c[3].metric("平均幅", f"{res['mean_width']:.1f} px")
        st.caption(f"元画像 {res['orig_w']}×{res['orig_h']} px → 解析サイズ {res['w']}×{res['h']} px に正規化 ／ "
                   f"面積率 {res['density_area']:.2f}% ／ 解析範囲 {res['roi_area'] / (res['w'] * res['h']) * 100:.0f}%")

        if court_input:
            if st.button("💾 この結果を履歴に保存", type="primary"):
                append_record(court_input, shot_date, res, image_bytes, note, uploaded.name)
                st.success(f"コート「{court_input}」の記録（{shot_date}）を保存しました。")
        else:
            st.info("保存するにはコート名を入力してください。")

# ── 一括登録タブ ───────────────────────────────────────
with tab_bulk:
    st.markdown("ファイル名が `月_日.jpg`（2か所目は `月_日(2).jpg`）の写真をまとめて解析・登録します。")
    year = st.number_input("撮影年", 2000, 2100, date.today().year)
    files = st.file_uploader("画像を複数選択", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bulk")
    if files:
        rows = []
        prog = st.progress(0.0)
        for i, f in enumerate(files):
            d, court = parse_filename(f.name, int(year))
            r = analyze(f.getvalue(), th_high, th_low, frame_cm)
            rows.append(dict(file=f.name, date=d, court=court, count=r["count"], total_px=r["total_px"],
                             density_len=round(r["density_len"]), density_area=round(r["density_area"], 2),
                             _res=r, _bytes=f.getvalue()))
            prog.progress((i + 1) / len(files))
        view = pd.DataFrame(rows).drop(columns=["_res", "_bytes"])
        st.dataframe(view, use_container_width=True)
        bad = view[view["date"].isna()]
        if not bad.empty:
            st.warning(f"ファイル名から日付を読み取れなかった {len(bad)} 件は登録されません。")
        if st.button("💾 まとめて履歴に保存", type="primary"):
            n = 0
            for r in rows:
                if r["date"] is not None:
                    append_record(r["court"], r["date"], r["_res"], r["_bytes"], "一括登録", r["file"])
                    n += 1
            st.success(f"{n} 件を保存しました。")

# ── 履歴タブ ──────────────────────────────────────────
with tab_history:
    df_all = load_history()
    if df_all.empty:
        st.info("まだ記録がありません。解析タブで画像を保存してください。")
    else:
        court_filter = st.selectbox("コートで絞り込み", ["全て"] + sorted(df_all["court"].unique().tolist()), key="hf")
        view = df_all if court_filter == "全て" else df_all[df_all["court"] == court_filter]
        st.dataframe(
            view[["date", "court", "count", "total_px", "density_len", "density_area", "mean_width", "algo", "note"]]
            .rename(columns={"date": "撮影日", "court": "コート", "count": "ひび本数", "total_px": "総延長(px)",
                             "density_len": "延長密度", "density_area": "面積率(%)", "mean_width": "平均幅(px)",
                             "algo": "解析方式", "note": "メモ"})
            .sort_values("撮影日", ascending=False),
            use_container_width=True,
        )
        st.download_button("📥 CSVダウンロード", df_all.to_csv(index=False).encode("utf-8-sig"),
                           "crack_history.csv", "text/csv")

# ── 推移タブ ──────────────────────────────────────────
with tab_trend:
    df_all = load_history()
    if df_all.empty:
        st.info("記録がありません。")
    else:
        court_t = st.selectbox("コートを選択", sorted(df_all["court"].unique().tolist()), key="tt")
        render_trend(df_all, court_t)

# ── 比較タブ ──────────────────────────────────────────
with tab_compare:
    df_all = load_history()
    if df_all.empty or len(df_all) < 2:
        st.info("比較するには同じコートの記録が2件以上必要です。")
    else:
        court_c = st.selectbox("コートを選択", sorted(df_all["court"].unique().tolist()), key="ct")
        sub_c = df_all[df_all["court"] == court_c].sort_values("date").reset_index(drop=True)
        labels_list = [f"{pd.Timestamp(r['date']):%Y-%m-%d}  本数:{int(r['count'])}" for _, r in sub_c.iterrows()]
        if len(labels_list) >= 2:
            col_a, col_b = st.columns(2)
            idx_a = col_a.selectbox("比較元", range(len(labels_list)), format_func=lambda i: labels_list[i])
            idx_b = col_b.selectbox("比較先", range(len(labels_list)), index=len(labels_list) - 1,
                                    format_func=lambda i: labels_list[i])
            if idx_a != idx_b:
                render_comparison(sub_c, idx_a, idx_b)
        else:
            st.info(f"コート「{court_c}」の記録がまだ1件しかありません。")

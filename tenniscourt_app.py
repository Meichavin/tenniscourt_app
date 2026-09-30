"""
テニスコート ひび割れ検出・変化追跡アプリ
usage: streamlit run tennis_crack_app.py
"""

import streamlit as st
import cv2
import numpy as np
from skimage import morphology, measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── 日本語フォント設定（japanize_matplotlib 不要）─────────
# packages.txt に fonts-noto-cjk を記載しておくと自動インストールされる
_JP_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _JP_FONT_CANDIDATES:
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        _prop = font_manager.FontProperties(fname=_fp)
        matplotlib.rcParams["font.family"] = _prop.get_name()
        break

# ── 設定 ─────────────────────────────────────────────
MIN_CRACK_AREA = 200
MORPH_ITER     = 2
DATA_DIR       = Path("crack_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_CSV    = DATA_DIR / "history.csv"
IMAGES_DIR     = DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# ── ひび割れ解析 ───────────────────────────────────────
def detect_cracks(image_bytes: bytes) -> dict:
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w    = img.shape[:2]
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat  = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_bh)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed    = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)

    labels  = measure.label(closed, connectivity=2)
    regions = measure.regionprops(labels)
    valid   = [r for r in regions if int(r.area) >= MIN_CRACK_AREA]
    count   = len(valid)

    skeleton     = morphology.skeletonize(closed.astype(bool))
    total_px     = int(np.sum(skeleton))
    image_area   = h * w
    total_area   = int(sum(r.area for r in valid))
    density_len  = total_px  / image_area * 1_000_000
    density_area = total_area / image_area * 100

    return dict(
        img=img, closed=closed, labels=labels, valid=valid,
        w=w, h=h, count=count,
        total_px=total_px, total_area=total_area,
        density_len=density_len, density_area=density_area,
    )

def render_figure(res: dict) -> plt.Figure:
    img, closed, labels, valid, count, total_px = (
        res["img"], res["closed"], res["labels"], res["valid"],
        res["count"], res["total_px"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'ひびの本数: {count} 本  |  総延長: {total_px:,} px', fontsize=13)

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('元画像'); axes[0].axis('off')

    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    overlay[closed > 0] = [255, 50, 50]
    axes[1].imshow(overlay); axes[1].set_title('検出されたひび（赤）'); axes[1].axis('off')

    cmap  = matplotlib.colormaps.get_cmap('tab20')
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i, r in enumerate(valid):
        c = (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
        color[labels == r.label] = c
    mixed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.5, color, 0.8, 0)
    axes[2].imshow(mixed); axes[2].set_title(f'色分け（{count} 本）'); axes[2].axis('off')

    plt.tight_layout()
    return fig

# ── 履歴 CSV ──────────────────────────────────────────
def load_history() -> pd.DataFrame:
    if HISTORY_CSV.exists():
        return pd.read_csv(HISTORY_CSV, parse_dates=["date"])
    return pd.DataFrame(columns=[
        "date", "court", "count", "total_px",
        "total_area", "density_len", "density_area", "image_path", "note"
    ])

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_CSV, index=False)

def append_record(court: str, res: dict, image_bytes: bytes, note: str) -> pd.DataFrame:
    df = load_history()
    now = datetime.now()
    fname = f"{court}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = IMAGES_DIR / fname
    img_path.write_bytes(image_bytes)

    new_row = pd.DataFrame([{
        "date": now,
        "court": court,
        "count": res["count"],
        "total_px": res["total_px"],
        "total_area": res["total_area"],
        "density_len": res["density_len"],
        "density_area": res["density_area"],
        "image_path": str(img_path),
        "note": note,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_history(df)
    return df

# ── 推移グラフ ─────────────────────────────────────────
def render_trend(df: pd.DataFrame, court: str):
    sub = df[df["court"] == court].sort_values("date").copy()
    if len(sub) < 2:
        st.info("推移グラフを表示するには同じコートの記録が2件以上必要です。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f'コート「{court}」の変化推移', fontsize=14)

    dates = sub["date"]
    for ax, col, title, unit in zip(
        axes.flat,
        ["count", "total_px", "density_len", "density_area"],
        ["ひびの本数", "総延長", "延長密度", "面積密度"],
        ["本", "px", "px/100万px²", "%"],
    ):
        ax.plot(dates, sub[col], marker="o", linewidth=2)
        ax.set_title(f"{title} [{unit}]")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── 比較ビュー ─────────────────────────────────────────
def render_comparison(sub: pd.DataFrame, court: str, idx_a: int, idx_b: int):
    row_a = sub.iloc[idx_a]
    row_b = sub.iloc[idx_b]

    cols = st.columns(2)
    for col, row, label in zip(cols, [row_a, row_b], ["比較元（古い）", "比較先（新しい）"]):
        with col:
            st.markdown(f"**{label}**  {pd.Timestamp(row['date']).strftime('%Y-%m-%d %H:%M')}")
            if Path(row["image_path"]).exists():
                st.image(str(row["image_path"]), use_container_width=True)
            st.caption(f"本数 {int(row['count'])} / 延長 {int(row['total_px']):,} px / 面積密度 {row['density_area']:.3f}%")

    delta_count = int(row_b["count"]) - int(row_a["count"])
    delta_px    = int(row_b["total_px"]) - int(row_a["total_px"])
    sign = lambda v: ("+" if v > 0 else "")
    st.markdown(
        f"**変化量：** ひび {sign(delta_count)}{delta_count} 本 ／ "
        f"延長 {sign(delta_px)}{delta_px:,} px"
    )

# ══ Streamlit UI ══════════════════════════════════════
st.set_page_config(page_title="テニスコート ひび割れ監視", layout="wide")
st.title("テニスコート ひび割れ監視システム")

tab_analyze, tab_history, tab_trend, tab_compare = st.tabs([
    "📸 解析", "📋 履歴一覧", "📈 推移グラフ", "🔍 比較"
])

# ── 解析タブ ──────────────────────────────────────────
with tab_analyze:
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
    with col2:
        df_all   = load_history()
        courts   = sorted(df_all["court"].dropna().unique().tolist()) if not df_all.empty else []
        court_input = st.text_input(
            "コート名",
            placeholder="例：A面、B面、第1コート",
            help="同じ名前で記録すると推移が追えます",
        )
        if courts:
            selected = st.selectbox("または既存コートを選択", ["（新規入力）"] + courts)
            if selected != "（新規入力）":
                court_input = selected
        note = st.text_input("メモ（任意）", placeholder="雨後、施工直後 など")

    if uploaded:
        image_bytes = uploaded.read()
        with st.spinner("解析中…"):
            res = detect_cracks(image_bytes)

        st.subheader("解析結果")
        fig = render_figure(res)
        st.pyplot(fig)
        plt.close(fig)

        st.success(f"""
        - 画像サイズ: {res['w']} × {res['h']} px
        - ひびの本数: **{res['count']} 本**
        - ひびの総延長: **{res['total_px']:,} px**
        - ひびの総面積: **{res['total_area']:,} px²**
        - 延長密度: {res['density_len']:.1f} px / 100万px²
        - 面積密度: {res['density_area']:.3f} %
        """)

        if court_input:
            if st.button("💾 この結果を履歴に保存", type="primary"):
                append_record(court_input, res, image_bytes, note)
                st.success(f"コート「{court_input}」の記録を保存しました。")
        else:
            st.info("保存するにはコート名を入力してください。")

# ── 履歴タブ ──────────────────────────────────────────
with tab_history:
    df_all = load_history()
    if df_all.empty:
        st.info("まだ記録がありません。解析タブで画像を保存してください。")
    else:
        court_filter = st.selectbox("コートで絞り込み", ["全て"] + sorted(df_all["court"].unique().tolist()), key="hf")
        view = df_all if court_filter == "全て" else df_all[df_all["court"] == court_filter]
        st.dataframe(
            view[["date", "court", "count", "total_px", "density_area", "note"]]
            .rename(columns={
                "date": "日時", "court": "コート", "count": "ひび本数",
                "total_px": "総延長(px)", "density_area": "面積密度(%)", "note": "メモ",
            })
            .sort_values("日時", ascending=False),
            use_container_width=True,
        )
        csv_bytes = df_all.to_csv(index=False).encode()
        st.download_button("📥 CSVダウンロード", csv_bytes, "crack_history.csv", "text/csv")

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
        sub_c   = df_all[df_all["court"] == court_c].reset_index(drop=True)
        labels_list = [
            f"{pd.Timestamp(r['date']).strftime('%Y-%m-%d %H:%M')}  本数:{int(r['count'])}"
            for _, r in sub_c.iterrows()
        ]
        if len(labels_list) >= 2:
            col_a, col_b = st.columns(2)
            idx_a = col_a.selectbox("比較元", range(len(labels_list)), format_func=lambda i: labels_list[i])
            idx_b = col_b.selectbox("比較先", range(len(labels_list)), index=len(labels_list)-1, format_func=lambda i: labels_list[i])
            if idx_a != idx_b:
                render_comparison(sub_c, court_c, idx_a, idx_b)
        else:
            st.info(f"コート「{court_c}」の記録がまだ1件しかありません。")

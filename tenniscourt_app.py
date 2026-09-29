import streamlit as st
import cv2
import numpy as np
from skimage import morphology, measure
import matplotlib.pyplot as plt
import japanize_matplotlib

MIN_CRACK_AREA = 200
MORPH_ITER     = 2

def detect_cracks(image_bytes):
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

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'ひびの本数: {count} 本  |  総延長: {total_px:,} px', fontsize=13)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title('元画像'); axes[0].axis('off')
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    overlay[closed > 0] = [255, 50, 50]
    axes[1].imshow(overlay); axes[1].set_title('検出されたひび（赤）'); axes[1].axis('off')
    cmap  = plt.cm.get_cmap('tab20', max(count, 1))
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i, r in enumerate(valid):
        c = (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
        color[labels == r.label] = c
    mixed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.5, color, 0.8, 0)
    axes[2].imshow(mixed); axes[2].set_title(f'色分け（{count} 本）'); axes[2].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    # 結果テキスト
    st.success(f"""
    - 画像サイズ: {w} x {h} px
    - ひびの本数: {count} 本
    - ひびの総延長: {total_px:,} px
    - ひびの総面積: {total_area:,} px²
    - 延長密度: {density_len:.1f} px / 100万px²
    - 面積密度: {density_area:.3f} %
    """)

# ── Streamlit UI ──────────────────────────────
st.title('🎾 テニスコート ひび割れ検出')
uploaded = st.file_uploader('画像をアップロード', type=['jpg', 'jpeg', 'png'])
if uploaded:
    detect_cracks(uploaded.read())
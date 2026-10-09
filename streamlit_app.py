"""Hanuman pose scanner - Streamlit version (camera + upload + popup)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hanuman_model.keras")
LORE_PATH = os.path.join(HERE, "lore.json")

IMG = 224
NOISE_SIGMA_MIN = 3.0
LOW_CONF = 0.50

# ลำดับต้องเหมือน CLASS_NAMES ตอนเทรน
CLASS_NAMES = [
    "hanuman_carry_mountain",
    "hanuman_flag",
    "hanuman_yawn_star_moon",
    "hanuman_suphanna_matcha",
    "hanuman_hold_pavilion",
]

with open(LORE_PATH, encoding="utf-8") as f:
    LORE = json.load(f)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def get_model():
    import keras

    return keras.saving.load_model(MODEL_PATH, compile=False)


# ---------------- Preprocess (เหมือนตอนเทรน) ----------------
def pil_to_rgb(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return np.array(im.convert("RGB"))


def fit_long_side(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def estimate_noise(gray):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.abs(conv).sum() * np.sqrt(np.pi / 2) / (6.0 * (w - 2) * (h - 2)))


def denoise_adaptive(img):
    sigma = estimate_noise(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    if sigma < NOISE_SIGMA_MIN:
        return img
    h = float(np.clip(sigma, 3, 12))
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def fix_light(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    mean = max(float(L.mean()) / 255.0, 1e-3)
    if mean < 0.35 or mean > 0.65:
        gamma = float(np.clip(np.log(0.5) / np.log(mean), 0.4, 2.0))
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
        L = cv2.LUT(L, lut)
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)


def pad_to_square(img, size, color=0):
    h, w = img.shape[:2]
    canvas = np.full((size, size, 3), color, np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def preprocess(img):
    x = fit_long_side(img, IMG)
    x = denoise_adaptive(x)
    x = fix_light(x)
    return pad_to_square(x, IMG, 0)



# ---------------- Popup ----------------
@st.dialog("ผลการสแกน", width="large")
def show_result(res):
    info = LORE[res["key"]]
    pct = res["probs"][res["top"]] * 100
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(res["image"], use_container_width=True)
    with c2:
        st.subheader(f"{info.get('emoji', '')} {info['thai_name']}")
        st.metric("ตรงกับโมเดล", f"{pct:.1f}%")
    if res["probs"][res["top"]] < LOW_CONF:
        st.warning(
            "โมเดลไม่ค่อยมั่นใจ (ต่ำกว่า 50%) ภาพนี้อาจไม่ใช่หนึ่งในห้าท่านี้ "
            "หรือภาพไม่ชัด ลองถ่ายใหม่ให้เห็นฉาก/วัตถุหลักครบ"
        )
    st.markdown("#### 📜 ที่มา")
    st.write(info["story"])
    st.markdown("#### 🙏 ความเชื่อ")
    for b in info["beliefs"]:
        st.markdown(f"- {b}")
    if info.get("note"):
        st.caption(f"หมายเหตุ: {info['note']}")
    st.markdown("#### ความน่าจะเป็นของทุกท่า")
    for i, k in enumerate(CLASS_NAMES):
        p = float(res["probs"][i])
        st.write(f"{LORE[k]['thai_name']} — {p * 100:.1f}%")
        st.progress(min(max(p, 0.0), 1.0))
    st.caption(
        "เปอร์เซ็นต์คือความน่าจะเป็นที่โมเดลให้ ไม่ใช่ความแม่นยำรวม "
        "และโมเดลถูกสอนให้แยกเฉพาะ 5 ท่านี้ ถ้าส่งภาพอื่นก็จะเลือกท่าที่ใกล้ที่สุดเสมอ"
    )


# ---------------- UI ----------------
st.set_page_config(page_title="สแกนท่าหนุมาน", page_icon="🐒")
st.title("🐒 สแกนท่าหนุมาน")
st.write(
    "ถ่ายรูปจากกล้อง หรืออัปโหลดภาพจิตรกรรม/ประติมากรรม "
    "แล้วให้ AI ทายว่าเป็นท่าไหน พร้อมที่มาและความเชื่อของไทย"
)

mode = st.radio("เลือกวิธีใส่ภาพ", ["📷 ถ่ายรูป", "🖼️ อัปโหลด"], horizontal=True)
if mode.startswith("📷"):
    file = st.camera_input("ถ่ายรูปท่าหนุมาน")
else:
    file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png", "webp", "bmp"])

if file is not None:
    if st.button("🔍 สแกนภาพ", type="primary", use_container_width=True):
        pil = Image.open(file)
        rgb = pil_to_rgb(pil)
        proc = preprocess(rgb)
        x = np.expand_dims(proc.astype(np.float32), 0)  # 0-255 ตามตอนเทรน
        with st.spinner("กำลังวิเคราะห์..."):
            probs = get_model().predict(x, verbose=0)[0]
        top = int(np.argmax(probs))
        thumb = Image.fromarray(rgb)
        thumb.thumbnail((400, 400))
        st.session_state["result"] = {
            "probs": probs,
            "top": top,
            "key": CLASS_NAMES[top],
            "image": thumb,
        }
        show_result(st.session_state["result"])

res = st.session_state.get("result")
if res:
    info = LORE[res["key"]]
    st.success(f"ผลล่าสุด: {info['thai_name']} — {res['probs'][res['top']] * 100:.1f}%")
    if st.button("เปิดดูที่มาและความเชื่ออีกครั้ง"):
        show_result(res)

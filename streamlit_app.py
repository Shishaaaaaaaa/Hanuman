"""HANUMATCH - Hanuman pose scanner (Streamlit) with custom camera UI."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import base64
import html
import io
import json
import shutil
import urllib.request

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [  # ลำดับไม่มีผลต่อผล (เฉลี่ยความน่าจะเป็นเท่ากัน)
    os.path.join(HERE, "hanuman_efficientnet.keras"),
    os.path.join(HERE, "hanuman_convnext.keras"),
]

# ไฟล์ที่ใหญ่เกิน 100 MB เก็บไว้ที่ GitHub Releases แล้วให้แอปดาวน์โหลดตอนเริ่มทำงาน (ไม่ต้องอยู่ใน repo)
# แก้ USER/REPO/TAG ให้ตรงกับของจริงหลังสร้าง Release
MODEL_URLS = {
    "hanuman_convnext.keras": "https://github.com/shishaaaaaaaa/hanuman/releases/download/models-v1/hanuman_convnext.keras",
}
LORE_PATH = os.path.join(HERE, "lore.json")
ASSETS = os.path.join(HERE, "assets")

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


def b64_file(name: str) -> str:
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


LOGO_B64 = b64_file("logo.jpg")
MARK_B64 = b64_file("mark.jpg")
PATTERN_B64 = b64_file("pattern.jpg")


class SoftVotingEnsemble:
    """รวมหลายโมเดลด้วย Soft Voting (เฉลี่ยความน่าจะเป็น)
    ใช้ .predict(x, verbose=0) เหมือนโมเดลเดี่ยว โค้ดส่วนอื่นจึงไม่ต้องแก้"""

    def __init__(self, models):
        self.models = models

    def predict(self, x, verbose=0):
        return np.mean([m.predict(x, verbose=verbose) for m in self.models], axis=0)


def ensure_file(path: str):
    """ถ้ายังไม่มีไฟล์ในเครื่องและมี URL ให้ดาวน์โหลดมาไว้ข้างๆ แอป (โหลดครั้งเดียวต่อการรีบูต)"""
    url = MODEL_URLS.get(os.path.basename(path))
    if os.path.exists(path) or not url:
        return
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "hanumatch-app"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if os.path.getsize(tmp) < 1_000_000:  # เล็กผิดปกติ = ไม่ใช่ไฟล์โมเดล (เช่น หน้า error)
        os.remove(tmp)
        raise RuntimeError(f"ดาวน์โหลดโมเดลไม่สำเร็จ: {url}")
    os.replace(tmp, path)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล (ครั้งแรกอาจใช้เวลาสักครู่)...")
def get_model():
    import keras

    for p in MODEL_PATHS:
        ensure_file(p)
    return SoftVotingEnsemble([keras.saving.load_model(p, compile=False) for p in MODEL_PATHS])


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


def data_url_to_pil(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def analyze(pil: Image.Image) -> dict:
    rgb = pil_to_rgb(pil)
    proc = preprocess(rgb)
    x = np.expand_dims(proc.astype(np.float32), 0)  # 0-255 ตามตอนเทรน
    probs = get_model().predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    thumb = Image.fromarray(rgb)
    thumb.thumbnail((360, 360))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=85)
    return {
        "probs": probs,
        "top": top,
        "key": CLASS_NAMES[top],
        "thumb": base64.b64encode(buf.getvalue()).decode(),
    }


# ---------------- Theme (HANUMATCH) ----------------
# Monk Robe #CC6621 | Lao Red Wood #662E26 | Temple Gold #D9B658 | Frangipani Cream #F2F0E1
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Thai:wght@400;600;700&family=Noto+Serif:wght@400;600;700&display=swap');
:root{--robe:#CC6621;--wood:#662E26;--gold:#D9B658;--cream:#F2F0E1;--pat:url(data:image/jpeg;base64,__PAT__)}
.stApp,.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp li,.stApp button,.stApp label,
div[role="dialog"] p,div[role="dialog"] h2,div[role="dialog"] li,div[role="dialog"] summary{
  font-family:'Noto Serif Thai','Noto Serif',serif}
header[data-testid="stHeader"]{display:none}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:460px;padding:1.2rem 1rem 2rem}
.stButton button{background:var(--robe)!important;color:var(--cream)!important;border:none!important;
  border-radius:999px!important;font-weight:600;min-height:2.9rem;box-shadow:0 5px 12px rgba(102,46,38,.25)}
.stButton button:hover{background:var(--gold)!important;color:var(--wood)!important}
.stButton button p{color:inherit!important}
button[data-testid="stBaseButton-secondary"]{background:transparent!important;color:var(--gold)!important;
  border:1.5px solid var(--gold)!important;box-shadow:none!important}
button[data-testid="stBaseButton-secondary"]:hover{background:var(--gold)!important;color:var(--wood)!important}

/* ---- Popup (ผลการวิเคราะห์) ---- */
div[role="dialog"]{background:var(--wood) var(--pat) center/300px!important;border:3px solid var(--gold)!important;
  border-radius:26px!important}
div[role="dialog"] h2{color:var(--cream)!important;text-align:center}
div[role="dialog"] button{color:var(--gold)!important}
.hm-emb{display:block;width:52px;height:52px;border-radius:50%;margin:0 auto 10px;border:2px solid var(--gold);object-fit:cover}
.hm-card{background:var(--cream);color:#3a2218;border-radius:22px;padding:18px 16px 14px;
  box-shadow:0 8px 24px rgba(0,0,0,.35)}
.hm-title{text-align:center;color:var(--wood);font-size:1.3rem;line-height:1.35;font-weight:700}
.hm-title small{display:block;font-size:.9rem;font-weight:400}
.hm-ring{width:176px;height:176px;border-radius:50%;margin:14px auto 10px;border:7px solid var(--robe);
  overflow:hidden;background:#fff;box-shadow:0 4px 12px rgba(0,0,0,.2)}
.hm-ring img{width:100%;height:100%;object-fit:cover;display:block}
.hm-conf{text-align:center;font-weight:600;color:var(--wood);margin-bottom:6px}
.hm-track{height:14px;background:#d8d2c0;border-radius:99px;overflow:hidden;margin-bottom:12px}
.hm-track i{display:block;height:100%;background:linear-gradient(90deg,var(--robe),var(--gold));border-radius:99px}
.hm-warn{background:#fff0d6;border:1px solid #e0a040;border-radius:12px;padding:8px 12px;font-size:.88rem;margin-bottom:10px}
.hm-box{background:rgba(102,46,38,.07);border-radius:14px;padding:10px 12px;margin-bottom:10px}
.hm-bt{font-weight:700;color:var(--wood);margin-bottom:4px}
.hm-bs{max-height:150px;overflow-y:auto;font-size:.92rem;line-height:1.65;padding-right:4px}
.hm-bs ul{margin:0;padding-left:18px}
.hm-note{margin-top:6px;font-size:.8rem;color:#7a6652}
.hm-card details{margin-top:4px;font-size:.88rem}
.hm-card summary{cursor:pointer;color:var(--wood);font-weight:600}
.hm-row{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:.82rem}
.hm-lbl{width:46%;flex:none}
.hm-bar{flex:1;height:10px;background:#d8d2c0;border-radius:6px;overflow:hidden}
.hm-bar i{display:block;height:100%;background:var(--gold)}
.hm-row.top .hm-bar i{background:var(--robe)}
.hm-val{width:46px;text-align:right;flex:none}
.hm-foot{margin-top:10px;font-size:.76rem;color:#7a6652;line-height:1.5}
</style>
""".replace("__PAT__", PATTERN_B64)

HOME_CSS = """
<style>
.stApp{background:linear-gradient(rgba(252,251,246,.93),rgba(252,251,246,.93)),var(--pat) center/360px}
.hm-home{text-align:center;padding-top:4vh}
.hm-home img{width:64%;max-width:260px;mix-blend-mode:multiply}
.hm-home h2{color:var(--wood);font-size:1.9rem;margin:.2rem 0 0;font-weight:700}
.hm-home p{color:#8a6a4a;margin:.2rem 0 1.6rem;font-size:1rem}
</style>
"""

CAMERA_CSS = """
<style>
.stApp{background:#120a08}
.hm-last{color:var(--cream);text-align:center;font-size:.92rem;margin:.6rem 0 .2rem}
</style>
"""

st.set_page_config(page_title="HANUMATCH - สแกนท่าหนุมาน", page_icon="🐒", layout="centered")
st.markdown(BASE_CSS, unsafe_allow_html=True)

camera = components.declare_component(
    "hanumatch_camera", path=os.path.join(HERE, "camera_component")
)


# ---------------- Popup ----------------
def result_card_html(res: dict) -> str:
    esc = html.escape
    info = LORE[res["key"]]
    probs = res["probs"]
    pct = float(probs[res["top"]]) * 100
    warn = ""
    if probs[res["top"]] < LOW_CONF:
        warn = (
            '<div class="hm-warn">⚠️ โมเดลไม่ค่อยมั่นใจ (ต่ำกว่า 50%) ภาพนี้อาจไม่ใช่หนึ่งในห้าท่านี้ '
            "หรือภาพไม่ชัด ลองถ่ายใหม่ให้เห็นท่าหลักครบทั้งองค์</div>"
        )
    beliefs = "".join(f"<li>{esc(b)}</li>" for b in info["beliefs"])
    note = f'<div class="hm-note">หมายเหตุ: {esc(info["note"])}</div>' if info.get("note") else ""
    bars = ""
    for i, k in enumerate(CLASS_NAMES):
        v = float(probs[i]) * 100
        bars += (
            f'<div class="hm-row{" top" if i == res["top"] else ""}">'
            f'<span class="hm-lbl">{esc(LORE[k]["thai_name"])}</span>'
            f'<span class="hm-bar"><i style="width:{v:.1f}%"></i></span>'
            f'<span class="hm-val">{v:.1f}%</span></div>'
        )
    emb = f'<img class="hm-emb" src="data:image/jpeg;base64,{MARK_B64}">' if MARK_B64 else ""
    # หมายเหตุ: ห้ามเว้นบรรทัดว่าง/ย่อหน้าใน HTML เพราะ Markdown จะตีความเป็นโค้ด
    return (
        '<div class="hm-wrap">'
        + emb
        + '<div class="hm-card">'
        + f'<div class="hm-title"><small>ผลการวิเคราะห์:</small>{info.get("emoji", "")} {esc(info["thai_name"])}</div>'
        + f'<div class="hm-ring"><img src="data:image/jpeg;base64,{res["thumb"]}"></div>'
        + f'<div class="hm-conf">ความมั่นใจ: {pct:.1f}%</div>'
        + f'<div class="hm-track"><i style="width:{pct:.1f}%"></i></div>'
        + warn
        + f'<div class="hm-box"><div class="hm-bt">ตำนาน</div><div class="hm-bs">{esc(info["story"])}</div></div>'
        + f'<div class="hm-box"><div class="hm-bt">ความเชื่อ</div><div class="hm-bs"><ul>{beliefs}</ul>{note}</div></div>'
        + f"<details><summary>ความน่าจะเป็นของทุกท่า</summary>{bars}</details>"
        + '<div class="hm-foot">เปอร์เซ็นต์คือความน่าจะเป็นที่โมเดลให้ ไม่ใช่ความแม่นยำรวม '
        + "และโมเดลถูกสอนให้แยกเฉพาะ 5 ท่านี้ ถ้าส่งภาพอื่นก็จะเลือกท่าที่ใกล้ที่สุดเสมอ</div>"
        + "</div></div>"
    )


@st.dialog("ผลการวิเคราะห์", width="large")
def show_result(res: dict):
    st.markdown(result_card_html(res), unsafe_allow_html=True)


# ---------------- Pages ----------------
def go(page: str):
    st.session_state["page"] = page


st.session_state.setdefault("page", "home")
st.session_state.setdefault("result", None)
st.session_state.setdefault("last_nonce", None)

if st.session_state["page"] == "home":
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hm-home">'
        + (f'<img src="data:image/jpeg;base64,{LOGO_B64}" alt="HANUMATCH">' if LOGO_B64 else "<h1>HANUMATCH</h1>")
        + "<h2>สแกนท่ามหาฤทธิ์</h2><p>Learn Thai Legends with AI</p></div>",
        unsafe_allow_html=True,
    )
    st.button("เริ่มต้นการใช้งาน", type="primary", use_container_width=True, on_click=go, args=("camera",))

else:
    st.markdown(CAMERA_CSS, unsafe_allow_html=True)
    val = camera(mark=MARK_B64, key="hm_cam", default=None)

    if isinstance(val, dict) and val.get("image") and val.get("nonce") != st.session_state["last_nonce"]:
        st.session_state["last_nonce"] = val["nonce"]
        try:
            with st.spinner("กำลังวิเคราะห์..."):
                st.session_state["result"] = analyze(data_url_to_pil(val["image"]))
            show_result(st.session_state["result"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"วิเคราะห์ภาพไม่สำเร็จ: {exc}")

    res = st.session_state["result"]
    if res:
        info = LORE[res["key"]]
        st.markdown(
            f'<div class="hm-last">ผลล่าสุด: {html.escape(info["thai_name"])} — {res["probs"][res["top"]] * 100:.1f}%</div>',
            unsafe_allow_html=True,
        )
    c1, c2 = st.columns(2)
    with c1:
        st.button("← หน้าแรก", type="secondary", use_container_width=True, on_click=go, args=("home",))
    with c2:
        if res and st.button("ดูผลล่าสุด", type="secondary", use_container_width=True):
            show_result(res)

    try:
        get_model()  # อุ่นโมเดลไว้ล่วงหน้าให้สแกนครั้งแรกเร็วขึ้น
    except Exception as exc:  # noqa: BLE001  (แสดงข้อความอ่านง่ายแทนหน้า traceback)
        st.error(f"โหลดโมเดลไม่สำเร็จ: {exc}")
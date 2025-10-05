# streamlit_react_app.py
# --------------------------------------------
# TanyaKakGem — Chat Edukasi PCOS (tanpa retrieval, tanpa tools)
# Stabil & sederhana: langsung pakai ChatGoogleGenerativeAI dengan riwayat percakapan.
# --------------------------------------------

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# =========================
# 1) Konfigurasi UI
# =========================
st.set_page_config(page_title="TanyaKakGem — Chat PCOS", page_icon="🩺")
st.title("🩺 TanyaKakGem — Chat Edukasi PCOS")
st.caption("Ingat! TanyaKakGem menyediakan informasi edukatif seputar PCOS, bukan pengganti diagnosis dokter.")

with st.sidebar:
    st.subheader("Tentang")
    st.write(
        "Chat ini fokus pada edukasi PCOS (Polycystic Ovary Syndrome): "
        "gejala, diagnosis, terapi, gaya hidup, kesuburan."
    )
    st.write("Bukan layanan medis darurat. Hubungi tenaga kesehatan bila gawat darurat.")
    st.divider()

    # API Key
    api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    # Tombol reset
    if st.button("🔄 Mulai Ulang", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.caption("Contoh pertanyaan:")
    examples = [
        "Apa saja gejala PCOS yang umum?",
        "Apakah PCOS memengaruhi kesuburan?",
        "Apa peran diet indeks glikemik rendah pada PCOS?",
        "Olahraga apa yang baik untuk PCOS?",
        "Bagaimana pilihan terapi PCOS secara umum?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append(HumanMessage(content=ex))
            st.rerun()

# Hentikan bila belum ada API key
if not api_key:
    st.info("Masukkan Google API Key di sidebar untuk mulai.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# =========================
# 2) Model LLM
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # cepat & cukup untuk edukasi
    temperature=0.3,            # lebih konsisten/aman untuk topik medis
    max_output_tokens=1024
)

# =========================
# 3) Guardrails ringan
# =========================
def is_pcos_topic(text: str) -> bool:
    kw = [
        "pcos", "polycystic", "kista ovarium", "sindrom ovarium polikistik",
        "haid", "menstruasi", "siklus", "ovulasi", "androgen", "insulin",
        "metformin", "hirsutisme", "jerawat", "indeks glikemik", "kesuburan",
        "folikel"
    ]
    t = text.lower()
    return any(k in t for k in kw)

RED_FLAGS = [
    "pendarahan berat", "pingsan", "nyeri perut hebat", "demam tinggi",
    "nyeri dada", "sesak napas", "kehamilan ektopik", "darurat"
]

def ensure_disclaimer(text: str) -> str:
    tag = "informasi edukatif, bukan nasihat medis"
    if tag in text.lower():
        return text
    return text.rstrip() + "\n\n*Disclaimer: informasi edukatif, bukan nasihat medis individual.*"

# =========================
# 4) Prompt sistem (peran)
# =========================
SYSTEM_PROMPT = """Anda adalah asisten edukasi kesehatan yang fokus pada PCOS (Polycystic Ovary Syndrome).
Tujuan:
- Jelaskan topik PCOS dalam bahasa Indonesia yang jelas, empatik, dan ringkas.
- Gunakan pengetahuan medis umum/evidence-based tingkat dasar.
- Berikan opsi tindakan umum (pola makan, olahraga, manajemen berat badan, tidur, manajemen stres).
- Jelaskan kapan harus konsultasi ke dokter/spesialis (red flags).
Batasan:
- Tidak mendiagnosis individu atau mengganti saran dokter.
- Hindari menyebut dosis obat spesifik atau merekomendasikan obat resep.
- Jika pertanyaan di luar PCOS, arahkan kembali ke topik PCOS secara sopan.
Format jawaban:
- Ringkas Inti
- Detail Penting
- Kapan Harus ke Dokter
- Catatan & Disclaimer (wajib): 'Informasi edukatif, bukan nasihat medis individual.'
"""

# Inisialisasi riwayat (pakai BaseMessage langsung agar simpel & kompatibel)
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# =========================
# 5) Render riwayat chat
# =========================
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue  # SystemMessage tidak ditampilkan ke UI
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# =========================
# 6) Input pengguna + alur jawaban
# =========================
user_text = st.chat_input("Tanyakan seputar PCOS (gejala, siklus, terapi, gaya hidup)...")

if user_text:
    # Tampilkan chat user
    with st.chat_message("user"):
        st.markdown(user_text)

    # Guardrails
    lower_q = user_text.lower()
    if any(rf in lower_q for rf in RED_FLAGS):
        st.warning("Gejala yang Anda sebutkan dapat termasuk kegawatdaruratan. Segera cari pertolongan medis.")

    if not is_pcos_topic(user_text):
        st.info("Aku fokus pada topik PCOS. Coba ajukan pertanyaan terkait PCOS ya 🙏")
        # Jika ingin HENTIKAN, uncomment baris di bawah:
        # st.stop()

    # Tambahkan ke riwayat sebagai HumanMessage
    st.session_state.messages.append(HumanMessage(content=user_text))

    # Panggil LLM dengan seluruh riwayat (sudah termasuk SystemMessage pertama)
    try:
        ai = llm.invoke(st.session_state.messages)
        # Pastikan konten ada
        answer = ai.content if hasattr(ai, "content") else str(ai)
    except Exception as e:
        answer = f"Terjadi kesalahan: {e}"

    # Pasang disclaimer fallback
    answer = ensure_disclaimer(answer)

    # Tampilkan & simpan jawaban
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append(AIMessage(content=answer))

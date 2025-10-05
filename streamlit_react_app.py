# Import the necessary libraries
import streamlit as st  # For creating the web app interface
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google Gemini via LangChain
from langgraph.prebuilt import create_react_agent  # For creating a ReAct agent
from langchain_core.messages import HumanMessage, AIMessage  # For message formatting
from langchain_core.prompts import ChatPromptTemplate

def is_pcos_topic(text: str) -> bool:
    kw = [
        "pcos", "polycystic", "kista ovarium", "sindrom ovarium polikistik",
        "haid", "menstruasi", "siklus", "ovulasi", "androgen", "insulin",
        "metformin", "hirsutisme", "jerawat", "indeks glikemik", "kesuburan"
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

# --- 1. Page Configuration and Title ---

# Set the title and a caption for the web page
st.set_page_config(page_title="TanyaKakGem — Chat PCOS Gem", page_icon="🩺")
st.title("🩺 TanyaKakGem — Chat Edukasi PCOS (Polycystic Ovarian Syndrome)")
st.caption("Ingat! TanyaKakGem sekedar menyediakan informasi edukatif seputar PCOS, bukan pengganti diagnosis dokter ya;).")

# --- 2. Sidebar for Settings ---

# Create a sidebar section for app settings using 'with st.sidebar:'
with st.sidebar:
    # Add a subheader to organize the settings
    st.subheader("Tentang")
    st.write("Chatbot ini fokus pada edukasi PCOS: gejala, diagnosis, terapi, gaya hidup.")
    st.write("Bukan layanan medis darurat. Hubungi tenaga kesehatan bila darurat.")
    
    # Create a text input field for the Google AI API Key.
    # 'type="password"' hides the key as the user types it.
    google_api_key = st.text_input("Google AI API Key", type="password")
    
    # Create a button to reset the conversation.
    # 'help' provides a tooltip that appears when hovering over the button.
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

# --- 3. API Key and Agent Initialization ---

# Check if the user has provided an API key.
# If not, display an informational message and stop the app from running further.
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# This block of code handles the creation of the LangGraph agent.
# It's designed to be efficient: it only creates a new agent if one doesn't exist
# or if the user has changed the API key in the sidebar.

# We use `st.session_state` which is Streamlit's way of "remembering" variables
# between user interactions (like sending a message or clicking a button).
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        # Initialize the LLM with the API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.3,
            max_output_tokens=1024
        )
        
        # Create a simple ReAct agent with the LLM
        SYSTEM_PROMPT = """Anda adalah asisten edukasi kesehatan yang fokus pada PCOS (Polycystic Ovary Syndrome).
        Tujuan:
        - Menjelaskan PCOS dalam bahasa Indonesia yang jelas, empatik, dan ringkas.
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
        - Catatan & Disclaimer (wajib): 'Informasi edukatif, bukan nasihat medis individual.'"""

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
        ])

        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=PROMPT
        )
        
        # Store the new key in session state to compare against later.
        st.session_state._last_key = google_api_key
        # Since the key changed, we must clear the old message history.
        st.session_state.pop("messages", None)
    except Exception as e:
        # If the key is invalid, show an error and stop.
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Chat History Management ---

# Initialize the message history (as a list) if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle the reset button click.
if reset_button:
    # If the reset button is clicked, clear the agent and message history from memory.
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    # st.rerun() tells Streamlit to refresh the page from the top.
    st.rerun()

# --- 5. Display Past Messages ---

# Loop through every message currently stored in the session state.
for msg in st.session_state.messages:
    # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
    with st.chat_message(msg["role"]):
        # Display the content of the message using Markdown for nice formatting.
        st.markdown(msg["content"])

# --- 6. Handle User Input and Agent Communication ---

# Create a chat input box at the bottom of the page.
# The user's typed message will be stored in the 'prompt' variable.
prompt = st.chat_input("Tanyakan seputar PCOS (gejala, siklus, kesuburan, terapi, gaya hidup)...")

if prompt:
    lower_q = prompt.lower()
    if any(rf in lower_q for rf in RED_FLAGS):
        st.warning("Gejala yang Anda sebutkan dapat termasuk kegawatdaruratan. Segera cari pertolongan medis atau hubungi layanan gawat darurat.")
    if not is_pcos_topic(prompt):
        st.info("Aku fokus pada topik PCOS. Coba ajukan pertanyaan terkait PCOS ya 🙏")

# Check if the user has entered a message.
if prompt:
    # 1. Add the user's message to our message history list.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Display the user's message on the screen immediately for a responsive feel.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get the assistant's response.
    # Use a 'try...except' block to gracefully handle potential errors (e.g., network issues, API errors).
    try:
    # siapkan messages untuk agent
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # kirim ke agent
        response = st.session_state.agent.invoke({"messages": messages})

        # ambil jawaban
        if "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        else:
            answer = "Maaf, aku belum bisa menghasilkan respons."

    except Exception as e:
        answer = f"Terjadi kesalahan: {e}"

    # ⬇️ SELALU pastikan ada disclaimer
    answer = ensure_disclaimer(answer)

    # 4. Display the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(answer)
    # 5. Add the assistant's response to the message history list.
    st.session_state.messages.append({"role": "assistant", "content": answer})
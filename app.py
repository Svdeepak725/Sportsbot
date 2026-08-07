import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

# The dashboard does not use LangSmith tracing. Disabling it avoids background
# network retries that otherwise delay every chat response.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from graph import graph
from kb_manager import add_uploaded_file_to_knowledge_base, init_knowledge_base_state

st.set_page_config(page_title="Sports AI Bot Dashboard", layout="wide")

HISTORY_FILE = Path("chat_history.json")
WELCOME_MESSAGE = {
    "role": "bot",
    "text": "Hello! 👋\nI'm your Sports AI Bot. Ask me anything about sports!",
}


def safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.stop()
        except Exception:
            pass


def load_history() -> list:
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
    except Exception:
        return []
    return []


def save_history(history: list) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file_handle:
            json.dump(history, file_handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def format_message(message: dict) -> str:
    role = message.get("role", "bot")
    text = message.get("text", "")
    if role == "user":
        return f"<div class='message-row message-user'><div class='message-avatar user-avatar'>⚽</div><div class='message-card msg-user'><div class='message-body'>{text}</div></div><div class='message-time'>3:01 PM</div></div>"
    return f"<div class='message-row message-bot'><div class='message-avatar bot-avatar'>⚽</div><div class='message-card msg-bot'><div class='message-body'>{text}</div></div><div class='message-time'>3:01 PM</div></div>"


def run_model(question: str) -> str:
    try:
        result = graph.invoke({"question": question}, config={"run_name": "sports_query"})
        if isinstance(result, dict) and "answer" in result:
            return result["answer"]
        return str(result)
    except Exception as exc:
        return f"(error) {exc}"


if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.get("reset_to_welcome", False):
    st.session_state.messages = []
    st.session_state.reset_to_welcome = False

# Initialize knowledge base
init_knowledge_base_state()

with st.sidebar:
    st.markdown("<div class='sidebar-wrap'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='brand-row'>
            <div style='font-size:42px; line-height:1'>⚽</div>
            <div>
                <div class='brand-title'>Sports AI Bot</div>
                <div class='brand-subtitle'>Your smart sports assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+  New Chat", key="new_chat"):
        st.session_state.messages = []
        st.session_state.reset_to_welcome = True
        safe_rerun()

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>CHATS</div>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("<div class='history-box'>No saved chats</div>", unsafe_allow_html=True)
    else:
        for index, chat in enumerate(st.session_state.chat_history):
            title = chat.get("title", f"Chat {index + 1}")
            if st.button(title, key=f"hist_{index}"):
                st.session_state.messages = chat.get("messages", [])
                safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>TOOLS</div>", unsafe_allow_html=True)

    if st.button("Save Chat", key="save_chat"):
        if st.session_state.messages:
            title = next((m.get("text", "")[:40] for m in st.session_state.messages if m.get("role") == "user"), None)
            if not title:
                title = f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            record = {
                "title": title,
                "created_at": datetime.utcnow().isoformat(),
                "messages": st.session_state.messages,
            }
            st.session_state.chat_history.insert(0, record)
            save_history(st.session_state.chat_history)
            st.success("Chat saved")
        else:
            st.info("Nothing to save yet")

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.reset_to_welcome = True
        safe_rerun()

    if st.button("Delete All History", key="delete_history"):
        st.session_state.chat_history = []
        save_history(st.session_state.chat_history)
        safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>TIPS</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='tip-card'>
            💡 Ask me anything about sports, players, matches, stats, or tournaments!
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.checkbox("Dark Mode", key="dark_mode")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Theme-aware colors must be computed after sidebar widgets update session state.
dark_mode = bool(st.session_state.dark_mode)
if dark_mode:
    app_background = "#0f1116"
    card_background = "rgba(20,24,31,0.96)"
    text_color = "#f8fafc"
    muted_color = "#94a3b8"
    input_background = "#111827"
    input_text = "#f9fafb"
    message_user = "#EFD9DA"
    message_bot = "#151924"
    sidebar_bg = "linear-gradient(180deg, #14141a 0%, #1a0f12 100%)"
else:
    app_background = "#ffffff"
    card_background = "rgba(255,255,255,0.98)"
    text_color = "#111827"
    muted_color = "#6b7280"
    input_background = "#ffffff"
    input_text = "#111827"
    message_user = "#F7E0E2"
    message_bot = "#F4F6F8"
    sidebar_bg = "linear-gradient(180deg, #1b0b0e 0%, #2b0f14 100%)"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', Arial, sans-serif;
    }}

    .stApp {{
        background: {app_background};
    }}

    .stApp > header {{
        background: transparent;
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        color: white;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    .sidebar-wrap {{
        padding: 12px 10px 20px 10px;
    }}

    .brand-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 4px;
        margin-bottom: 18px;
    }}

    .brand-title {{
        font-size: 21px;
        font-weight: 800;
        line-height: 1;
        color: white;
    }}

    .brand-subtitle {{
        font-size: 14px;
        color: rgba(255,255,255,0.82);
        margin-top: 6px;
    }}

    .sidebar-section {{
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.10);
    }}

    .sidebar-label {{
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #FF6B6B;
        margin-bottom: 10px;
    }}

    .history-box {{
        padding: 12px;
        border-radius: 12px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        color: white;
        margin-bottom: 10px;
    }}

    .tip-card {{
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(157,242,90,0.30);
        border-radius: 16px;
        padding: 14px;
        color: white;
        line-height: 1.5;
    }}

    .stButton > button {{
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.04) !important;
        color: white !important;
        width: 100%;
        font-weight: 700 !important;
        padding: 0.75rem 1rem !important;
        transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
    }}

    .stButton > button:hover {{
        border-color: rgba(255,255,255,0.24) !important;
        background: rgba(255,255,255,0.08) !important;
        transform: translateY(-1px);
    }}

    .main-shell {{
        position: relative;
        z-index: 1;
        max-width: 1180px;
        margin: 0 auto;
        padding: 0 8px 16px 8px;
    }}

    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        gap: 20px;
        padding: 18px 12px 0 12px;
        border-radius: 0;
        background: {card_background};
        border: none;
        box-shadow: none;
    }}

    .title {{
        font-size: 34px;
        font-weight: 800;
        color: {text_color};
        line-height: 1.05;
    }}

    .title .accent {{
        color: #bb3761;
    }}

    .subtitle {{
        color: {muted_color};
        font-size: 15px;
        margin-top: 6px;
    }}

    .status {{
        background: #eef9ef;
        color: #15803d;
        border-radius: 999px;
        padding: 8px 14px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(34,197,94,0.10);
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 18px 0 24px 0;
    }}

    .metric-card {{
        background: {card_background};
        border-radius: 18px;
        padding: 16px 18px 14px;
        box-shadow: 0 14px 28px rgba(16,24,40,0.08);
        border: 1px solid rgba(15,23,42,0.07);
    }}

    .metric-label {{
        font-size: 12px;
        font-weight: 800;
        color: {muted_color};
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    .metric-value {{
        font-size: 32px;
        font-weight: 900;
        margin-top: 6px;
        color: {text_color};
        line-height: 1;
    }}

    .metric-sub {{
        font-size: 13px;
        color: {muted_color};
        margin-top: 6px;
    }}

    .chat-panel {{
        background: {card_background};
        border-radius: 18px;
        box-shadow: 0 14px 28px rgba(16,24,40,0.08);
        border: 1px solid rgba(15,23,42,0.07);
        padding: 0;
        margin-bottom: 18px;
    }}

    .chat-list {{
        max-height: 420px;
        overflow-y: auto;
        padding: 16px 18px 10px 18px;
    }}

    .message-row {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 14px;
    }}

    .message-avatar {{
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #ba1d1d;
        border: 4px solid #781515;
        color: white;
        font-size: 18px;
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }}

    .message-user {{
        justify-content: flex-end;
    }}

    .message-user .message-avatar {{
        display: none;
    }}

    .message-bot .message-avatar {{
        display: flex;
    }}

    .message-time {{
        font-size: 12px;
        color: #98a2b3;
        margin-top: 28px;
        min-width: 56px;
        text-align: right;
    }}

    .message-user .message-time {{
        order: 3;
    }}

    .message-card {{
        border-radius: 18px;
        padding: 14px 16px;
        border: 1px solid rgba(15,23,42,0.06);
        box-shadow: 0 6px 14px rgba(15,23,42,0.04);
        max-width: 76%;
    }}

    .message-body {{
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 15px;
    }}

    .msg-user {{
        background: {message_user};
        color: {text_color};
        border-radius: 18px 18px 6px 18px;
    }}

    .msg-bot {{
        background: {message_bot};
        color: {text_color};
        border-radius: 18px 18px 18px 6px;
    }}

    .chat-empty {{
        text-align: center;
        padding: 40px 18px 42px 18px;
        color: {muted_color};
    }}

    .chat-empty-icon {{
        font-size: 54px;
        opacity: 0.9;
        margin-bottom: 10px;
    }}

    .stTextInput input {{
        border-radius: 12px !important;
        padding: 14px 16px !important;
        background: {input_background} !important;
        color: {input_text} !important;
        border: 1px solid rgba(15,23,42,0.12) !important;
        box-shadow: none !important;
    }}

    .composer-shell {{
        background: {card_background};
        border-radius: 18px;
        box-shadow: 0 14px 28px rgba(16,24,40,0.08);
        border: 1px solid rgba(15,23,42,0.07);
        padding: 0 18px 18px 18px;
    }}

    .composer-row {{
        display: flex;
        gap: 12px;
        align-items: center;
        padding-top: 16px;
    }}

    .composer-icon {{
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #1f2937;
        background: transparent;
        border: none;
        flex-shrink: 0;
    }}

    .composer-input {{
        flex: 1;
    }}

    .composer-send .stButton > button {{
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        width: 100%;
        height: 42px;
        min-width: 74px;
        box-shadow: 0 10px 22px rgba(220, 38, 38, 0.24) !important;
    }}

    .composer-send {{
        display: flex;
        align-items: end;
        height: 100%;
        padding-top: 0;
    }}

    .composer-send .stButton {{
        width: 100%;
    }}

    div[data-testid="stFileUploader"] {{
        margin: 16px 0 0 !important;
    }}

    div[data-testid="stFileUploader"] section,
    div[data-testid="stFileUploaderDropzone"] {{
        min-height: 42px !important;
        height: 42px !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }}

    div[data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}

    div[data-testid="stFileUploader"] button {{
        width: 96px !important;
        min-width: 96px !important;
        height: 42px !important;
        padding: 0 12px !important;
        border: 1px solid rgba(15,23,42,0.16) !important;
        border-radius: 12px !important;
        background: {input_background} !important;
        color: {text_color} !important;
        font-size: 0 !important;
        box-shadow: none !important;
    }}

    div[data-testid="stFileUploader"] button > * {{
        display: none !important;
    }}

    div[data-testid="stFileUploader"] button::after {{
        content: "Attach";
        font-size: 14px;
        font-weight: 700;
    }}

    div[data-testid="stFileUploader"] small {{
        display: none !important;
    }}

    div[data-testid="column"]:has(.composer-send) {{
        align-self: end;
    }}

    .hero-graphic {{
        width: 150px;
        height: 150px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 72px;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.92), rgba(255,255,255,0.65));
        box-shadow: 0 18px 34px rgba(255, 120, 120, 0.18);
        position: relative;
        overflow: hidden;
    }}

    .hero-graphic::after {{
        content: "";
        position: absolute;
        right: -10px;
        top: 24px;
        width: 84px;
        height: 84px;
        background: linear-gradient(135deg, rgba(255,120,120,0.25), rgba(255,120,120,0));
        transform: rotate(18deg);
        filter: blur(2px);
    }}

    .chat-toolbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 18px 12px 18px;
        border-bottom: 1px solid rgba(15,23,42,0.08);
    }}

    .chat-toolbar-title {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 20px;
        font-weight: 800;
        color: {text_color};
    }}

    .chat-toolbar-subtitle {{
        font-size: 14px;
        color: {muted_color};
        margin-top: 4px;
        font-weight: 500;
    }}

    .suggested-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid rgba(239,68,68,0.24);
        color: #e11d48;
        background: #fff;
        font-weight: 600;
        white-space: nowrap;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("<div class='main-shell'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='header'>
            <div>
                <div class='title'>Sports AI Bot <span class='accent'>Dashboard</span></div>
                <div class='subtitle'>Fast answers for players, fixtures, stats, and live sports context.</div>
            </div>
            <div style='display:flex; align-items:center; gap:18px;'>
                <div class='status'><span style='width:10px;height:10px;border-radius:50%;background:#22c55e;display:inline-block;'></span> Live assistant online</div>
                <div class='hero-graphic'>⚽</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-label'>Chats</div>
                <div class='metric-value'>{len(st.session_state.chat_history)}</div>
                <div class='metric-sub'>Saved conversations</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Messages</div>
                <div class='metric-value'>{len(st.session_state.messages)}</div>
                <div class='metric-sub'>Current thread</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Mode</div>
                <div class='metric-value'>{'Dark' if dark_mode else 'Light'}</div>
                <div class='metric-sub'>Theme setting</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Status</div>
                <div class='metric-value'>Online</div>
                <div class='metric-sub'>Ready to answer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='chat-toolbar'>
            <div>
                <div class='chat-toolbar-title'><span style='color:#ef4444;'>💬</span> Start a conversation</div>
                <div class='chat-toolbar-subtitle'>Ask about players, matches, stats, or tournaments.</div>
            </div>
            <div class='suggested-pill'>💡 Suggested Questions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.messages:
        st.markdown("<div class='chat-list'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            st.markdown(format_message(message), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class='chat-empty'>
                <div class='chat-empty-icon'>💬</div>
                <div style='font-size:20px; font-weight:800; color:inherit;'>No messages yet</div>
                <div style='margin-top:8px;'>Your conversations will appear here</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Render the user's message first, then generate the answer on this run.
    # This keeps the conversation visible while the model is working.
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        with st.spinner("Thinking..."):
            answer = run_model(question)
        st.session_state.messages.append({"role": "bot", "text": answer})
        st.session_state.pending_question = None
        safe_rerun()

    st.markdown("<div class='composer-shell'>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        upload_col, input_col, button_col = st.columns([2, 11, 2])
        with upload_col:
            uploaded_file = st.file_uploader(
                "Attach a document",
                type=["txt", "pdf"],
                key="chat_attachment",
                label_visibility="collapsed",
                help="Attach a TXT or PDF document for the assistant to use in this chat.",
            )
        with input_col:
            st.markdown("<div class='composer-row'>", unsafe_allow_html=True)
            user_input = st.text_input(
                "Message",
                placeholder="Type your message...",
                key="message_input",
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with button_col:
            st.markdown("<div class='composer-send'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("➤", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if submitted and (user_input or uploaded_file):
        attachment_name = None
        if uploaded_file:
            try:
                with st.spinner("Adding attachment to this chat..."):
                    attachment_name = add_uploaded_file_to_knowledge_base(uploaded_file)
            except Exception as exc:
                st.error(f"Could not process the attachment: {exc}")
                attachment_name = None

        if user_input:
            message_text = user_input
            if attachment_name:
                message_text = f"{user_input}\n\nAttached file: {attachment_name}"
            st.session_state.messages.append({"role": "user", "text": message_text})
            st.session_state.pending_question = user_input
        elif attachment_name:
            st.session_state.messages.append({"role": "user", "text": f"Attached file: {attachment_name}"})
            st.session_state.messages.append(
                {"role": "bot", "text": "I added the document to this chat's knowledge base. What would you like to know about it?"}
            )
        safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

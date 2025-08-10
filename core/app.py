import streamlit as st
from ragpipeline import RAGPipeline
import warnings
import uuid
import streamlit_authenticator as stauth
# import yaml
# from yaml.loader import SafeLoader

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Doctor Neha 🩺", layout="wide")

st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Open Sans', sans-serif;
            }
            body {
                background: linear-gradient(to right, #121212, #1e1e1e);
                color: #e0e0e0;
            }
            .stApp {
                background: #1c1c1c;
            }
            h1 {
                margin-bottom: 10px;
                color: #ffffff;
            }
            .centered-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                max-width: 850px;
                margin: auto;
            }
            .intro-box {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #2a2a2a, #3d3d3d);
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(255, 255, 255, 0.05);
                color: #f5f5f5;
                margin-top: 5px;
                margin-bottom: 10px;
            }
            .chat-box {
                width: 100%;
                max-height: 70vh;
                overflow-y: auto;
                background-color: #2b2b2b;
                border-radius: 20px;
                border: 1px solid #555;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                padding: 20px;
                margin-top: 10px;
            }
            .chat-container {
                margin-top: 20px;
                padding: 20px;
                background: #262626;
                border-radius: 16px;
                box-shadow: 0 6px 20px rgba(255, 255, 255, 0.05);
                border: 1px solid #444;
            }
            .message-bubble-user {
                background-color: #37474f;
                padding: 12px;
                border-radius: 20px;
                margin-bottom: 10px;
                text-align: right;
                margin-left: 25%;
                font-size: 15px;
                color: #ffffff;
            }
            .message-bubble-neha {
                background-color: #452744;
                padding: 12px;
                border-radius: 20px;
                margin-bottom: 10px;
                margin-right: 25%;
                font-size: 15px;
                color: #ffffff;
            }
            .input-box {
                margin-top: 20px;
                padding: 15px;
                background-color: #333;
                border-radius: 12px;
                width: 100%;
                box-shadow: 0 4px 10px rgba(255,255,255,0.05);
            }
            input {
                background-color: #222 !important;
                color: #fff !important;
            }
            input[type="text"] {
                color: white !important;
                background-color: #333333 !important;
                border: 1px solid #555 !important;
                border-radius: 8px;
                padding: 10px;
            }

            input::placeholder {
                color: #cccccc !important;
                opacity: 1 !important;
            }   
            .e1y5xkzn3 {  /* Send button styling - class may vary by version */
                background-color: #2196f3 !important;
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
            }
        </style>
    """, unsafe_allow_html=True)


st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #e1bee7, #e0f7fa); 
                border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-top: 10px;">
        <h1 style="margin: 0; font-size: 2.5em; color: #333;">
            👩‍⚕️ Doctor Neha - Your AI Physiotherapist
        </h1>
    </div>
    """, unsafe_allow_html=True)



st.markdown("<div class='centered-container'>", unsafe_allow_html=True)


st.markdown("""
        <div class="intro-box">
            <h3>🧠 Hello future Doctor!</h3>
            <p style="font-size: 16px;">
                I'm here to make your study stress-free and insightful.<br><br>
                💡 Created with care by <b>Vishal Lamkhade</b> as your <b>learning companion</b>.<br><br>
                🩺 Ask me anything about <b>physiotherapy</b>, <b>anatomy</b>, or <b>healthcare topics</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)


if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


if "assistant" not in st.session_state:
        st.session_state.assistant = RAGPipeline(
            doc_path="C:/Users/comp/Desktop/doctorneha/data/doctorneha.pdf"
        )

if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


with st.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

        for sender, msg in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f"<div class='message-bubble-user'><b>🧑 You:</b> {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='message-bubble-neha'><b>👩‍⚕️ Doctor Neha:</b> {msg}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


with st.form("chat_form", clear_on_submit=True):
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        user_input = st.text_input("You:", placeholder="Type your question here...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
        st.markdown("</div>", unsafe_allow_html=True)


if submitted and user_input:
        with st.spinner("👩‍⚕️ Doctor Neha is thinking..."):
            response = st.session_state.assistant.ask(
                user_input, 
                session_id=st.session_state.session_id  
            )

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("neha", response))
        st.rerun()

st.markdown("""
    <br><hr>
    <center>
        <p style="color: white;">
            💝 A Rakshabandhan Gift by Vishal Lamkhade | Made with LangChain, Groq, HuggingFace & Streamlit
        </p>
    </center>
    """, unsafe_allow_html=True)

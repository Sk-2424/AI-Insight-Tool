import os
import sys
import streamlit as st
import base64
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

# Set Page Configuration with CODM Icon
st.set_page_config(
    page_title="CODM: AI Tool",
    page_icon=os.path.join(os.getcwd(),"Data\images.jpg"),
    layout="centered"
)

# Load Background Image and Encode it in Base64
background_image_path = os.path.join(os.getcwd(),"Data\img8.jpg") # Ensure this path is correct
if os.path.exists(background_image_path):
    with open(background_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    background_style = f"""
    <style>

        .stApp {{
        margin: 0 !important;
        padding: 0 !important;
        background: url('data:image/jpg;base64,{encoded_image}') no-repeat center center fixed;
        background-size: cover;
        height: 100vh;
        width: 100vw;
        display: flex;
        align-items: center;
        justify-content: center;
        }}


        /* Center Content */
        .chat-container {{
            max-width: 600px;
            width: 90%;
            background: rgba(0, 0, 0,0.0);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: auto;
            margin-top: 10vh;

        }}

        /* Title */
        .title {{
            color: #FFD700;
            font-family: 'Impact', sans-serif;
            font-size: 36px;
            text-shadow: 3px 3px 5px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }}

        /* Chat Bubble */
        .chat-bubble {{
            padding: 12px 16px;
            border-radius: 12px;
            margin: 8px 0;
            font-size: 16px;
            max-width: 75%;
            word-wrap: break-word;
            display: inline-block;
            color: white;
            background: rgba(255, 255, 255, 0.2);
        }}

        /* User and Bot Messages */
        .user-query {{ 
            background-color: rgba(255, 215, 0, 0.8);
            color: black;
            text-align: left; 
            float: left;
            clear: both; 
            }}

        .bot-response {{
            background-color: rgba(0, 0, 0, 0.7);
            text-align: left;
            float: right;
            }}

        /* Input Styling */
        .stTextInput > div > div > input {{
            font-size: 18px; 
            padding: 10px;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            border: 2px solid #FFD700;
        }}

        /* Buttons */
        .stButton > button {{
            font-size: 16px;
            padding: 10px 15px;
            border-radius: 8px;
            background-color: #ff4500;
            color: white;
            font-weight: bold;
            text-shadow: 1px 1px 2px black;
        }}

        .stButton > button:hover {{ background-color: #b22222; }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
else:
    st.warning("⚠️ Background image not found. Please check the file path.")

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.logger import logging
from src.chains import create_rag_chain, ask_question, create_retriever, clear_memory

# Load environment variables
load_dotenv()
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Initialize embeddings and retriever
embeddings = OpenAIEmbeddings()
index_name = "aibot"

# Cache RAG Chain
@st.cache_resource
def get_rag_chain(index_name, _embeddings):
    retriever = create_retriever(index_name, _embeddings)
    rag_chain = create_rag_chain(retriever)
    return rag_chain

rag_chain = get_rag_chain(index_name, embeddings)

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

# ---- UI Layout ---- #
with st.container():  # This ensures everything stays in one page
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="title">💬 COD Mobile: AI Insight Tool</div>', unsafe_allow_html=True)
    # st.write("🔫 **Ask me anything about Call of Duty Mobile!** 🎯")

    query = st.text_input("Enter your question:", "")

    # Buttons: Submit & Clear Conversation
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        submit = st.button("🔥 Submit", use_container_width=True)
    with col2:
        clear_chat = st.button("💣 Clear Chat", use_container_width=True)

    # Processing Query
    if submit:
        if query.strip():
            with st.spinner("Thinking... 🎮"):
                try:
                    input_data = {"chat_history": st.session_state["memory"].load_memory_variables({})["chat_history"], "input": query}
                    response = ask_question(input_data, rag_chain)
                    st.session_state["memory"].save_context({"input": query}, {"answer": response})                
                    logging.info("Latest conversation saved in memory.")
                except Exception as e:
                    st.error("⚠️ An error occurred. Please try again.")
                    st.text(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a valid question.")

    # Clearing Conversation
    if clear_chat:
        clear_memory(st.session_state["memory"])
        st.session_state["memory"] = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
        st.success("✅ Chat history erased!")

    # Display Chat History
    if st.session_state["memory"].buffer:
        st.write("📝 **Conversation History:**")
        for msg in st.session_state["memory"].buffer:
            if msg.type == "human":
                st.markdown(f'<div class="chat-bubble user-query">👤 <strong>You:</strong> <br> {msg.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble bot-response">🤖 <strong>AI:</strong> <br> {msg.content}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Closing the chat-container

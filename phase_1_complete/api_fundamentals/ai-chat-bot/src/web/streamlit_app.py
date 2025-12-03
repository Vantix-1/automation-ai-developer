"""
Streamlit Web Interface - Modern chat UI for AI assistant
"""
import streamlit as st
import os
from datetime import datetime
from src.core.chat_engine import AIChatEngine
from src.utils.config_loader import load_config

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'chat_engine' not in st.session_state:
    config = load_config()
    st.session_state.chat_engine = AIChatEngine(
        api_key=config['openai_api_key'],
        model=config.get('model', 'gpt-3.5-turbo')
    )
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b3137;
        border-left: 4px solid #00d4ff;
    }
    .chat-message.assistant {
        background-color: #1a1a2e;
        border-left: 4px solid #764ba2;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🤖 AI Chat Settings")
    st.markdown("---")
    
    # Model selection
    model_option = st.selectbox(
        "AI Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    
    # Conversation controls
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.chat_engine.memory.clear_conversation("user")
        st.rerun()
    
    if st.button("💾 Save Conversation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/conversations/chat_{timestamp}.json"
        st.session_state.chat_engine.memory.save_conversation("user", filename)
        st.success(f"Conversation saved to {filename}")
    
    st.markdown("---")
    st.markdown("### Conversation Info")
    st.info(f"Messages in history: {len(st.session_state.messages)}")

# Main chat interface
st.title("🤖 AI Chat Assistant")
st.markdown("Chat with an intelligent AI assistant powered by OpenAI")

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-header">👤 You</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="message-header">🤖 Assistant</div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.spinner("🤖 Thinking..."):
        response = st.session_state.chat_engine.chat(prompt, "user")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the display
    st.rerun()
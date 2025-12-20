# streamlit_app.py
"""
Streamlit Web Interface for OpenAI Chat
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import openai
import os
from dotenv import load_dotenv
import json
import datetime
from pathlib import Path

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="OpenAI Chat Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Pricing information
PRICING = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo-16k": {"input": 0.0030, "output": 0.0040},
    "gpt-4": {"input": 0.0300, "output": 0.0600},
    "gpt-4-turbo-preview": {"input": 0.0100, "output": 0.0300},
}

def calculate_cost(prompt_tokens, completion_tokens, model):
    """Calculate cost based on token usage"""
    if model not in PRICING:
        model = "gpt-3.5-turbo"
    
    pricing = PRICING[model]
    prompt_cost = (prompt_tokens / 1000) * pricing["input"]
    completion_cost = (completion_tokens / 1000) * pricing["output"]
    
    return prompt_cost + completion_cost

def save_conversation():
    """Save conversation to JSON file"""
    if not st.session_state.conversation_history:
        return None
    
    export_data = {
        "metadata": {
            "export_date": datetime.datetime.now().isoformat(),
            "total_exchanges": len(st.session_state.conversation_history),
            "total_tokens": st.session_state.total_tokens,
            "total_cost": st.session_state.total_cost,
        },
        "conversation": st.session_state.conversation_history
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"web_conversation_{timestamp}.json"
    
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    filepath = exports_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return filepath

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv('OPENAI_API_KEY', ''),
        help="Enter your OpenAI API key"
    )
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        options=list(PRICING.keys()),
        index=0,
        help="Choose which GPT model to use"
    )
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        height=100,
        help="Instructions for the AI's behavior"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random"
    )
    
    # Max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum tokens in response"
    )
    
    # Stats display
    st.divider()
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    with col2:
        st.metric("Total Cost", f"${st.session_state.total_cost:.6f}")
    
    st.metric("Messages", len(st.session_state.messages))
    
    # Export button
    st.divider()
    if st.button("💾 Export Conversation", use_container_width=True):
        filepath = save_conversation()
        if filepath:
            st.success(f"Conversation saved to: {filepath}")
            # Provide download link
            with open(filepath, "r") as f:
                st.download_button(
                    label="📥 Download JSON",
                    data=f,
                    file_name=filepath.name,
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.warning("No conversation to export")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# Main chat interface
st.title("🤖 OpenAI Chat Interface")
st.caption("Powered by OpenAI GPT API with cost tracking")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show token info for assistant messages
            if message["role"] == "assistant" and "token_info" in message:
                token_info = message["token_info"]
                st.caption(f"Tokens: {token_info['total']:,} | Cost: ${token_info['cost']:.6f}")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare messages for API
    messages_for_api = [{"role": "system", "content": system_prompt}]
    messages_for_api.extend(st.session_state.messages)
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            stream = client.chat.completions.create(
                model=model,
                messages=messages_for_api,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Estimate tokens and cost (streaming doesn't provide usage)
            estimated_prompt_tokens = len(prompt) // 4
            estimated_completion_tokens = len(full_response) // 4
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
            
            cost = calculate_cost(estimated_prompt_tokens, estimated_completion_tokens, model)
            
            # Display token info
            st.caption(f"Estimated tokens: {estimated_total_tokens:,} | Cost: ${cost:.6f}")
        
        # Add assistant message to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "token_info": {
                "prompt": estimated_prompt_tokens,
                "completion": estimated_completion_tokens,
                "total": estimated_total_tokens,
                "cost": cost
            }
        })
        
        # Update statistics
        st.session_state.total_tokens += estimated_total_tokens
        st.session_state.total_cost += cost
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model,
            "user_message": prompt,
            "assistant_message": full_response,
            "estimated_tokens": estimated_total_tokens,
            "estimated_cost": cost
        })
        
        # Force rerun to update sidebar stats
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {e}")

# Cost information section
with st.expander("💰 Cost Information"):
    st.markdown("""
    ### Pricing per 1,000 tokens:
    
    | Model | Input | Output |
    |-------|-------|--------|
    | gpt-3.5-turbo | $0.0015 | $0.0020 |
    | gpt-3.5-turbo-16k | $0.0030 | $0.0040 |
    | gpt-4 | $0.0300 | $0.0600 |
    | gpt-4-turbo-preview | $0.0100 | $0.0300 |
    
    *Note: Token estimates are approximate. Actual usage may vary.*
    """)
    
    # Cost calculator
    st.subheader("Cost Calculator")
    col1, col2, col3 = st.columns(3)
    with col1:
        calc_tokens = st.number_input("Tokens", min_value=1, value=1000, step=100)
    with col2:
        calc_model = st.selectbox("Model", options=list(PRICING.keys()), index=0)
    with col3:
        calc_ratio = st.slider("Output/Input ratio", 0.1, 2.0, 1.0, 0.1)
    
    if calc_model in PRICING:
        pricing = PRICING[calc_model]
        input_tokens = calc_tokens / (1 + calc_ratio)
        output_tokens = calc_tokens - input_tokens
        cost = calculate_cost(input_tokens, output_tokens, calc_model)
        
        st.info(f"""
        **Estimated Cost: ${cost:.4f}**
        - Input tokens: {input_tokens:,.0f}
        - Output tokens: {output_tokens:,.0f}
        - Total tokens: {calc_tokens:,.0f}
        """)

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    1. **Enter your OpenAI API key** in the sidebar (get one from [OpenAI Platform](https://platform.openai.com/api-keys))
    2. **Configure** the model, system prompt, and parameters in the sidebar
    3. **Start chatting** by typing in the chat input at the bottom
    4. **Monitor costs** in real-time in the sidebar
    5. **Export conversations** using the export button
    6. **Clear conversation** to start fresh
    
    ### Tips:
    - Use GPT-3.5-turbo for testing (cheapest)
    - GPT-4 is more capable but 20x more expensive
    - Adjust temperature for creativity (higher = more random)
    - Set max tokens to control response length
    """)

# Footer
st.divider()
st.caption("OpenAI API Mastery - Phase 2 | Enhanced Day 1-2 | AI Developer Roadmap 2025")
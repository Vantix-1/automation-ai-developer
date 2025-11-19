🚀 Deployment Guide
Local Development
```
# 1. Clone and setup
git clone https://github.com/Vantix-1/ai-chat-bot.git
cd ai-chat-bot

# 2. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key

# 4. Run application
streamlit run src/web/streamlit_app.py
```

Production Deployment
```
# Using Docker
docker build -t ai-chat-bot .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key ai-chat-bot

# Or with Docker Compose
docker-compose up -d
```
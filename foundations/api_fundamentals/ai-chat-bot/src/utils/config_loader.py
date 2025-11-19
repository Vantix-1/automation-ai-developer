"""
Configuration Loader - Manages environment variables and settings
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('MODEL', 'gpt-3.5-turbo'),
        'max_history': int(os.getenv('MAX_HISTORY', '20')),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '500'))
    }
    
    # Validate required settings
    if not config['openai_api_key']:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return config
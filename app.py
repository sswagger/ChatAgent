import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuration file path
CONFIG_FILE = 'config.json'

# Lock for thread-safe config writes
config_lock = threading.Lock()

# Configuration from environment (LLM settings stay in .env, user settings in config.json)
LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:11434/v1/chat/completions')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')

# User-editable config defaults
USER_CONFIG_DEFAULTS = {
    'systemPrompt': os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant.'),
    'bufferSize': int(os.getenv('BUFFER_SIZE', '10')),
    'temperature': float(os.getenv('TEMPERATURE', '0.7')),
    'port': int(os.getenv('PORT', '8080'))
}

# Global config dictionary (user-editable settings)
config = {}


def load_config():
    """Load user config from file, fall back to defaults."""
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in USER_CONFIG_DEFAULTS.items():
                    if key not in config:
                        config[key] = value
        except (json.JSONDecodeError, IOError):
            config = USER_CONFIG_DEFAULTS.copy()
    else:
        config = USER_CONFIG_DEFAULTS.copy()
        save_config()


def save_config():
    """Save config to file."""
    global config
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)


# Load config on startup
load_config()

# In-memory chat storage (replace with database for production)
chat_history = {}


def estimate_tokens(text):
    """Estimate token count using tiktoken."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback: ~4 chars per token
        return len(text) // 4


def get_total_tokens(messages):
    """Calculate total token count for messages."""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            if 'role' in msg:
                total += estimate_tokens(msg['role'])
            if 'content' in msg:
                total += estimate_tokens(msg['content'])
    return total


def call_llm(messages):
    """Call the OpenAI-compatible API."""
    headers = {
        'Content-Type': 'application/json',
    }
    if LLM_API_KEY:
        headers['Authorization'] = f'Bearer {LLM_API_KEY}'
    
    payload = {
        'model': LLM_MODEL,
        'messages': messages,
        'temperature': config.get('temperature', 0.7)
    }
    
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}).get('content', ''), None
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """Serve the main chat interface."""
    return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Get or update user configuration."""
    global config
    if request.method == 'GET':
        return jsonify({
            'systemPrompt': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt']),
            'bufferSize': config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']),
            'temperature': config.get('temperature', USER_CONFIG_DEFAULTS['temperature'])
        })
    else:
        data = request.json
        if 'systemPrompt' in data:
            config['systemPrompt'] = data['systemPrompt']
        if 'bufferSize' in data:
            config['bufferSize'] = int(data['bufferSize'])
        if 'temperature' in data:
            config['temperature'] = float(data['temperature'])
        save_config()
        return jsonify({'success': True})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat message."""
    data = request.json
    message = data.get('message', '')
    chat_id = data.get('chatId', 'default')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    # Initialize chat history if needed
    if chat_id not in chat_history:
        chat_history[chat_id] = []

    # Add system prompt if this is a new chat
    if len(chat_history[chat_id]) == 0:
        chat_history[chat_id].append({'role': 'system', 'content': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt'])})

    # Add user message
    chat_history[chat_id].append({'role': 'user', 'content': message})

    # Apply buffer size limit (keep system prompt + buffer)
    if len(chat_history[chat_id]) > config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']) + 1:
        chat_history[chat_id] = [chat_history[chat_id][0]] + chat_history[chat_id][-config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']):]

    # Get response from LLM
    response_content, error = call_llm(chat_history[chat_id])

    if error:
        return jsonify({'error': f'Failed to get response: {error}'}), 500

    # Add assistant response
    chat_history[chat_id].append({'role': 'assistant', 'content': response_content})
    
    # Calculate token usage
    token_count = get_total_tokens(chat_history[chat_id])
    
    return jsonify({
        'response': response_content,
        'tokenCount': token_count,
        'chatId': chat_id
    })


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """Start a new chat session."""
    data = request.json or {}
    chat_id = data.get('chatId', 'default')
    chat_history[chat_id] = []
    
    # Add system prompt to new chat
    chat_history[chat_id].append({'role': 'system', 'content': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt'])})
    
    return jsonify({
        'success': True,
        'chatId': chat_id,
        'tokenCount': get_total_tokens(chat_history[chat_id])
    })


@app.route('/api/token-count', methods=['POST'])
def token_count():
    """Get current token count for a chat."""
    data = request.json
    chat_id = data.get('chatId', 'default')
    
    if chat_id not in chat_history:
        return jsonify({'tokenCount': 0})
    
    return jsonify({
        'tokenCount': get_total_tokens(chat_history[chat_id])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']))

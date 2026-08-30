import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# load the environment variables for LLM configuration
load_dotenv()

# set up directory for API
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuration file path
CONFIG_FILE = 'config.json'
LOG_FILE = 'log.txt'

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

# Global config dictionaries
config = {}
chat_history = {}


def estimate_tokens(text):
	"""Estimate token count using tiktoken."""
	try:
		if not text or not isinstance(text, str):
			return 0
		import tiktoken
		encoding = tiktoken.get_encoding("cl100k_base")
		return len(encoding.encode(text))
	except Exception:
		return len(str(text)) // 4 if text else 0


def get_total_tokens(messages):
	"""Calculate the total token count for messages."""
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
		response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=100)
		response.raise_for_status()
		result = response.json()

		log_actions("call to LLM", payload, result, error=str(response.status_code))
		return result.get('choices', [{}])[0].get('message', {}), None
	except Exception as e:
		return None, str(e)


def load_config():
	"""Load user config from file, fall back to defaults."""
	global config
	if os.path.exists(CONFIG_FILE):
		try:
			with open(CONFIG_FILE, 'r') as f:
				config = json.load(f)
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


def log_actions(path, request_payload, response_payload, is_new_chat=False, request_method="unknown", error="200"):
	if is_new_chat:
		with open(LOG_FILE, 'w') as log:
			log.write(
				f"{path}\n"+
				f"request: json={request_payload} | method={request_method}\n"+
				f"response: json={response_payload} | error={error}\n\n"
			)
	else:
		with open(LOG_FILE, 'a') as log:
			log.write(
				f"{path}\n"+
				f"request: json={request_payload} | method={request_method}\n"+
				f"response: json={response_payload} | error={error}\n\n"
			)


# Load configs on startup - load mcp_config first for add_mcp_capabilities_instruction
load_config()


@app.route('/')
def index():
	"""Serve the main chat interface."""
	log_actions("/", "", "static/index.html", is_new_chat=True)
	return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
	"""Get or update user configuration."""
	global config
	if request.method == 'GET':
		json_config = {
			'systemPrompt': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt']),
			'bufferSize': config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']),
			'temperature': config.get('temperature', USER_CONFIG_DEFAULTS['temperature'])
		}
		log_actions("/api/config", "", str(json_config), request_method=request.method)
		return jsonify(json_config)
	else:
		data = request.json
		if 'systemPrompt' in data:
			config['systemPrompt'] = data['systemPrompt']
		if 'bufferSize' in data:
			config['bufferSize'] = int(data['bufferSize'])
		if 'temperature' in data:
			config['temperature'] = float(data['temperature'])
		save_config()

		response = {'success': True}
		log_actions("/api/config", data, str(response), request_method=request.method)
		return jsonify(response)

@app.route('/api/chat', methods=['POST'])
def chat():
	"""Handle chat message with optional MCP tool integration."""
	data = request.json
	message = data.get('message', '')
	chat_id = data.get('chatId', 'default')

	if not message:
		json_response = {'error': 'Message is required'}
		log_actions("/api/chat", data, str(json_response), error="400")
		return jsonify(json_response), 400

	# Initialize chat history if needed
	if chat_id not in chat_history:
		chat_history[chat_id] = []

	# Add system prompt if this is a new chat
	if len(chat_history[chat_id]) == 0:
		chat_history[chat_id].append({'role': 'system', 'content': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt'])})

	# Add user message
	chat_history[chat_id].append({'role': 'user', 'content': message})

	# Apply buffer size limit
	if len(chat_history[chat_id]) > config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']) + 1:
		chat_history[chat_id] = [chat_history[chat_id][0]] + chat_history[chat_id][-config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']):]

	# Include get_capabilities function when MCP is enabled
	full_response, error = call_llm(chat_history[chat_id])

	if error:
		json_response = {'error': f'Failed to get response= {error}'}
		log_actions("/api/chat", data, str(json_response), error="500")
		return jsonify(json_response), 500

	# Add assistant response
	chat_history[chat_id].append({'role': 'assistant', 'content': full_response.get('content', '')})

	# Calculate token usage
	token_count = get_total_tokens(chat_history[chat_id])

	json_response = {
		'response': full_response.get('content', ''),
		'tokenCount': token_count,
		'chatId': chat_id
	}
	log_actions("/api/chat", data, str(json_response),)
	return jsonify(json_response)


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
	"""Start a new chat session."""
	data = request.json or {}
	chat_id = data.get('chatId', 'default')
	chat_history[chat_id] = []
	chat_history[chat_id].append({'role': 'system', 'content': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt'])})

	json_response = {
		'success': True,
		'chatId': chat_id,
		'tokenCount': get_total_tokens(chat_history[chat_id])
	}
	log_actions("/api/new-chat", data, str(json_response), is_new_chat=True)
	return jsonify(json_response)


@app.route('/api/token-count', methods=['POST'])
def token_count():
	"""Get the current token count for a chat."""
	data = request.json
	chat_id = data.get('chatId', 'default')

	if chat_id not in chat_history:
		json_response = jsonify({'tokenCount': 0})
		log_actions("/api/token-count", data, str(json_response),)
		return json_response

	json_response = {'tokenCount': get_total_tokens(chat_history[chat_id])}
	log_actions("/api/token-count", data, str(json_response),)
	return jsonify(json_response)


@app.route('/api/log', methods=['POST'])
def log_data():
	data = request.json

	with open(LOG_FILE, 'a') as log:
		log.write(data.get("log") + "\n\n")

	json_response = {"success": True}
	log_actions('/api/log', data, str(json_response))
	return jsonify(json_response)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']), debug=True)

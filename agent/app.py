import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests
import uuid

# load the environment variables for LLM configuration
load_dotenv()

# set up directory for API
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuration file path
CONFIG_FILE = 'config.json'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'logs', 'log-')

# Lock for thread-safe config writes
config_lock = threading.Lock()

# Configuration from environment (LLM settings stay in .env, user settings in config.json)
LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:11434/v1/chat/completions')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')

# User-editable config defaults
USER_CONFIG_DEFAULTS = {
	'systemPrompt': "",
	'bufferSize': 1,
	'temperature': 0,
	'port': ""
}

# Global chat dictionary
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


def call_llm(messages, session):
	"""Call the OpenAI-compatible API."""
	config = load_config(session)
	headers = {
		'Content-Type': 'application/json',
	}
	if LLM_API_KEY:
		headers['Authorization'] = f'Bearer {LLM_API_KEY}'

	payload = {
		'model': LLM_MODEL,
		'messages': messages,
		'temperature': config.get('temperature', USER_CONFIG_DEFAULTS['temperature'])
	}

	try:
		response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=100)
		response.raise_for_status()
		result = response.json()

		log_actions(session, "call to LLM", payload, result, error=str(response.status_code))
		return result.get('choices', [{}])[0].get('message', {}), None
	except Exception as e:
		return None, str(e)


def load_config(session) -> dict:
	"""Load user config from file, fall back to defaults."""
	if os.path.exists(CONFIG_FILE):
		try:
			with open(CONFIG_FILE, 'r') as f:
				# load all json
				configs = json.load(f)

				if configs.get(session):
					return {'systemPrompt': configs.get(session).get("systemPrompt"), 'bufferSize': configs.get(session).get("bufferSize"), 'temperature': configs.get(session).get("temperature"), 'port': configs.get(session).get("port")}
				else:
					configs[session] = {'systemPrompt': configs.get("default").get("systemPrompt"), 'bufferSize': configs.get("default").get("bufferSize"), 'temperature': configs.get("default").get("temperature"), 'port': configs.get("default").get("port")}
					with open(CONFIG_FILE, 'w') as file:
						json.dump(configs, file, indent=4)

					return configs.get(session)
		except (json.JSONDecodeError, IOError):
			return {}
	else:
		return {}


def save_config(session, config):
	"""Save config to file."""
	if os.path.exists(CONFIG_FILE):
		try:
			with open(CONFIG_FILE, 'r') as f:
				# load all json
				configs = json.load(f)

				configs[session] = config
				with open(CONFIG_FILE, 'w') as file:
					json.dump(configs, file, indent=4)

		except (json.JSONDecodeError, IOError):
			pass
	else:
		pass


def log_actions(session, path, request_payload, response_payload, is_new_chat=False, request_method="unknown", error="200"):
	if is_new_chat:
		with open(LOG_FILE + str(session) + ".txt", 'w') as log:
			log.write(
				f"{path}\n"+
				f"request: json={request_payload} | method={request_method}\n"+
				f"response: json={response_payload} | error={error}\n\n"
			)
	else:
		with open(LOG_FILE + str(session) + ".txt", 'a') as log:
			log.write(
				f"{path}\n"+
				f"request: json={request_payload} | method={request_method}\n"+
				f"response: json={response_payload} | error={error}\n\n"
			)


@app.route('/')
def index():
	"""Serve the main chat interface."""
	return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
	"""Get or update user configuration."""

	if request.method == 'GET':
		# get session
		session = request.headers.get('session')
		if not session or chat_history.get(session, None) is None:
			json_response = {'success': False, 'error': 'session not found'}
			log_actions("error", "/api/config", json_response, str(json_response), request_method=request.method)
			return jsonify(json_response)

		config = load_config(session)

		json_config = {
			'systemPrompt': config.get('systemPrompt'),
			'bufferSize': config.get('bufferSize'),
			'temperature': config.get('temperature'),
		}
		log_actions(session, "/api/config", "", str(json_config), request_method=request.method)
		return jsonify(json_config)
	else:
		data = request.json
		session = data.get('session')
		if not session or chat_history.get(session, None) is None:
			json_response = {'success': False, 'error': 'session not found'}
			log_actions("error", "/api/config", data, str(json_response), request_method=request.method, error="400")
			return jsonify(json_response), 400

		config = load_config(session)

		if 'systemPrompt' in data:
			config['systemPrompt'] = data['systemPrompt']
		if 'bufferSize' in data:
			config['bufferSize'] = int(data['bufferSize'])
		if 'temperature' in data:
			config['temperature'] = float(data['temperature'])
		save_config(session, config)

		response = {'success': True}
		log_actions(session, "/api/config", data, str(response), request_method=request.method)
		return jsonify(response)


@app.route('/api/chat', methods=['POST'])
def chat():
	"""Handle chat message"""
	data = request.json
	message = data.get('message', '')
	session = data.get('session')

	# ensure session is valid
	if not session or chat_history.get(session, None) is None:
		json_response = {'error': 'session not found'}
		log_actions("error", "/api/chat", data, str(json_response), request_method="POST", error="400")
		return jsonify(json_response), 400
	config = load_config(session)

	# ensure that there is a message
	if not message:
		json_response = {'error': 'Message is required'}
		log_actions("error", "/api/chat", data, str(json_response), request_method="POST", error="400")
		return jsonify(json_response), 400

	# Add system prompt if this is a new chat
	if len(chat_history[session]) == 0:
		chat_history[session].append({'role': 'system', 'content': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt'])})

	# Add user message
	chat_history[session].append({'role': 'user', 'content': message})

	# Apply buffer size limit
	if len(chat_history[session]) > config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']) + 1:
		chat_history[session] = [chat_history[session][0]] + chat_history[session][-config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']):]

	# Include get_capabilities function when MCP is enabled
	full_response, error = call_llm(chat_history[session], session)

	if error:
		json_response = {'error': f'Failed to get response= {error}'}
		log_actions("error", "/api/chat", data, str(json_response), request_method="POST", error="500")
		return jsonify(json_response), 500

	# Add assistant response
	chat_history[session].append({'role': 'assistant', 'content': full_response.get('content', '')})

	# Calculate token usage
	token_count = get_total_tokens(chat_history[session])

	json_response = {
		'response': full_response.get('content', ''),
		'tokenCount': token_count
	}
	log_actions(session, "/api/chat", data, str(json_response), request_method="POST")
	return jsonify(json_response)


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
	"""Start a new chat session."""
	data = request.json
	session = data.get('session')

	if not session or chat_history.get(session, None) is None:
		new_session = str(uuid.uuid4())
		chat_history[new_session] = []

		json_response = {
			'success': True,
			'session': new_session,
			'tokenCount': get_total_tokens(chat_history[new_session])
		}
		log_actions(new_session, "/api/new-chat", data, str(json_response), request_method="POST", is_new_chat=True)
		return jsonify(json_response)
	else:
		config = load_config(session)
		chat_history[session] = []
		chat_history[session].append({'role': 'system', 'content': config['systemPrompt']})

		json_response = {
			'success': True,
			'session': session,
			'tokenCount': get_total_tokens(chat_history[session])
		}
		log_actions(session, "/api/new-chat", data, str(json_response), request_method="POST", is_new_chat=True)
		return jsonify(json_response)


@app.route('/api/token-count', methods=['POST'])
def token_count():
	"""Get the current token count for a chat."""
	data = request.json
	session = data.get('session')

	if not session or chat_history.get(session, None) is None:
		json_response = jsonify({'tokenCount': 0})
		log_actions("error", "/api/token-count", data, str(json_response), request_method="POST")
		return json_response

	json_response = {'tokenCount': get_total_tokens(chat_history[session])}
	log_actions(session, "/api/token-count", data, str(json_response), request_method="POST")
	return jsonify(json_response)


@app.route('/api/log', methods=['POST'])
def log_data():
	data = request.json
	session = data.get('session')

	if not session or chat_history.get(session, None) is None:
		json_response = {'success': False, 'error': 'session not found'}
		log_actions("error", "/api/log", data, str(json_response), request_method="POST")
		return jsonify(json_response)

	with open(LOG_FILE + session + ".txt", 'a') as log:
		log.write(data.get("log") + "\n\n")

	json_response = {"success": True}
	log_actions(session, '/api/log', data, str(json_response), request_method="POST")
	return jsonify(json_response)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=load_config("default").get("port"), debug=True)

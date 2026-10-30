"""
AI Chat Agent - File Implementation
Combines Flask API, Configuration Loader, and RAG Service
"""

import os
import json
import threading
from typing import List, Optional, Tuple

# Flask and web imports
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests
import uuid

# RAG imports
from model.RagService import RAGService

# Load environment variables for LLM configuration
load_dotenv()

# set up directory for API
app = Flask(__name__, static_folder='static', static_url_path='/static')

# === Configuration Constants ===
CONFIG_FILE = 'config.json'
RAG_CONFIG_FILE = 'rag_config.json'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, 'logs', 'log-')

# Lock for thread-safe config writes
config_lock = threading.Lock()

# Configuration from environment (LLM settings stay in .env, user settings in config.json)
LLM_API_URL = os.getenv('LLM_API_URL', '')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')

# User-editable config defaults
USER_CONFIG_DEFAULTS = {
	'systemPrompt': "",
	'bufferSize': 1,
	'temperature': 0,
	'port': ""
}

# === Global State ===
usr_msgs = {"default": ""}
msg_count = {"default": 0}
rag_services = {}
chat_history = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_tokens(text: str) -> int:
	"""Estimate token count using tiktoken."""
	try:
		if not text or not isinstance(text, str):
			return 0
		import tiktoken
		encoding = tiktoken.get_encoding("cl100k_base")
		return len(encoding.encode(text))
	except Exception:
		return len(str(text)) // 4 if text else 0


def get_total_tokens(messages: List[dict]) -> int:
	"""Calculate the total token count for a list of messages."""
	total = 0
	for msg in messages:
		if isinstance(msg, dict):
			if 'role' in msg:
				total += estimate_tokens(msg['role'])
			if 'content' in msg:
				total += estimate_tokens(msg['content'])
	return total


def call_llm(messages: List[dict], session) -> Tuple[dict|None, Optional[str]]:
	"""Call the OpenAI-compatible LLM API."""
	headers = {
		'Content-Type': 'application/json',
	}
	if LLM_API_KEY:
		headers['Authorization'] = f'Bearer {LLM_API_KEY}'

	config = load_config(session, CONFIG_FILE)

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


def load_config(session, config_file: str):
	"""Load user config from file, fall back to defaults."""
	if os.path.exists(config_file):
		try:
			with (open(config_file, 'r') as f):
				# load all json
				configs = json.load(f)

				if configs.get(session):
					for key, value in configs.get("default", {}).items():
						if key not in configs.get(session):
							configs[session][key] = value
				else:
					configs[session] = configs.get("default", {})
					with open(CONFIG_FILE, 'w') as file:
						json.dump(configs, file, indent=4)

				return configs.get(session)
		except (json.JSONDecodeError, IOError):
			return {}
	else:
		return {}


def save_config(session, configuration, config_file: str):
	"""Save configuration to a JSON file with thread-safe locking."""
	if os.path.exists(config_file):
		try:
			with open(config_file, 'r') as f:
				# load all json
				configs = json.load(f)

				configs[session] = configuration
				with open(config_file, 'w') as file:
					json.dump(configs, file, indent=4)

		except (json.JSONDecodeError, IOError):
			pass
	else:
		pass


def log_actions(session, path: str, request_payload, response_payload, is_new_chat: bool = False, request_method: str = "unknown", error: str = "200"):
	"""Log API actions to the log file."""
	if is_new_chat:
		with open(LOG_FILE + str(session) + ".txt", 'w') as log:
			log.write(
				f"{path}\n" +
				f"request: json={request_payload} | method={request_method}\n" +
				f"response: json={response_payload} | error={error}\n\n"
			)
	else:
		with open(LOG_FILE + str(session) + ".txt", 'a') as log:
			log.write(
				f"{path}\n" +
				f"request: json={request_payload} | method={request_method}\n" +
				f"response: json={response_payload} | error={error}\n\n"
			)


# =============================================================================
# FLASK API ENDPOINTS
# =============================================================================
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

		config = load_config(session, CONFIG_FILE)

		json_config = {
			'systemPrompt': config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt']),
			'bufferSize': config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']),
			'temperature': config.get('temperature', USER_CONFIG_DEFAULTS['temperature'])
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

		config = load_config(session, CONFIG_FILE)

		if 'systemPrompt' in data:
			config['systemPrompt'] = data['systemPrompt']
		if 'bufferSize' in data:
			config['bufferSize'] = int(data['bufferSize'])
		if 'temperature' in data:
			config['temperature'] = float(data['temperature'])
		save_config(session, config, CONFIG_FILE)

		response = {'success': True}
		log_actions(session, "/api/config", data, str(response), request_method=request.method)
		return jsonify(response)


@app.route('/api/chat', methods=['POST'])
def chat():
	"""Handle chat messages with optional RAG integration."""
	data = request.json
	message = data.get('message', '')
	session = data.get('session')

	# ensure session is valid
	if not session or chat_history.get(session, None) is None:
		json_response = {'error': 'session not found'}
		log_actions("error", "/api/chat", data, str(json_response), request_method="POST", error="400")
		return jsonify(json_response), 400
	config = load_config(session, CONFIG_FILE)
	rag_service = rag_services[session]

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

	# If RAG is enabled, retrieve relevant documents and augment the prompt
	sources = []
	if rag_service.config.get('enabled', False) and rag_service.config.get('ready', False):
		try:
			# parse users messages so far
			msg_count[session] += 1
			usr_msgs[session] += "msg" + str(msg_count[session]) + ":  " + chat_history[session][-1]['content'] + " | "
			# call llm
			rag_prompt, error = call_llm([
				{'role': 'system', 'content': 'These are messages sent by the user; I need you to summarize them. Be sure to mention all topics and use lots of adjectives (eg: very tall green tree, big red hot air balloon, long exciting international trip). **HIGH IMPORTANCE:** Answer in only in as few as possible comma separated nouns (5 nouns max).'},
				{'role': 'user', 'content': usr_msgs[session]},
			], session)
			if error:
				json_response = {'error': f'Failed to get response= {error}'}
				log_actions(session, "/api/chat", data, str(json_response), request_method="POST", error="500")
				return jsonify(json_response), 500

			# prompt rag
			if rag_prompt is not None:
				augmented_prompt, docs = rag_service.query_documents(str(rag_prompt.get('content')))

				if len(docs) <= 0:
					chat_history[session][-1]['content'] = message
					log_actions(session, "call to RAG", rag_prompt.get('content'), augmented_prompt)
				else:
					chat_history[session][-1]['content'] = augmented_prompt +  "\nUser Question: " + message
					sources = docs
					log_actions(session, "call to RAG", rag_prompt.get('content'), augmented_prompt)
		except Exception as e:
			log_actions(session, "/api/chat", data, str({'error': str(e)}), request_method="POST", error="500")

	# call the llm
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
		'tokenCount': token_count,
		'sources': sources
	}
	log_actions(session, "/api/chat", data, str(json_response), request_method="POST")
	return jsonify(json_response)


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
	"""Start a new chat session."""
	global usr_msgs, msg_count
	data = request.json or {}
	session = str(data.get('session'))

	if not session or chat_history.get(session, None) is None:
		new_session = str(uuid.uuid4())
		chat_history[new_session] = []
		rag_services[new_session] = RAGService(new_session)
		rag_services[new_session].save_config()
		usr_msgs[new_session] = usr_msgs["default"]
		msg_count[new_session] = msg_count["default"]

		json_response = {
			'success': True,
			'session': new_session,
			'tokenCount': get_total_tokens(chat_history[new_session])
		}
		log_actions(new_session, "/api/new-chat", data, str(json_response), request_method="POST", is_new_chat=True)
		return jsonify(json_response)
	else:
		config = load_config(session, CONFIG_FILE)
		chat_history[session] = []
		chat_history[session].append({'role': 'system', 'content': config['systemPrompt']})
		usr_msgs[session] = usr_msgs["default"]
		msg_count[session] = msg_count["default"]

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
	"""Append data to the log file."""
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


# =============================================================================
# RAG API ENDPOINTS
# =============================================================================
@app.route('/api/rag/config', methods=['GET', 'POST'])
def rag_config_endpoint():
	"""
	GET: Retrieve current RAG configuration.
	POST: Update RAG configuration.
	"""
	if request.method == 'GET':
		# get session
		session = request.headers.get('session')
		if not session or chat_history.get(session, None) is None:
			json_response = {'success': False, 'error': 'session not found'}
			log_actions("error", "/api/config", json_response, str(json_response), request_method=request.method)
			return jsonify(json_response)

		rag_service = rag_services.get(session)

		response = {
			'enabled': rag_service.config.get('enabled', True),
			'collectionName': rag_service.config.get('collectionName', ''),
			'topK': rag_service.config.get('topK', 4),
			'similarityThreshold': rag_service.config.get('similarityThreshold', 0.25),
			'chromaPath': rag_service.config.get('chromaPath', './chroma_data'),
			'ready': rag_service.config.get('ready', False),
			'chunkSize': rag_service.config.get('chunkSize', 500),
			'chunkOverlap': rag_service.config.get('chunkOverlap', 50)
		}
		log_actions(session, "/api/rag/config", "", str(response), request_method=request.method)
		return jsonify(response)
	else:
		data = request.json
		session = data.get('session')
		if not session or chat_history.get(session, None) is None:
			json_response = {'success': False, 'error': 'session not found'}
			log_actions("error", "/api/config", data, str(json_response), request_method=request.method, error="400")
			return jsonify(json_response), 400
		rag_service = rag_services.get(session)

		if 'enabled' in data:
			rag_service.config['enabled'] = bool(data['enabled'])

		if 'collectionName' in data:
			rag_service.config['collectionName'] = data['collectionName']

		if 'topK' in data:
			rag_service.config['topK'] = int(data['topK'])

		if 'similarityThreshold' in data:
			rag_service.config['similarityThreshold'] = float(data['similarityThreshold'])

		if 'chromaPath' in data:
			rag_service.config['chromaPath'] = data['chromaPath']

		if 'chunkSize' in data:
			rag_service.config['chunkSize'] = int(data['chunkSize'])

		if 'chunkOverlap' in data:
			rag_service.config['chunkOverlap'] = int(data['chunkOverlap'])

		rag_service.save_config()

		response = {'success': True}
		log_actions(session, "/api/rag/config", request.json, str(response), request_method=request.method)
		return jsonify(response)


@app.route('/api/rag/upload', methods=['POST'])
def rag_upload():
	"""Upload a new file for RAG processing."""
	# get session
	session = request.headers.get('session')
	if not session or chat_history.get(session, None) is None:
		json_response = {'success': False, 'error': 'session not found'}
		log_actions("error", "/api/config", json_response, str(json_response), request_method=request.method)
		return jsonify(json_response)
	rag_service = rag_services.get(session)

	if not rag_service.config.get('enabled', False):
		response = {'success': False, 'error': 'RAG not enabled'}
		log_actions(session, "/api/rag/upload", request, str(response), request_method="POST", error="403")
		return jsonify(response), 403

	if 'file' in request.files:
		# Handle file upload
		file = request.files['file']
		if file.filename == '':
			response = {'success': False, 'error': 'No file selected'}
			log_actions(session, "/api/rag/upload", file, str(response), request_method="POST", error="400")
			return jsonify(response), 400

		# Save file temporarily
		import tempfile
		with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
			file.save(tmp.name)
			tmp_path = tmp.name

		try:
			result = rag_service.upload_document(tmp_path, file_name=file.filename)
			log_actions(session, "/api/rag/upload", file, str(result), request_method="POST")
			return jsonify(result)
		finally:
			if os.path.exists(tmp_path):
				os.unlink(tmp_path)

	elif request.is_json:
		# Handle direct content upload
		data = request.json
		content = data.get('content', '')
		filename = data.get('filename', 'document.txt')

		result = rag_service.upload_document(
			file_path="",
			file_content=content,
			file_name=filename
		)
		log_actions(session, "/api/rag/upload", data, str(result))
		return jsonify(result)

	response = {'success': False, 'error': 'No file or content provided'}
	log_actions(session, "/api/rag/upload", request, str(response), request_method="POST", error="400")
	return jsonify(response, 400)


@app.route('/api/rag/documents', methods=['GET'])
def rag_list_documents():
	"""List all uploaded documents."""
	# get session
	session = request.headers.get('session')
	if not session or chat_history.get(session, None) is None:
		json_response = {'success': False, 'error': 'session not found'}
		log_actions("error", "/api/config", json_response, str(json_response), request_method=request.method)
		return jsonify(json_response)
	rag_service = rag_services.get(session)

	documents = rag_service.list_documents()

	response = {'documents': documents}
	log_actions(session, "/api/rag/documents", "", str(response))
	return jsonify(response)


@app.route('/api/rag/document/<doc_id>', methods=['DELETE'])
def rag_delete_document(doc_id):
	"""Delete a specific document."""
	session = request.json.get('session')
	if not session or chat_history.get(session, None) is None:
		json_response = {'success': False, 'error': 'session not found'}
		log_actions("error", "/api/config", request.json, str(json_response), request_method=request.method, error="400")
		return jsonify(json_response), 400
	rag_service = rag_services.get(session)

	result = rag_service.delete_document(doc_id)

	log_actions(session, "/api/rag/document/" + doc_id, doc_id, str(result), request_method="DELETE")
	return jsonify(jsonify(result))


@app.route('/api/rag/clear', methods=['POST'])
def rag_clear():
	"""Clear all documents from the RAG system."""
	session = request.json.get('session')
	if not session or chat_history.get(session, None) is None:
		json_response = {'success': False, 'error': 'session not found'}
		log_actions("error", "/api/config", request.json, str(json_response), request_method=request.method, error="400")
		return jsonify(json_response), 400
	rag_service = rag_services.get(session)

	if len(rag_service.list_documents()) > 0:
		result = rag_service.clear_all()

		log_actions(session, "/api/rag/clear", "", str(result), request_method="POST")
		return jsonify(result)
	else:
		result = {'success': True, 'message': 'All documents cleared'}
		log_actions(session, "/api/rag/clear", "", str(result), request_method="POST")
		return jsonify(result)


@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
	"""Get RAG statistics."""
	session = request.json.get('session')
	if not session or chat_history.get(session, None) is None:
		json_response = {'success': False, 'error': 'session not found'}
		log_actions("error", "/api/config", request.json, str(json_response), request_method=request.method, error="400")
		return jsonify(json_response), 400
	rag_service = rag_services.get(session)
	stats = rag_service.get_stats()

	log_actions(session, "/api/rag/clear", "", str(stats), request_method="GET")
	return jsonify(stats)


if __name__ == '__main__':
	USER_CONFIG_DEFAULTS = load_config("default", CONFIG_FILE)
	# Run the Flask application
	app.run(host='0.0.0.0', port=load_config("default", CONFIG_FILE).get("port"), debug=True)

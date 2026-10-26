"""
AI Chat Agent - Single File Implementation
Combines Flask API, Configuration Loader, and RAG Service
"""

# The rag-service doesn't always return a chunk (as expected). However, when it does, it does not look at previous chats
# Must add the following:
# todo: 1. pass in previous chats to the search query for rag
# todo: 2. let the llm review document titles and "demand" a chunk from the rag server even if it is outside of the similarity threshold.

import os
import json
import threading
from typing import List, Optional, Tuple

# Flask and web imports
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# RAG imports
from model.RagService import RAGService

# Load environment variables for LLM configuration
load_dotenv()

# set up directory for API
app = Flask(__name__, static_folder='static', static_url_path='/static')

# === Configuration Constants ===
CONFIG_FILE = 'config.json'
RAG_CONFIG_FILE = 'rag_config.json'
LOG_FILE = 'log.txt'

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

# === Global State ===
config = {}
rag_service = RAGService()
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


def call_llm(messages: List[dict]) -> Tuple[dict, Optional[str]]:
	"""Call the OpenAI-compatible LLM API."""
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


def log_actions(path: str, request_payload: any, response_payload: any, is_new_chat: bool = False, request_method: str = "unknown", error: str = "200"):
	"""Log API actions to the log file."""
	if is_new_chat:
		with open(LOG_FILE, 'w') as log:
			log.write(
				f"{path}\n" +
				f"request: json={request_payload} | method={request_method}\n" +
				f"response: json={response_payload} | error={error}\n\n"
			)
	else:
		with open(LOG_FILE, 'a') as log:
			log.write(
				f"{path}\n" +
				f"request: json={request_payload} | method={request_method}\n" +
				f"response: json={response_payload} | error={error}\n\n"
			)


def load_config(config_file: str, default_config: dict) -> dict:
	"""Load user config from file, fall back to defaults."""
	if os.path.exists(config_file):
		try:
			with open(config_file, 'r') as f:
				configurations = json.load(f)
				for key, value in default_config.items():
					if key not in configurations:
						configurations[key] = value
		except (json.JSONDecodeError, IOError):
			configurations = default_config.copy()
			save_config(configurations, config_file)
	else:
		configurations = default_config.copy()
		save_config(configurations, config_file)

	return configurations


def save_config(configuration: dict, config_file: str):
	"""Save configuration to a JSON file with thread-safe locking."""
	with threading.Lock():
		with open(config_file, 'w') as f:
			json.dump(configuration, f, indent=4)


def load_user_config():
	"""Load user configuration from config.json with defaults."""
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
	save_user_config()


def save_user_config():
	"""Save user configuration to config.json."""
	global config
	with config_lock:
		with open(CONFIG_FILE, 'w') as f:
			json.dump(config, f, indent=4)


# =============================================================================
# FLASK API ENDPOINTS
# =============================================================================
@app.route('/')
def index():
	"""Serve the main chat interface."""
	log_actions("/", "", "static/index.html", request_method="GET", is_new_chat=True)
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
		save_user_config()

		response = {'success': True}
		log_actions("/api/config", data, str(response), request_method=request.method)
		return jsonify(response)


@app.route('/api/chat', methods=['POST'])
def chat():
	"""Handle chat messages with optional RAG integration."""
	global rag_service

	data = request.json
	message = data.get('message', '')
	chat_id = data.get('chatId', 'default')

	if not message:
		json_response = {'error': 'Message is required'}
		log_actions("/api/chat", data, str(json_response), request_method="POST", error="400")
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

	# If RAG is enabled, retrieve relevant documents and augment the prompt
	json_response = {}
	if rag_service.config.get('enabled', False):
		if rag_service.config.get('enabled', False) and rag_service.config.get('ready', False):
			try:
				augmented_prompt, docs = rag_service.query_documents(message)
				chat_history[chat_id][-1]['content'] = augmented_prompt
				json_response['sources'] = docs
			except Exception as e:
				log_actions("/api/chat", data, str({'error': str(e)}), error="500")

	# call the llm
	full_response, error = call_llm(chat_history[chat_id])

	if error:
		json_response = {'error': f'Failed to get response= {error}'}
		log_actions("/api/chat", data, str(json_response), request_method="POST", error="500")
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
	log_actions("/api/chat", data, str(json_response), request_method="POST")
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
	log_actions("/api/new-chat", data, str(json_response), request_method="POST", is_new_chat=True)
	return jsonify(json_response)


@app.route('/api/token-count', methods=['POST'])
def token_count():
	"""Get the current token count for a chat."""
	data = request.json
	chat_id = data.get('chatId', 'default')

	if chat_id not in chat_history:
		json_response = jsonify({'tokenCount': 0})
		log_actions("/api/token-count", data, str(json_response), request_method="POST")
		return json_response

	json_response = {'tokenCount': get_total_tokens(chat_history[chat_id])}
	log_actions("/api/token-count", data, str(json_response), request_method="POST")
	return jsonify(json_response)


@app.route('/api/log', methods=['POST'])
def log_data():
	"""Append data to the log file."""
	data = request.json

	with open(LOG_FILE, 'a') as log:
		log.write(data.get("log") + "\n\n")

	json_response = {"success": True}
	log_actions('/api/log', data, str(json_response), request_method="POST")
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
	global rag_service

	if request.method == 'GET':
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
		log_actions("/api/rag/config", "", str(response), request_method=request.method)
		return jsonify(response)
	else:
		data = request.json
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
		log_actions("/api/rag/config", request.json, str(response), request_method=request.method)
		return jsonify(response)


@app.route('/api/rag/upload', methods=['POST'])
def rag_upload():
	"""Upload a new file for RAG processing."""
	global rag_service

	if not rag_service.config.get('enabled', False):
		response = {'success': False, 'error': 'RAG not enabled'}
		log_actions("/api/rag/upload", request, str(response), request_method="POST", error="403")
		return jsonify(response), 403

	if 'file' in request.files:
		# Handle file upload
		file = request.files['file']
		if file.filename == '':
			response = {'success': False, 'error': 'No file selected'}
			log_actions("/api/rag/upload", file, str(response), request_method="POST", error="400")
			return jsonify(response), 400

		# Save file temporarily
		import tempfile
		with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
			file.save(tmp.name)
			tmp_path = tmp.name

		try:
			result = rag_service.upload_document(tmp_path, file_name=file.filename)
			log_actions("/api/rag/upload", file, str(result), request_method="POST")
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
			file_path=None,
			file_content=content,
			file_name=filename
		)
		log_actions("/api/rag/upload", data, str(result))
		return jsonify(result)

	response = {'success': False, 'error': 'No file or content provided'}
	log_actions("/api/rag/upload", request, str(response), request_method="POST", error="400")
	return jsonify(response, 400)


@app.route('/api/rag/documents', methods=['GET'])
def rag_list_documents():
	"""List all uploaded documents."""
	documents = rag_service.list_documents()

	response = {'documents': documents}
	log_actions("/api/rag/documents", "", str(response))
	return jsonify(response)


@app.route('/api/rag/document/<doc_id>', methods=['DELETE'])
def rag_delete_document(doc_id):
	"""Delete a specific document."""
	global rag_service
	result = rag_service.delete_document(doc_id)

	log_actions("/api/rag/document/" + doc_id, doc_id, str(result), request_method="DELETE")
	return jsonify(jsonify(result))


@app.route('/api/rag/clear', methods=['POST'])
def rag_clear():
	"""Clear all documents from the RAG system."""
	if len(rag_service.list_documents()) > 0:
		result = rag_service.clear_all()

		log_actions("/api/rag/clear", "", str(result), request_method="POST")
		return jsonify(result)
	else:
		result = {'success': True, 'message': 'All documents cleared'}
		log_actions("/api/rag/clear", "", str(result), request_method="POST")
		return jsonify(result)


@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
	"""Get RAG statistics."""
	stats = rag_service.get_stats()

	log_actions("/api/rag/clear", "", str(stats), request_method="GET")
	return jsonify(stats)


if __name__ == '__main__':
	# Load user config on startup
	load_user_config()
	# Run the Flask application
	app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']), debug=True)

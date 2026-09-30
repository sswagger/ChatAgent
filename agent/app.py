import datetime
import os
import json
import time
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
MCP_CONFIG_FILE = 'mcp_config.json'
LOG_FILE = 'log.txt'

# Lock for thread-safe config writes
config_lock = threading.Lock()

# Configuration from environment (LLM settings stay in .env, user settings in config.json)
LLM_API_URL = os.getenv('LLM_API_URL', '')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')

# User-editable config defaults
USER_CONFIG_DEFAULTS = {
	'systemPrompt': os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant.'),
	'bufferSize': int(os.getenv('BUFFER_SIZE', '10')),
	'temperature': float(os.getenv('TEMPERATURE', '0.7')),
	'port': int(os.getenv('PORT', '8080'))
}

# MCP config defaults
MCP_CONFIG_DEFAULTS = {
	'enabled': False,
	'url': os.getenv('MCP_URL', 'http://restaurant-mcp:8000/mcp'),
	'connectionStatus': 'disconnected',
	'tools': []
}

# Global config dictionaries
chat_history = {}
mcp_config: dict
config: dict
mcp_id: str = ""

#=== Built-in Tools ===#
def get_capabilities():
	try:
		tools = mcp_config['tools']
		built_in_tools = [
			{'name': 'get_capabilities', 'description': 'Get a list of all available tools from the MCP server. Use this when someone asks what tools are available.'},
			{'name': 'get_time', 'description': 'Gets the current datetime'},
			{'name': 'get_config', 'description': 'Gets the settings for the chat agent'},
			{'name': 'get_timezone', 'description': 'Gets the users timezone'},
			{'name': 'log_message', 'description': 'Adds a message to the log file', 'inputSchema': {
                "additionalProperties": False,
                "properties": {
                    "sql": {
                        "type": "text",
                        "description": "(str) The text to log"
                    }
                },
                "required": [
                    "text"
                ],
                "type": "object"
			}}
		]

		tool_list = '\n'.join([f"- {t.get('name', 'Unknown')}: {t.get('description', 'No description')} | params: {t.get('inputSchema', 'No parameters')}" for t in tools])
		tool_list += '\n'.join([f"- {t.get('name', 'Unknown')}: {t.get('description', 'No description')} | params: {t.get('inputSchema', 'No parameters')}" for t in built_in_tools])
		response = f'The MCP server at {mcp_config['url']} has {len(tools)} tools:\n{tool_list}'
	except Exception:
		response = f"Error executing tool get_capabilities: An issue with retrieving tools"

	return response

def get_time():
	return str(datetime.datetime.now())

def get_timezone():
	return time.tzname

def get_config():
	return config, mcp_config

def log_text(text: str):
	try:
		with open(LOG_FILE, 'a') as log:
			log.write(
				f"a note from the LLM\n"+
				f"note: {text}\n\n"
			)
		return "success"
	except Exception:
		return "failed to log message"

#=== Agent Functions ===#
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


def call_llm(messages, use_tools=False, include_capabilities=False):
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

	if use_tools and mcp_config.get('enabled', False):
		# Convert MCP tools to OpenAI tools format
		tools = []
		for tool in mcp_config.get('tools', []):
			tools.append({
				'type': 'function',
				'function': {
					'name': tool.get('name', ''),
					'description': tool.get('description', ''),
					'parameters': tool.get('parameters', {})
				}
			})

		# Add built-in tools if requested
		if include_capabilities:
			tools.append({
				'type': 'function',
				'function': {
					'name': 'get_capabilities',
					'description': 'Get a list of all available tools from the MCP server. Use this when someone asks what tools are available.',
					'parameters': {
						'type': 'object',
						'properties': {}
					}
				}
			})
			tools.append({
				'type': 'function',
				'function': {
					'name': 'get_time',
					'description': 'Gets the current datetime',
					'parameters': {
						'type': 'object',
						'properties': {}
					}
				}
			})
			tools.append({
				'type': 'function',
				'function': {
					'name': 'get_config',
					'description': 'Gets the settings for the chat agent',
					'parameters': {
						'type': 'object',
						'properties': {}
					}
				}
			})
			tools.append({
				'type': 'function',
				'function': {
					'name': 'get_timezone',
					'description': 'Gets the users timezone',
					'parameters': {
						'type': 'object',
						'properties': {}
					}
				}
			})
			tools.append({
				'type': 'function',
				'function': {
					'name': 'log_message',
					'description': 'Adds a message to the log file',
					'parameters': {
						'type': 'object',
						'properties': {'text': {'type': 'string', 'description': '(str) message you want to log'}}
					}
				}
			})
		payload['tools'] = tools

	try:
		response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=100)
		response.raise_for_status()
		result = response.json()

		log_actions("call to LLM", payload, result, error=str(response.status_code))
		return result.get('choices', [{}])[0].get('message', {}), None
	except Exception as e:
		return None, str(e)


def load_config(config_file: str, default_config):
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
	"""Save config to file."""
	with threading.Lock():
		with open(config_file, 'w') as f:
			json.dump(configuration, f, indent=4)


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


#=== MCP Functions ===#
def add_mcp_capabilities_instruction(prompt):
	"""Add MCP capabilities instruction to system prompt."""
	mcp_url = mcp_config.get('url', '')
	tool_count = len(mcp_config.get('tools', []))

	instruction = f"""
    
		IMPORTANT: This application has access to an MCP (Model Context Protocol) server at {mcp_url} that provides {tool_count} tools.
		
		When asked what the MCP server can do, or what tools are available, you MUST call the get_capabilities function to retrieve the list of available tools. Do NOT just make up a response.
		
		The get_capabilities function will return a list of all available tools with their descriptions. Use this information to answer questions about the robot's capabilities.
"""

	# Check if already has MCP instruction
	if 'MCP (Model Context Protocol)' not in prompt:
		return prompt.rstrip() + instruction

	return prompt


def request_mcp(json_body, mcp_id=None) -> tuple[requests.Response, dict] | str:
	headers = {
		'Content-Type': 'application/json',
		'Accept': 'application/json, text/event-stream'
	}
	if mcp_id is not None:
		headers["mcp-session-id"] = mcp_id

	try:
		response = requests.post(
			url=mcp_config["url"],
			json=json_body,
			headers=headers,
			timeout=10
		)
		raw_text = response.text
		# Extract JSON from 'data:' field
		json_data_start = raw_text.find('data: ')

		if json_data_start != -1:
			json_str = raw_text[json_data_start + 6:].strip()  # Skip 'data: ' prefix
			try:
				json_response = json.loads(json_str)
				log_actions("call to MCP", json_body, json_response)

				return response, json_response
			except json.JSONDecodeError as e:
				mcp_config['connectionStatus'] = 'error'
				mcp_config['lastError'] = str(e)
				save_config(mcp_config, MCP_CONFIG_FILE)
				return "error: failed to parse response into json"
		else:
			mcp_config['connectionStatus'] = 'error'
			mcp_config['lastError'] = raw_text
			save_config(mcp_config, MCP_CONFIG_FILE)
			return "error: no data to parse into json"

	except Exception as ex:
		return str(ex)


def init_mcp_connection():
	"""Initialize MCP connection if enabled."""
	if not mcp_config.get('enabled', False):
		return None

	try:
		url = mcp_config.get('url', '')
		if not url:
			return None

		# HTTP/HTTPS connection (standard MCP JSON-RPC)
		return {'type': 'http', 'url': url}
	except Exception as e:
		mcp_config['connectionStatus'] = 'error'
		mcp_config['lastError'] = str(e)
		save_config(mcp_config, MCP_CONFIG_FILE)
		return None


def execute_mcp_tool(tool_name, tool_input):
	"""Execute an MCP tool and return the result."""
	if not mcp_config.get('enabled', False):
		return None, 'MCP is not enabled'
	url = mcp_config.get('url', '')
	if not url:
		return None, 'MCP URL not configured'

	try:
		# HTTP/HTTPS connection with JSON-RPC 2.0 (standard MCP)
		# Build headers, including auth only if API key is set
		response = request_mcp({
				'jsonrpc': '2.0',
				'method': 'tools/call',
				'params': {
			        'name': tool_name,
			        'arguments': tool_input
				},
				'id': 1
			},
			mcp_id
		)
		if type(response) is str:
			return jsonify({'success': False, 'error': 'failed to execute tool call | '.format(response)}), 400

		return response[1].get('result').get('structuredContent').get('result'), None
	except Exception as e:
		return None, str(e)


def handle_tool_calls(messages, full_response):
	"""Handle tool calls from LLM response."""
	message_content = full_response.get('content', '')
	tool_calls = full_response.get('tool_calls', [])

	# If there are tool calls, execute them and continue
	if tool_calls:
		for tool_call in tool_calls:
			tool_name = tool_call.get('function', {}).get('name', '')
			tool_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
			error = None

			# Execute the tool
			if tool_name == 'get_capabilities':
				# Special handling for get_capabilities - don't call MCP server
				tool_response = get_capabilities()
			elif tool_name == 'get_time':
				# Special handling for get_time - don't call MCP server
				tool_response = get_time()
			elif tool_name == 'get_config':
				# Special handling for get_config - don't call MCP server
				tool_response = get_config()
			elif tool_name == 'get_timezone':
				# Special handling for get_timezone - don't call MCP server
				tool_response = get_timezone()
			elif tool_name == 'log_message':
				# Special handling for log_message - don't call MCP server
				tool_response = log_text(tool_args.get('text', "no text provided"))
			else:
				tool_result, error = execute_mcp_tool(tool_name, tool_args)
				tool_response = tool_result

			if error:
				tool_response = f"Error executing tool {tool_name}: {tool_response.get('error')} | {error}"

			# Add tool response to history
			messages.append({
				'role': 'assistant',
				'content': None,
				'tool_calls': [tool_call]
			})
			messages.append({
				'role': 'tool',
				'name': tool_name,
				'content': str(tool_response)
			})

			# Get next response from LLM
			next_response, error = call_llm(messages, use_tools=False)
			if error:
				return f"Error: {error}"
			if next_response:
				return next_response.get('content', '')

		return message_content

	return message_content


# Load configs on startup - load mcp_config first for add_mcp_capabilities_instruction
config = load_config(CONFIG_FILE, USER_CONFIG_DEFAULTS)
mcp_config = load_config(MCP_CONFIG_FILE, MCP_CONFIG_DEFAULTS)

# Add MCP capabilities instruction to system prompt if MCP is enabled
if mcp_config.get('enabled', False):
	mcp_config['systemPrompt'] = add_mcp_capabilities_instruction(config.get('systemPrompt', ''))

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
		save_config(config, CONFIG_FILE)

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

	# Get response from LLM (with tool calling enabled if MCP is enabled)
	use_tools = mcp_config.get('enabled', False) and mcp_config.get('connectionStatus', False)
	# Include get_capabilities function when MCP is enabled
	full_response, error = call_llm(chat_history[chat_id], use_tools=use_tools, include_capabilities=use_tools)

	if error:
		json_response = {'error': f'Failed to get response= {error}'}
		log_actions("/api/chat", data, str(json_response), error="500")
		return jsonify(json_response), 500

	# Handle tool calls if any
	response_content = handle_tool_calls(chat_history[chat_id], full_response)

	# Add assistant response
	chat_history[chat_id].append({'role': 'assistant', 'content': response_content})

	# Calculate token usage
	token_count = get_total_tokens(chat_history[chat_id])

	json_response = {
		'response': response_content,
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


#=== MCP ===#
@app.route('/api/mcp/config', methods=['GET', 'POST'])
def mcp_config_endpoint():
	"""Get or update MCP configuration."""
	global mcp_config
	if request.method == 'GET':
		return jsonify({
			'enabled': mcp_config.get('enabled', False),
			'url': mcp_config.get('url', MCP_CONFIG_DEFAULTS['url']),
			'connectionStatus': mcp_config.get('connectionStatus', 'disconnected'),
			'tools': mcp_config.get('tools', [])
		})
	else:
		data = request.json
		if 'enabled' in data:
			mcp_config['enabled'] = bool(data['enabled'])
		if 'url' in data:
			mcp_config['url'] = data['url']
		save_config(mcp_config, MCP_CONFIG_FILE)
		return jsonify({'success': True})


@app.route('/api/mcp/connect', methods=['POST'])
def mcp_connect():
	"""Connect to MCP server."""
	global mcp_config, mcp_id

	url = mcp_config.get('url', '')
	if not url:
		return jsonify({'success': False, 'error': 'MCP URL not configured'}), 400

	try:
		# HTTP/HTTPS connection (standard MCP JSON-RPC)
		# Test connection by calling tools/list endpoint
		try:
			response = request_mcp(
				{
				  "jsonrpc": "2.0",
				  "id": 1,
				  "method": "initialize",
				  "params": {
				    "protocolVersion": "2024-11-05",
				    "capabilities": {},
				    "clientInfo": {
				      "name": "Chat Assistant",
				      "version": "1.0.0"
				    }
				  }
				}
			)
			if type(response) is str:
				return jsonify({'success': False, 'error': 'initial connection request | {}'.format(response)}), 400

			mcp_id = response[0].headers["mcp-session-id"]
			request_mcp(
				{
					"jsonrpc": "2.0",
					"method": "notifications/initialized",
					"params": {}
				},
				mcp_id
			)
			if type(response) is str:
				return jsonify({'success': False, 'error': 'second connection request | {}'.format(response)}), 400

			mcp_config['connectionStatus'] = 'connected'
			mcp_config['lastError'] = None
			save_config(mcp_config, MCP_CONFIG_FILE)
			return jsonify({'success': True, 'message': 'Connected to MCP server', 'sessionId': mcp_id})

		except Exception as e:
			mcp_config['connectionStatus'] = 'error'
			mcp_config['lastError'] = str(e)
			save_config(mcp_config, MCP_CONFIG_FILE)
			return jsonify({'success': False, 'error': str(e)}), 500

	except Exception as e:
		mcp_config['connectionStatus'] = 'error'
		mcp_config['lastError'] = str(e)
		save_config(mcp_config, MCP_CONFIG_FILE)
		return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mcp/disconnect', methods=['POST'])
def mcp_disconnect():
	"""Disconnect from MCP server."""
	mcp_config['connectionStatus'] = 'disconnected'
	mcp_config['tools'] = []
	save_config(mcp_config, MCP_CONFIG_FILE)
	return jsonify({'success': True, 'message': 'Disconnected from MCP server'})


@app.route('/api/mcp/tools', methods=['GET'])
def mcp_tools():
	"""Get list of available MCP tools."""
	# Try to fetch tools from MCP server
	try:
		response = request_mcp({
				'jsonrpc': '2.0',
				'method': 'tools/list',
				'id': 1
			},
			mcp_id
		)

		if type(response) is str:
			return jsonify({'success': False, 'error': 'failed to get tools list | {}'.format(response)}), 400

		# Extract tools from result
		tools = []
		if 'result' in response[1] and isinstance(response[1]['result'], dict):
			tools = response[1]['result'].get('tools', [])
		elif 'result' in response[1]:
			tools = response[1]['result'] if isinstance(response[1]['result'], list) else []

		if not tools:
			tools = response[1].get('tools', [])

		mcp_config['tools'] = tools
		save_config(mcp_config, MCP_CONFIG_FILE)
	except Exception as e:
		return jsonify({'tools': [], 'error': str(e)})

	return jsonify({'tools': tools})


@app.route('/api/mcp/execute', methods=['POST'])
def mcp_execute():
	"""Execute an MCP tool."""
	data = request.json
	tool_name = data.get('tool', '')
	tool_input = data.get('input', {})

	if not tool_name:
		return jsonify({'success': False, 'error': 'Tool name required'}), 400

	output, error = execute_mcp_tool(tool_name, tool_input)

	if error:
		return jsonify({'success': False, 'error': error}), 500

	return jsonify({'success': True, 'output': output})


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']), debug=True)

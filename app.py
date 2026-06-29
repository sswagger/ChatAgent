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
MCP_CONFIG_FILE = 'mcp_config.json'

# Lock for thread-safe config writes
config_lock = threading.Lock()
mcp_lock = threading.Lock()

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

# MCP config defaults - updated to use robot MCP server
MCP_CONFIG_DEFAULTS = {
    'enabled': False,
    'url': os.getenv('MCP_URL', 'http://10.24.10.62:8000/mcp'),
    'apiKey': os.getenv('MCP_API_KEY', ''),
    'connectionStatus': 'disconnected',
    'tools': []
}

# Global config dictionaries
config = {}
mcp_config = {}


def add_mcp_capabilities_instruction(prompt):
    """Add MCP capabilities instruction to system prompt."""
    mcp_url = mcp_config.get('url', '')
    tool_count = len(mcp_config.get('tools', []))
    
    instruction = f"""
    
IMPORTANT: This application has access to an MCP (Model Context Protocol) server at {mcp_url} that provides {tool_count} tools for controlling a robot.

When asked what the robot or MCP server can do, or what tools are available, you MUST call the get_capabilities function to retrieve the list of available tools. Do NOT just make up a response.

The get_capabilities function will return a list of all available tools with their descriptions. Use this information to answer questions about the robot's capabilities."""
    
    # Check if already has MCP instruction
    if 'MCP (Model Context Protocol)' not in prompt:
        return prompt.rstrip() + instruction
    
    return prompt


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
    
    # Add MCP capabilities instruction to system prompt if MCP is enabled
    if mcp_config.get('enabled', False):
        config['systemPrompt'] = add_mcp_capabilities_instruction(config.get('systemPrompt', ''))


def save_config():
    """Save config to file."""
    global config
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)


def load_mcp_config():
    """Load MCP config from file, fall back to defaults."""
    global mcp_config
    if os.path.exists(MCP_CONFIG_FILE):
        try:
            with open(MCP_CONFIG_FILE, 'r') as f:
                mcp_config = json.load(f)
                for key, value in MCP_CONFIG_DEFAULTS.items():
                    if key not in mcp_config:
                        mcp_config[key] = value
        except (json.JSONDecodeError, IOError):
            mcp_config = MCP_CONFIG_DEFAULTS.copy()
    else:
        mcp_config = MCP_CONFIG_DEFAULTS.copy()
        save_mcp_config()


def save_mcp_config():
    """Save MCP config to file."""
    global mcp_config
    with mcp_lock:
        with open(MCP_CONFIG_FILE, 'w') as f:
            json.dump(mcp_config, f, indent=4)


def init_mcp_connection():
    """Initialize MCP connection if enabled."""
    if not mcp_config.get('enabled', False):
        return None
    
    try:
        url = mcp_config.get('url', '')
        if not url:
            return None
        
        # HTTP/HTTPS connection (standard MCP JSON-RPC)
        return {'type': 'http', 'url': url, 'apiKey': mcp_config.get('apiKey', '')}
    except Exception as e:
        mcp_config['connectionStatus'] = 'error'
        mcp_config['lastError'] = str(e)
        save_mcp_config()
        return None


def execute_mcp_tool(tool_name, tool_input):
    """Execute an MCP tool and return the result."""
    if not mcp_config.get('enabled', False):
        return None, 'MCP is not enabled'
    
    try:
        url = mcp_config.get('url', '')
        if not url:
            return None, 'MCP URL not configured'
        
        # HTTP/HTTPS connection with JSON-RPC 2.0 (standard MCP)
        # Build headers, including auth only if API key is set
        headers = {'Content-Type': 'application/json'}
        if mcp_config.get('apiKey'):
            headers['Authorization'] = f'Bearer {mcp_config.get("apiKey", "")}'
        
        response = requests.post(
            url,
            json={
                'jsonrpc': '2.0',
                'method': tool_name,
                'params': tool_input,
                'id': 1
            },
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # Check for JSON-RPC error
        if 'error' in result and result['error']:
            error_msg = result['error'].get('message', 'Unknown error') if isinstance(result['error'], dict) else str(result['error'])
            return None, error_msg
        
        return result.get('result'), None
    
    except Exception as e:
        return None, str(e)


# Load configs on startup - load mcp_config first for add_mcp_capabilities_instruction
load_mcp_config()
load_config()

# In-memory chat storage
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
    """Calculate total token count for messages."""
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
        
        # Add get_capabilities tool if requested
        if include_capabilities:
            tools.append({
                'type': 'function',
                'function': {
                    'name': 'get_capabilities',
                    'description': 'Get a list of all available tools from the MCP server. Use this when someone asks what the robot or MCP server can do, or what tools are available.',
                    'parameters': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            })
        payload['tools'] = tools

    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=100)
        response.raise_for_status()
        print("response: ", response)
        result = response.json()
        print("result", result)
        return result.get('choices', [{}])[0].get('message', {}), None
    except Exception as e:
        return None, str(e)


def execute_mcp_capabilities():
    """Execute the get_capabilities tool and return the result."""
    try:
        url = mcp_config.get('url', '')
        if not url:
            return 'Error: MCP URL not configured'
        
        headers = {'Content-Type': 'application/json'}
        if mcp_config.get('apiKey'):
            headers['Authorization'] = f'Bearer {mcp_config.get("apiKey", "")}'
        
        response = requests.post(
            url,
            json={
                'jsonrpc': '2.0',
                'method': 'tools/list',
                'id': 1
            },
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        # Check for JSON-RPC error
        if 'error' in result and result['error']:
            error_msg = result['error'].get('message', 'Unknown error') if isinstance(result['error'], dict) else str(result['error'])
            return f'Error: {error_msg}'
        
        # Extract tools from result
        tools = []
        if 'result' in result and isinstance(result['result'], dict):
            tools = result['result'].get('tools', [])
        elif 'result' in result:
            tools = result['result'] if isinstance(result['result'], list) else []
        
        if not tools:
            tools = result.get('tools', [])
        
        # Update stored tools
        mcp_config['tools'] = tools
        save_mcp_config()
        
        if tools:
            tool_list = '\n'.join([f"- {t.get('name', 'Unknown')}: {t.get('description', 'No description')}" for t in tools])
            return f'The MCP server at {url} has {len(tools)} tools:\n\n{tool_list}'
        else:
            return f'The MCP server at {url} returned no tools.'
        
    except Exception as e:
        return f'Error: {str(e)}'


def handle_tool_calls(messages, full_response, chat_id):
    """Handle tool calls from LLM response."""
    message_content = full_response.get('content', '')
    tool_calls = full_response.get('tool_calls', [])
    
    # If there are tool calls, execute them and continue
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.get('function', {}).get('name', '')
            tool_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
            
            # Execute the tool
            if tool_name == 'get_capabilities':
                # Special handling for get_capabilities - call MCP server directly
                tool_response = execute_mcp_capabilities()
                error = None
            else:
                tool_result, error = execute_mcp_tool(tool_name, tool_args)
                tool_response = tool_result
            
            if error:
                tool_response = f"Error executing tool {tool_name}: {error}"
            
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
        if 'apiKey' in data:
            mcp_config['apiKey'] = data['apiKey']
        save_mcp_config()
        return jsonify({'success': True})


@app.route('/api/mcp/connect', methods=['POST'])
def mcp_connect():
    """Connect to MCP server."""
    global mcp_config
    
    try:
        url = mcp_config.get('url', '')
        if not url:
            return jsonify({'success': False, 'error': 'MCP URL not configured'}), 400
        
        # HTTP/HTTPS connection (standard MCP JSON-RPC)
        # Test connection by calling tools/list endpoint
        try:
            response = requests.post(
                url,
                json={
                    'jsonrpc': '2.0',
                    'method': 'tools/list',
                    'id': 1
                },
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # If we get a successful response, connection is good
            mcp_config['connectionStatus'] = 'connected'
            mcp_config['lastError'] = None
            
            # Try to fetch tools automatically on connect
            tools = []
            if 'result' in result and 'tools' in result['result']:
                tools = result['result']['tools']
            elif 'result' in result:
                # Try to parse result as tools directly
                tools = result['result'] if isinstance(result['result'], list) else []
            
            mcp_config['tools'] = tools
            save_mcp_config()
            return jsonify({'success': True, 'message': 'Connected to MCP server', 'tools': tools})
            
        except Exception as e:
            mcp_config['connectionStatus'] = 'error'
            mcp_config['lastError'] = str(e)
            save_mcp_config()
            return jsonify({'success': False, 'error': str(e)}), 500
        
    except Exception as e:
        mcp_config['connectionStatus'] = 'error'
        mcp_config['lastError'] = str(e)
        save_mcp_config()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/mcp/disconnect', methods=['POST'])
def mcp_disconnect():
    """Disconnect from MCP server."""
    global mcp_config
    mcp_config['connectionStatus'] = 'disconnected'
    save_mcp_config()
    return jsonify({'success': True, 'message': 'Disconnected from MCP server'})


@app.route('/api/mcp/get-capabilities', methods=['GET'])
def mcp_get_capabilities():
    """Get full capabilities/information about MCP server."""
    try:
        url = mcp_config.get('url', '')
        if not url:
            return jsonify({
                'error': 'MCP URL not configured',
                'capabilities': None
            }), 400
        
        headers = {'Content-Type': 'application/json'}
        if mcp_config.get('apiKey'):
            headers['Authorization'] = f'Bearer {mcp_config.get("apiKey", "")}'
        
        # Try to get tools from the MCP server
        response = requests.post(
            url,
            json={
                'jsonrpc': '2.0',
                'method': 'tools/list',
                'id': 1
            },
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        # Check for JSON-RPC error
        if 'error' in result and result['error']:
            return jsonify({
                'error': result['error'].get('message', 'Unknown error'),
                'capabilities': None
            })
        
        # Extract tools from result
        tools = []
        if 'result' in result and isinstance(result['result'], dict):
            tools = result['result'].get('tools', [])
        elif 'result' in result:
            tools = result['result'] if isinstance(result['result'], list) else []
        
        if not tools:
            tools = result.get('tools', [])
        
        # Update stored tools
        mcp_config['tools'] = tools
        save_mcp_config()
        
        return jsonify({
            'error': None,
            'capabilities': {
                'serverUrl': url,
                'tools': tools,
                'toolCount': len(tools),
                'description': f'MCP Server at {url} provides {len(tools)} tools'
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'capabilities': None
        }), 500


@app.route('/api/mcp/tools', methods=['GET'])
def mcp_tools():
    """Get list of available MCP tools."""
    tools = mcp_config.get('tools', [])
    
    if not tools:
        # Try to fetch tools from MCP server
        try:
            url = mcp_config.get('url', '')
            
            # HTTP/HTTPS connection (standard MCP JSON-RPC)
            headers = {'Content-Type': 'application/json'}
            if mcp_config.get('apiKey'):
                headers['Authorization'] = f'Bearer {mcp_config.get("apiKey", "")}'
            
            response = requests.post(
                url,
                json={
                    'jsonrpc': '2.0',
                    'method': 'tools/list',
                    'id': 1
                },
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Check for JSON-RPC error
            if 'error' in result and result['error']:
                return jsonify({'tools': [], 'error': result['error'].get('message', 'Unknown error') if isinstance(result['error'], dict) else str(result['error'])})
            
            # Extract tools from result
            tools = []
            if 'result' in result and isinstance(result['result'], dict):
                tools = result['result'].get('tools', [])
            elif 'result' in result:
                tools = result['result'] if isinstance(result['result'], list) else []
            
            if not tools:
                tools = result.get('tools', [])
            
            mcp_config['tools'] = tools
            save_mcp_config()
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


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat message with optional MCP tool integration."""
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
    
    # Apply buffer size limit
    if len(chat_history[chat_id]) > config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']) + 1:
        chat_history[chat_id] = [chat_history[chat_id][0]] + chat_history[chat_id][-config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']):]
    
    # Get response from LLM (with tool calling enabled if MCP is enabled)
    use_tools = mcp_config.get('enabled', False)
    # Include get_capabilities function when MCP is enabled
    full_response, error = call_llm(chat_history[chat_id], use_tools=use_tools, include_capabilities=use_tools)
    
    if error:
        return jsonify({'error': f'Failed to get response: {error}'}), 500
    
    # Handle tool calls if any
    response_content = handle_tool_calls(chat_history[chat_id], full_response, chat_id)
    
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
    app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']), debug=True)

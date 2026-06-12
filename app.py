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

# MCP config defaults
MCP_CONFIG_DEFAULTS = {
    'enabled': False,
    'url': os.getenv('MCP_URL', 'ws://localhost:3001'),
    'apiKey': os.getenv('MCP_API_KEY', ''),
    'connectionStatus': 'disconnected',
    'tools': []
}

# Global config dictionaries
config = {}
mcp_config = {}


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
        import websockets
        import asyncio
        
        async def connect():
            async with websockets.connect(mcp_config['url']) as websocket:
                return websocket
        return asyncio.run(connect())
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
        import websockets
        import asyncio
        
        async def execute():
            async with websockets.connect(mcp_config['url']) as websocket:
                message = {
                    'type': 'execute',
                    'tool': tool_name,
                    'input': tool_input
                }
                if mcp_config.get('apiKey'):
                    message['apiKey'] = mcp_config['apiKey']
                
                await websocket.send(json.dumps(message))
                response = await websocket.recv()
                result = json.loads(response)
                return result.get('output'), None
        
        return asyncio.run(execute())
    except Exception as e:
        return None, str(e)


# Load configs on startup
load_config()
load_mcp_config()

# In-memory chat storage
chat_history = {}


def estimate_tokens(text):
    """Estimate token count using tiktoken."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
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


def call_llm(messages, use_tools=False):
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
    
    if use_tools and mcp_config.get('enabled', False) and mcp_config.get('tools', []):
        # Convert MCP tools to OpenAI function calling format
        functions = []
        for tool in mcp_config.get('tools', []):
            functions.append({
                'name': tool.get('name', ''),
                'description': tool.get('description', ''),
                'parameters': tool.get('parameters', {})
            })
        payload['functions'] = functions
    
    try:
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}), None
    except Exception as e:
        return None, str(e)


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
            tool_result, error = execute_mcp_tool(tool_name, tool_args)
            
            if error:
                tool_response = f"Error executing tool {tool_name}: {error}"
            else:
                tool_response = tool_result
            
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
        import websockets
        import asyncio
        
        if not mcp_config.get('url'):
            return jsonify({'success': False, 'error': 'MCP URL not configured'}), 400
        
        async def test_connection():
            try:
                async with websockets.connect(mcp_config['url']) as websocket:
                    return True, None
            except Exception as e:
                return False, str(e)
        
        success, error = asyncio.run(test_connection())
        
        if success:
            mcp_config['connectionStatus'] = 'connected'
            mcp_config['lastError'] = None
            save_mcp_config()
            return jsonify({'success': True, 'message': 'Connected to MCP server'})
        else:
            mcp_config['connectionStatus'] = 'error'
            mcp_config['lastError'] = error
            save_mcp_config()
            return jsonify({'success': False, 'error': error}), 500
            
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


@app.route('/api/mcp/tools', methods=['GET'])
def mcp_tools():
    """Get list of available MCP tools."""
    tools = mcp_config.get('tools', [])
    
    if not tools:
        # Try to fetch tools from MCP server
        try:
            import websockets
            import asyncio
            
            async def fetch_tools():
                async with websockets.connect(mcp_config['url']) as websocket:
                    message = {'type': 'list_tools'}
                    if mcp_config.get('apiKey'):
                        message['apiKey'] = mcp_config['apiKey']
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    result = json.loads(response)
                    return result.get('tools', [])
            
            tools = asyncio.run(fetch_tools())
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
    full_response, error = call_llm(chat_history[chat_id], use_tools=use_tools)
    
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
    app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']))
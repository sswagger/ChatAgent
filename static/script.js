// Chat configuration
let config = {
    systemPrompt: '',
    bufferSize: 10,
    temperature: 0.7
};

// MCP configuration
let mcpConfig = {
    enabled: false,
    url: 'ws://localhost:3001',
    apiKey: '',
    connectionStatus: 'disconnected',
    tools: []
};

let chatId = 'default';
let tokenCount = 0;

// DOM Elements
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const systemPromptInput = document.getElementById('systemPrompt');
const bufferSizeInput = document.getElementById('bufferSize');
const temperatureInput = document.getElementById('temperature');
const saveConfigBtn = document.getElementById('saveConfigBtn');
const newChatBtn = document.getElementById('newChatBtn');
const tokenCountDisplay = document.getElementById('tokenCount');
const tokenBar = document.getElementById('tokenBar');

// MCP DOM Elements
const mcpConnectBtn = document.getElementById('mcpConnectBtn');
const mcpDisconnectBtn = document.getElementById('mcpDisconnectBtn');
const mcpUrlInput = document.getElementById('mcpUrl');
const mcpApiKeyInput = document.getElementById('mcpApiKey');
const mcpEnabledCheckbox = document.getElementById('mcpEnabled');
const mcpStatusBadge = document.getElementById('mcpStatusBadge');
const mcpToolsList = document.getElementById('mcpToolsList');
const refreshToolsBtn = document.getElementById('refreshToolsBtn');

// Initialize
async function init() {
    try {
        // Load regular config
        const configResponse = await fetch('/api/config');
        const configData = await configResponse.json();
        config.systemPrompt = configData.systemPrompt;
        config.bufferSize = configData.bufferSize;
        config.temperature = configData.temperature || 0.7;
        systemPromptInput.value = config.systemPrompt;
        bufferSizeInput.value = config.bufferSize;
        temperatureInput.value = config.temperature;
        
        // Load MCP config
        const mcpResponse = await fetch('/api/mcp/config');
        const mcpData = await mcpResponse.json();
        mcpConfig.enabled = mcpData.enabled || false;
        mcpConfig.url = mcpData.url || 'ws://localhost:3001';
        mcpConfig.connectionStatus = mcpData.connectionStatus || 'disconnected';
        mcpConfig.tools = mcpData.tools || [];
        
        mcpUrlInput.value = mcpConfig.url;
        mcpEnabledCheckbox.checked = mcpConfig.enabled;
        updateMcpUI();
        updateMcpToolsList(mcpConfig.tools);
        
        updateTokenDisplay(0);
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

// Update MCP UI based on connection status
function updateMcpUI() {
    if (mcpConfig.connectionStatus === 'connected') {
        mcpStatusBadge.textContent = 'Connected';
        mcpStatusBadge.className = 'status-badge status-connected';
        mcpConnectBtn.style.display = 'none';
        mcpDisconnectBtn.style.display = 'inline-block';
    } else if (mcpConfig.connectionStatus === 'error') {
        mcpStatusBadge.textContent = 'Error';
        mcpStatusBadge.className = 'status-badge status-error';
        mcpConnectBtn.style.display = 'inline-block';
        mcpDisconnectBtn.style.display = 'none';
    } else {
        mcpStatusBadge.textContent = 'Disconnected';
        mcpStatusBadge.className = 'status-badge';
        mcpConnectBtn.style.display = 'inline-block';
        mcpDisconnectBtn.style.display = 'none';
    }
}

// Update MCP tools list
function updateMcpToolsList(tools) {
    if (!tools || tools.length === 0) {
        mcpToolsList.innerHTML = '<p class="no-tools">No tools loaded. Connect to MCP server first.</p>';
        return;
    }
    
    let html = '<div class="tools-grid">';
    for (const tool of tools) {
        html += `
            <div class="tool-item">
                <div class="tool-name">${tool.name || 'Unnamed Tool'}</div>
                <div class="tool-desc">${tool.description || ''}</div>
            </div>
        `;
    }
    html += '</div>';
    mcpToolsList.innerHTML = html;
}

// Connect to MCP server
async function connectMCP() {
    try {
        const url = mcpUrlInput.value.trim();
        const apiKey = mcpApiKeyInput.value.trim();
        
        const response = await fetch('/api/mcp/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: url,
                apiKey: apiKey
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to connect to MCP server');
        }
        
        // Update config
        mcpConfig.url = url;
        mcpConfig.apiKey = apiKey;
        mcpConfig.connectionStatus = 'connected';
        updateMcpUI();
        
        // Fetch tools
        await refreshTools();
        
        addMessageToUI('system', data.message || 'Connected to MCP server');
    } catch (error) {
        mcpConfig.connectionStatus = 'error';
        updateMcpUI();
        addMessageToUI('system', `Error connecting to MCP: ${error.message}`);
    }
}

// Disconnect from MCP server
async function disconnectMCP() {
    try {
        const response = await fetch('/api/mcp/disconnect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            mcpConfig.connectionStatus = 'disconnected';
            updateMcpUI();
            addMessageToUI('system', 'Disconnected from MCP server');
        }
    } catch (error) {
        addMessageToUI('system', `Error disconnecting: ${error.message}`);
    }
}

// Refresh MCP tools
async function refreshTools() {
    try {
        const response = await fetch('/api/mcp/tools');
        const data = await response.json();
        
        if (response.ok) {
            mcpConfig.tools = data.tools || [];
            updateMcpToolsList(mcpConfig.tools);
            addMessageToUI('system', `Loaded ${mcpConfig.tools.length} tools from MCP`);
        } else {
            addMessageToUI('system', `Error fetching tools: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        addMessageToUI('system', `Error fetching tools: ${error.message}`);
    }
}

// Save configuration
async function saveConfig() {
    const newSystemPrompt = systemPromptInput.value;
    const newBufferSize = parseInt(bufferSizeInput.value);
    const newTemperature = parseFloat(temperatureInput.value);
    const newMcpUrl = mcpUrlInput.value.trim();
    const newMcpApiKey = mcpApiKeyInput.value.trim();
    const newMcpEnabled = mcpEnabledCheckbox.checked;
    
    if (newBufferSize < 1 || newBufferSize > 50) {
        alert('Buffer size must be between 1 and 50');
        return;
    }
    
    if (newTemperature < 0 || newTemperature > 2) {
        alert('Temperature must be between 0 and 2');
        return;
    }
    
    try {
        // Save regular config
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                systemPrompt: newSystemPrompt,
                bufferSize: newBufferSize,
                temperature: newTemperature
            })
        });
        
        // Save MCP config
        await fetch('/api/mcp/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: newMcpUrl,
                apiKey: newMcpApiKey,
                enabled: newMcpEnabled
            })
        });
        
        config.systemPrompt = newSystemPrompt;
        config.bufferSize = newBufferSize;
        config.temperature = newTemperature;
        mcpConfig.url = newMcpUrl;
        mcpConfig.apiKey = newMcpApiKey;
        mcpConfig.enabled = newMcpEnabled;
        
        alert('Configuration saved! Start a new chat for system prompt changes to take effect.');
    } catch (error) {
        alert(`Failed to save configuration: ${error.message}`);
    }
}

// Send message to API
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message to UI
    addMessageToUI('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show loading indicator
    const loadingId = 'loading-' + Date.now();
    addLoadingToUI(loadingId);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                chatId: chatId
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get response');
        }

        // Remove loading indicator
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }

        // Add assistant response to UI
        addMessageToUI('assistant', data.response);

        // Update token count
        tokenCount = data.tokenCount;
        updateTokenDisplay(tokenCount);

    } catch (error) {
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
        addMessageToUI('system', `Error: ${error.message}`);
    }
}

// Add message to UI
function addMessageToUI(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add loading indicator to UI
function addLoadingToUI(id) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = id;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<span class="loading"></span>';

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update token display
function updateTokenDisplay(count) {
    tokenCountDisplay.textContent = count;
    // Scale bar: assume max 4096 tokens
    const percentage = Math.min((count / 4096) * 100, 100);
    tokenBar.style.width = `${percentage}%`;
}

// Start new chat
async function startNewChat() {
    try {
        const response = await fetch('/api/new-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chatId: chatId })
        });

        const data = await response.json();

        if (response.ok) {
            chatMessages.innerHTML = '';
            addMessageToUI('system', 'Started new chat session.');
            chatId = data.chatId;
            tokenCount = data.tokenCount;
            updateTokenDisplay(tokenCount);
        }
    } catch (error) {
        addMessageToUI('system', `Error starting new chat: ${error.message}`);
    }
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

newChatBtn.addEventListener('click', startNewChat);
saveConfigBtn.addEventListener('click', saveConfig);

// MCP Event Listeners
mcpConnectBtn.addEventListener('click', connectMCP);
mcpDisconnectBtn.addEventListener('click', disconnectMCP);
refreshToolsBtn.addEventListener('click', refreshTools);

// MCP config change listeners
mcpUrlInput.addEventListener('change', (e) => {
    mcpConfig.url = e.target.value.trim();
});

mcpApiKeyInput.addEventListener('change', (e) => {
    mcpConfig.apiKey = e.target.value;
});

mcpEnabledCheckbox.addEventListener('change', (e) => {
    mcpConfig.enabled = e.target.checked;
});

// Initialize on load
init();
// Chat configuration
let config = {
	systemPrompt: '',
	bufferSize: 10,
	temperature: 0.7
};

let chatId = 'default';
let tokenCount = 0;

// DOM Elements
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const systemPromptInput = document.getElementById('system-prompt');
const bufferSizeInput = document.getElementById('buffer-size');
const temperatureInput = document.getElementById('temperature');
const saveConfigBtn = document.getElementById('save-config-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const tokenCountDisplay = document.getElementById('token-count');
const tokenBar = document.getElementById('token-bar');

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

        await startNewChat()
        addMessageToUI('system', 'System initialized with configured system prompt.');
	} catch (error) {
		console.error('Failed to load config:', error);
	}
}

// Save configuration
async function saveConfig() {
	const newSystemPrompt = systemPromptInput.value;
	const newBufferSize = parseInt(bufferSizeInput.value);
	const newTemperature = parseFloat(temperatureInput.value);

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

		config.systemPrompt = newSystemPrompt;
		config.bufferSize = newBufferSize;
		config.temperature = newTemperature;

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
		await fetch('/api/log', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				log: data.response
			})
		});

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

	// parse response from llm
	let messageContent = content
	messageContent = messageContent.replaceAll("\n", "<br />")

	console.log(messageContent.match(/```/g) || [])

	console.log(messageContent)
	contentDiv.innerHTML = messageContent;

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

// Initialize on load
init();
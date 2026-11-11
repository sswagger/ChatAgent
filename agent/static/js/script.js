// MCP configuration
let tools = [];
let skills = {};
let connected = 'disconnected';
let isNewSkill = false

// Chat configuration
let tokenCount = 0;
let sessionId = "";
let userMessages = [];
let userMessageIndex = 0;

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
const chatTitle = document.getElementById('chat-title');

// MCP DOM Elements
const mcpConnectBtn = document.getElementById('mcpConnectBtn');
const mcpDisconnectBtn = document.getElementById('mcpDisconnectBtn');
const mcpUrlInput = document.getElementById('mcpUrl');
const mcpId = document.getElementById('mcpId');
const mcpEnabledCheckbox = document.getElementById('mcp-enabled');
const mcpStatusBadge = document.getElementById('status-badge');
const mcpToolsList = document.getElementById('mcpToolsList');
const refreshToolsBtn = document.getElementById('refreshToolsBtn');

// Skill DOM Elements
const skillEditorSection = document.getElementById('skill-editor');
const chatSection = document.getElementById('chat-section');
const newSkillBtn = document.getElementById('add-skill');
const textEditorDiv = document.getElementById('text-editor');
const mdViewerDiv = document.getElementById('skill-preview');
const saveSkillBtn = document.getElementById('save-skill-btn');
const skillsListDiv = document.getElementById('skills-list');
const skillTitleInput = document.getElementById('skill-title');
const skillIdInput = document.getElementById('skill-id');
const skillRuleInput = document.getElementById('make-rule');

// Initialize
async function init() {
	try {
		// start a new chat
        await startNewChat()

		// Load regular config
		const configResponse = await fetch('/api/config', {
			method: 'GET',
			headers: { 'Content-Type': 'application/json', 'session': sessionId },
		});
		const configData = await configResponse.json();
		systemPromptInput.value = configData.systemPrompt;
		bufferSizeInput.value = configData.bufferSize;
		temperatureInput.value = configData.temperature;

		// Load MCP config
		const mcpResponse = await fetch('/api/mcp/config', {
			method: 'GET',
			headers: { 'Content-Type': 'application/json', 'session': sessionId },
		});
		const mcpData = await mcpResponse.json();
		mcpEnabledCheckbox.checked = mcpData.enabled || false;
		mcpUrlInput.value = mcpData.url || '';
		connected = mcpData.connectionStatus || 'disconnected';
		tools = mcpData.tools || [];

		await disconnectMCP()
		updateMcpUI();
		await loadSkills();

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
	const newMcpUrl = mcpUrlInput.value.trim();
	const newMcpEnabled = mcpEnabledCheckbox.checked;

	// validate form
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
		let response = await fetch('/api/config', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				systemPrompt: newSystemPrompt,
				bufferSize: newBufferSize,
				temperature: newTemperature,
				session: sessionId
			})
		});

        // Save MCP config
        await fetch('/api/mcp/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: newMcpUrl,
                enabled: newMcpEnabled,
				session: sessionId
            })
        });

		let data = response.json();

		if (response.ok) {
			alert('Configuration saved! Start a new chat for system prompt changes to take effect.');
		}
		else {
			addMessageToUI("system", "Error:" + data.error);
		}
	} catch (error) {
		alert(`Failed to save configuration: ${error.message}`);
	}
}

// Send message to API
async function sendMessage() {
	const message = messageInput.value.trim();
	if (!message) return;

	userMessages.push(message);
	if (userMessages.length > 5) {
		userMessages.shift();
	}
	userMessageIndex = userMessages.length;

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
				session: sessionId
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
		if (data.calledTools && data.calledTools.length > 0) {
			addMessageToUI('system', data.calledTools.join("\n"));
		}
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
	console.log(content)

	contentDiv.innerHTML = marked.parse(content);

	messageDiv.appendChild(contentDiv);
	chatMessages.appendChild(messageDiv);
	scrollToBottom();
}

// Add loading indicator to UI
function addLoadingToUI(id, sender='assistant') {
	const messageDiv = document.createElement('div');
	messageDiv.className = 'message ' + sender;
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
	hideSkills("AI Chat Assistant (MCP)")
	try {
		const response = await fetch('/api/new-chat', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				session: localStorage.getItem('session')
			})
		});

		const data = await response.json();

		if (response.ok) {
			chatMessages.innerHTML = '';
			addMessageToUI('system', 'Started new chat session.');
			sessionId = data.session;
			tokenCount = data.tokenCount;
			updateTokenDisplay(tokenCount);
			localStorage.setItem('session', data.session)
		}
	} catch (error) {
		addMessageToUI('system', `Error starting new chat: ${error.message}`);
	}
}

//=== MCP ===\\
// Update MCP UI based on connection status
function updateMcpUI() {
    if (connected === 'connected') {
        mcpStatusBadge.textContent = 'Connected';
        mcpStatusBadge.className = 'status-badge status-connected';
        mcpConnectBtn.style.display = 'none';
        mcpDisconnectBtn.style.display = 'inline-block';
    } else if (connected === 'error') {
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

    fetch('/api/mcp/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            url: mcpUrlInput.value.trim(),
            enabled: mcpEnabledCheckbox.checked,
            connectionStatus: connected,
            tools: tools,
			session: sessionId
        })
    });
}

// Update MCP tools list
function updateMcpToolsList(tools) {
    if (!tools || tools.length === 0) {
        mcpToolsList.innerHTML = '<p class="no-tools">No tools loaded. Connect to an MCP server first.</p>';
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
	// add a loading indicator
	const loadingId = 'loading-' + Date.now();
	addLoadingToUI(loadingId, 'system');
    try {
        const url = mcpUrlInput.value.trim();

        const response = await fetch('/api/mcp/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: url,
				session: sessionId
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to connect to MCP server');
        }

		// Remove loading indicator
		const loadingElement = document.getElementById(loadingId);
		if (loadingElement) {
			loadingElement.remove();
		}

        // Update config
        mcpUrlInput.value = url;
        connected = 'connected';
        mcpId.value = data.sessionId
        updateMcpUI();

        // Fetch tools
        await refreshTools()

        addMessageToUI('system', data.message || 'Connected to MCP server');
    } catch (error) {
		// Remove loading indicator
		const loadingElement = document.getElementById(loadingId);
		if (loadingElement) {
			loadingElement.remove();
		}

        connected = 'error';
        updateMcpUI();
        addMessageToUI('system', `Error connecting to MCP: ${error.message}`);
    }
}

// Disconnect from MCP server
async function disconnectMCP() {
    try {
        const response = await fetch('/api/mcp/disconnect', {
            method: 'POST',
			body:  JSON.stringify({
				session: sessionId
            }),
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (response.ok) {
            connected = 'disconnected';
            updateMcpUI();
            addMessageToUI('system', data.message);
            mcpToolsList.innerHTML = "<p class=\"no-tools\">No tools loaded. Connect to an MCP server first.</p>"
            mcpId.value = "Connect for Id"
        }
    } catch (error) {
        addMessageToUI('system', `Error disconnecting: ${error.message}`);
    }
}

// Refresh MCP tools
async function refreshTools() {
    try {
		// add a loading indicator
		const loadingId = 'loading-' + Date.now();
		addLoadingToUI(loadingId, 'system');

        const response = await fetch('/api/mcp/tools', {
			method: 'GET',
			headers: { 'Content-Type': 'application/json', 'session': sessionId }
		});
        const data = await response.json();

        if (response.ok) {
            tools = data.tools || [];
            updateMcpToolsList(tools);
            addMessageToUI('system', `Loaded ${tools.length} tools from MCP`);
        } else {
            addMessageToUI('system', `Error fetching tools: ${data.error || 'Unknown error'}`);
        }

		// Remove loading indicator
		const loadingElement = document.getElementById(loadingId);
		if (loadingElement) {
			loadingElement.remove();
		}
    } catch (error) {
        addMessageToUI('system', `Error fetching tools: ${error.message}`);
    }
}

//=== Skills ===\\
// hide skills section and show chat section
function hideSkills(title) {
	skillEditorSection.style.display = 'none';
	chatSection.style.display = 'flex';
	saveSkillBtn.style.display = 'none';
	newChatBtn.style.display = 'block';
	chatTitle.innerText = title;
	skillTitleInput.disabled = true;
	skillRuleInput.disabled = true;
	skillRuleInput.checked = false;
	skillTitleInput.value = "";
	skillIdInput.value = "";
}

// show skills section and hide chat section
function showSkills(content, title) {
	skillEditorSection.style.display = "flex";
	chatSection.style.display = "none";
	saveSkillBtn.style.display = 'block';
	newChatBtn.style.display = 'none';
	textEditorDiv.innerText = content;
	chatTitle.innerText = title;
	skillTitleInput.disabled = false;
	skillRuleInput.disabled = false;
}

// update the mark-down viewer
function updateMd() {
	mdViewerDiv.innerHTML = marked.parse(textEditorDiv.innerText);
}

// load the skills into the ui
async function loadSkills() {
	try {
		// Load regular config
		const response = await fetch('/api/skill', {
			method: 'GET',
			headers: {'Content-Type': 'application/json', 'session': sessionId},
		});
		const data = await response.json();

		skillsListDiv.innerHTML = "";

		if (data.skills && data.skills.length > 0) {
			data.skills.forEach(skill => {
				const skillDiv = document.createElement("div");
				skillDiv.className = "skill-item";
				skillDiv.innerHTML = `
					<div class="skill-name">${skill.title}</div>
					<div class="skill-delete"><a><img src="/static/img/trash_can.png" alt="Delete" /></a></div>
					<div class="skill-info">Id: ${skill.id}</div>
				`;
				skillsListDiv.appendChild(skillDiv);

				skills[skill.title] = {'Id': skill.id, 'Title': skill.title, 'Contents': skill.content, 'isSelected': false, 'isRule': skill.rule};
			});

			// add click event to documents
			document.querySelectorAll(".skill-item").forEach(skill => {
				skill.addEventListener('click', function() {
					isNewSkill = false;
					let selectedSkill = skills[skill.firstElementChild.innerText];
					// remove border color to all skills
					skillsListDiv.childNodes.forEach(e => {
						e.style.borderColor = "#333";
					});

					if (!selectedSkill["isSelected"]) {
						Object.values(skills).forEach(value => {
							value["isSelected"] = false;
						});

						// update docBadge and add border color
						skill.style.borderColor = "#f63b82";

						selectedSkill["isSelected"] = true;
						skillTitleInput.value = selectedSkill["Title"];
						skillRuleInput.checked = selectedSkill["isRule"];
						showSkills(selectedSkill["Contents"], "Editing Skill: " + selectedSkill["Title"]);
						skillIdInput.value = selectedSkill["Id"];
					}
					else {
						selectedSkill["isSelected"] = false;
						hideSkills("AI Chat Assistant (MCP)");
					}
				});
			});

			document.querySelectorAll(".skill-delete").forEach(skill => {
				skill.addEventListener('click', async function() {
					isNewSkill = false;
					let selectedSkill = skills[skill.parentElement.firstElementChild.innerText];
					// remove border color to all skills
					skillsListDiv.childNodes.forEach(e => {
						e.style.borderColor = "#333";
					});

					let confirmDelete = window.confirm("Delete this skill?");
					if (confirmDelete) {
						const response = await fetch('/api/skill/delete', {
							method: 'POST',
							headers: { 'Content-Type': 'application/json' },
							body: JSON.stringify({
								id: selectedSkill["Id"],
								session: sessionId
							})
						});

						if (response.ok) {
							await loadSkills();
							addMessageToUI("system", "Successfully deleted skill");

							await startNewChat();
						}
					}
				});
			});
		}
	}
	catch (error) {
		addMessageToUI("system", "Failed to refresh documents:" + error);
	}
}

// save skills
async function saveSkill() {
	console.log(skills);
	if (isNewSkill) {
		if (skills[skillTitleInput.value.trim()]) {
			return false;
		}
	}
	isNewSkill = false;

	try {
		const response = await fetch('/api/skill/upload', {
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify({
				title: skillTitleInput.value.trim(),
				content: textEditorDiv.innerText,
				id: skillIdInput.value.trim(),
				rule: skillRuleInput.checked,
				session: sessionId
			})
		});

		const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to upload skill');
        }

		return true;
	}
	catch (error) {
        addMessageToUI('system', `Error connecting to MCP: ${error.message}`);
		return false;
	}
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);

messageInput.addEventListener('keydown', (e) => {
	if (e.key === 'Enter' && !e.shiftKey) {
		e.preventDefault();
		sendMessage();
	}
	else if (e.key === 'ArrowUp' && !e.shiftKey) {
		e.preventDefault();
        if (userMessageIndex-1 >= 0) {
			messageInput.value = userMessages[--userMessageIndex]
		}
		else {
			messageInput.value = "End of history"
			userMessageIndex = -1
		}
	}
	else if (e.key === 'ArrowDown' && !e.shiftKey) {
		e.preventDefault();
		if (userMessages[userMessageIndex+1] !== undefined) {
			messageInput.value = userMessages[++userMessageIndex];
		}
		else {
			messageInput.value = ""
			userMessageIndex = userMessages.length
		}
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

// new skill
newSkillBtn.addEventListener('click', function () {
	// set headers
	showSkills("# Brand New Skill!!", "New Skill: New Skill");
	// set inputs
	skillTitleInput.value = "New Skill";
	skillIdInput.value = "";
	skillRuleInput.checked = false;
	isNewSkill = true;

	// deselect all skills
	skillsListDiv.childNodes.forEach(e => {
		e.style.borderColor = "#333";
	});
	Object.values(skills).forEach(value => {
		value["isSelected"] = false;
	});
});
// save button
saveSkillBtn.addEventListener('click', async function () {
	// save and get skills
	let successSave = await saveSkill();
	await loadSkills();

	if (successSave) {
		// show chat
		hideSkills("AI Chat Assistant (MCP)");

		// deselect all skills
		skillsListDiv.childNodes.forEach(e => {
			e.style.borderColor = "#333";
		});
		Object.values(skills).forEach(value => {
			value["isSelected"] = false;
		});
	}
	else {
		alert("Title must be unique");
	}
});

const observer = new MutationObserver(() => {
    updateMd();
});
observer.observe(textEditorDiv, {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
    characterDataOldValue: true
});

// Initialize mark down display on load
updateMd();
// Initialize on load
init();

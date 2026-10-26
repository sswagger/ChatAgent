//=== Configs ===\\
// Chat configuration
let chatId = "default";
let tokenCount = 0;
let userMessages = []
let userMessageIndex = 0;
let ragDocuments = {}

//=== DOM Elements ===\\
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const chatMessages = document.getElementById("chat-messages");
const systemPromptInput = document.getElementById("system-prompt");
const bufferSizeInput = document.getElementById("buffer-size");
const temperatureInput = document.getElementById("temperature");
const saveConfigBtn = document.getElementById("save-config-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const tokenCountDisplay = document.getElementById("token-count");
const tokenBar = document.getElementById("token-bar");

//=== RAG DOM Elements ===\\
const ragEnabledCheckbox = document.getElementById("rag-enabled");
const ragTopK = document.getElementById("top-k");
const ragSimilarityThreshold = document.getElementById("sim-threshold");
const ragChunkSize = document.getElementById("chunk-size");
const ragChunkOverlap = document.getElementById("chunk-overlap");
const ragUploadInput = document.getElementById("rag-upload");
const ragUploadBtn = document.getElementById("rag-upload-btn");
const ragClearBtn = document.getElementById("rag-clear-btn");
const ragDocsList = document.getElementById("rag-documents-list");
const ragStatusBadge = document.getElementById("status-badge");
const ragDocBadge = document.getElementById("doc-badge");
const ragDocId = document.getElementById("doc-id")

// Initialize
async function init() {
	try {
		// Load regular config
		const configResponse = await fetch("/api/config");
		const configData = await configResponse.json();
		systemPromptInput.value = configData.systemPrompt;
		bufferSizeInput.value = configData.bufferSize;
		temperatureInput.value = configData.temperature;

		await startNewChat()
		addMessageToUI("system", "System initialized with configured system prompt.");

		// Load RAG config
		await loadRagConfig();
	} catch (error) {
		console.error("Failed to load config:", error);
	}
}

// Save configuration
async function saveConfig() {
	const newSystemPrompt = systemPromptInput.value;
	const newBufferSize = parseInt(bufferSizeInput.value);
	const newTemperature = parseFloat(temperatureInput.value);

	// validate form
	if (newBufferSize < 1 || newBufferSize > 50) {
		alert("Buffer size must be between 1 and 50");
		return;
	}

	if (newTemperature < 0 || newTemperature > 2) {
		alert("Temperature must be between 0 and 2");
		return;
	}

	try {
		// Save regular config
		await fetch("/api/config", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				systemPrompt: newSystemPrompt,
				bufferSize: newBufferSize,
				temperature: newTemperature
			})
		});

		// Save RAG config
		if (await saveRagConfig()) {
			alert("Configuration saved! Start a new chat for system prompt changes to take effect.");
		}
	} catch (error) {
		alert("Failed to save configuration: ${error.message}");
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
	addMessageToUI("user", message);
	messageInput.value = "";
	messageInput.style.height = "auto";

	// Show loading indicator
	const loadingId = "loading-" + Date.now();
	addLoadingToUI(loadingId);

	try {
		const response = await fetch("/api/chat", {
			method: "POST",
			headers: {"Content-Type": "application/json"},
			body: JSON.stringify({
				message: message,
				chatId: chatId,
				useRag: ragEnabledCheckbox && ragEnabledCheckbox.checked
			})
		});

		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || "Failed to get response")
		}

		// Remove loading indicator
		const loadingElement = document.getElementById(loadingId);
		if (loadingElement) {
			loadingElement.remove();
		}

		// Add assistant response to UI
		addMessageToUI("assistant", data.response);

		// Update token count
		tokenCount = data.tokenCount;
		updateTokenDisplay(tokenCount);

		// Show sources if available
		if (data.sources && data.sources.length > 0) {
			const sourcesDiv = document.createElement("div");
			sourcesDiv.innerHTML = "<strong>Sources:</strong> " + data.sources.map((s, i) => `${i + 1}. ${s.metadata.source}`).join(", ");
			chatMessages.appendChild(sourcesDiv);
			scrollToBottom();
		}

	} catch (error) {
		const loadingElement = document.getElementById(loadingId);
		if (loadingElement) {
			loadingElement.remove();
		}
		addMessageToUI("system", "Error: ${error.message}");
	}
}

// Add message to UI
function addMessageToUI(role, content) {
	const messageDiv = document.createElement("div");
	messageDiv.className = `message ${role}`;
	messageDiv.id = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

	const contentDiv = document.createElement("div");
	contentDiv.className = "message-content";

	// parse response from llm
	console.log(content)

	contentDiv.innerHTML = marked.parse(content);

	messageDiv.appendChild(contentDiv);
	chatMessages.appendChild(messageDiv);
	scrollToBottom();
}

// Add loading indicator to UI
function addLoadingToUI(id) {
	const messageDiv = document.createElement("div");
	messageDiv.className = "message assistant";
	messageDiv.id = id;

	const contentDiv = document.createElement("div");
	contentDiv.className = "message-content";
	contentDiv.innerHTML = "<span class=\"loading\"></span>";

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
		const response = await fetch("/api/new-chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ chatId: chatId })
		});

		const data = await response.json();

		if (response.ok) {
			chatMessages.innerHTML = "";
			addMessageToUI("system", "Started new chat session.");
			chatId = data.chatId;
			tokenCount = data.tokenCount;
			updateTokenDisplay(tokenCount);
		}
	} catch (error) {
		addMessageToUI("system", "Error starting new chat: ${error.message}");
	}
}

// start a chat with the system
async function systemSession(question) {
	// ask question and remove normal send
	addMessageToUI("assistant", question)
	messageInput.removeEventListener("keydown", send);

	return new Promise((resolve) => {
		function answer(e) {
			if (e.key === "Enter" && !e.shiftKey) {
				// get answer
				e.preventDefault();
				let answer = messageInput.value.trim();

				messageInput.value = '';
				addMessageToUI("user", answer)
				// update event listeners
				messageInput.removeEventListener("keydown", sessionSend);
				messageInput.addEventListener("keydown", send);

				// exit
				resolve(answer);
			}
		}

		// setup new send
		const sessionSend = event => answer(event);
		messageInput.addEventListener("keydown", sessionSend)
	});
}

// named send function
function checkSend(e) {
	// if user clicks enter, then send message
	if (e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		sendMessage();
	}
	// scroll through past prompts
	else if (e.key === "ArrowUp" && !e.shiftKey) {
		e.preventDefault();
        if (userMessageIndex-1 >= 0) {
			messageInput.value = userMessages[--userMessageIndex]
		}
		else {
			messageInput.value = "End of history"
			userMessageIndex = -1
		}
	}
	// scroll through past prompts
	else if (e.key === "ArrowDown" && !e.shiftKey) {
		e.preventDefault();
		if (userMessages[userMessageIndex+1] !== undefined) {
			messageInput.value = userMessages[++userMessageIndex];
		}
		else {
			messageInput.value = ""
			userMessageIndex = userMessages.length
		}
	}
}

//=== RAG Functions ===\\
// update status bar
function updateRagUI(message) {
	ragStatusBadge.textContent = message;
	if (message === "Ready") {
		ragStatusBadge.className = "status-ready badge";
	}
	else {
		ragStatusBadge.className = "status-not-ready badge";
	}
}

// loads config from server
async function loadRagConfig() {
	try {
		const response = await fetch("/api/rag/config");
		const data = await response.json();

		// update settings
		ragEnabledCheckbox.checked = data.enabled;
		ragTopK.value = data.topK;
		ragSimilarityThreshold.value = data.similarityThreshold;
		ragChunkSize.value = data.chunkSize;
		ragChunkOverlap.value = data.chunkOverlap;

		// update ui status
		if (data.ready) {
			updateRagUI("Ready");
		}
		await refreshRagDocuments();
	} catch (error) {
		updateRagUI("Error")
		addMessageToUI("system", "Failed to load RAG config:" + error);
	}
}

// save current rag configuration
async function saveRagConfig() {
	if (ragTopK.value > 20 || ragTopK.value < 1) {
		updateRagUI("Error");
		addMessageToUI("system", "Top K not valid");
		return false;
	}
	if (ragSimilarityThreshold.value > 1 || ragSimilarityThreshold.value < 0) {
		updateRagUI("Error");
		addMessageToUI("system", "Similarity Threshold not valid");
		return false;
	}

	try {
		await fetch("/api/rag/config", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				enabled: ragEnabledCheckbox.checked,
				topK: ragTopK.value,
				similarityThreshold: ragSimilarityThreshold.value
			})
		});
		return true;
	}
	catch (error) {
		updateRagUI("Error");
		addMessageToUI("system", "Failed to save RAG configuration:" + error.message);
		return false;
	}
}

// save the current Chunk config
async function saveDocConfig() {
	if (ragChunkSize.value > 1000 || ragChunkSize.value < 100) {
		addMessageToUI("system", "Chunk Size not valid");
		return false;
	}
	if (ragChunkOverlap.value > 100 || ragChunkOverlap.value < 0) {
		addMessageToUI("system", "Chunk Overlap not valid");
		return false;
	}
	if (ragChunkOverlap.value >= ragChunkSize.value) {
		addMessageToUI("system", "Chunk Overlap must be smaller than Chunk Size");
		return false;
	}

	try {
		await fetch("/api/rag/config", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				chunkSize: ragChunkSize.value,
				chunkOverlap: ragChunkOverlap.value
			})
		});
		return true;
	}
	catch (error) {
		addMessageToUI("system", "Failed to save RAG configuration:" + error.message);
		return false;
	}
}

// request documents from server
async function refreshRagDocuments() {
	try {
		const response = await fetch("/api/rag/documents");
		const data = await response.json();

		ragDocsList.innerHTML = "";

		if (data.documents && data.documents.length > 0) {
			data.documents.forEach(doc => {
				const docDiv = document.createElement("div");
				docDiv.className = "document-item";
				docDiv.innerHTML = `
					<div class="doc-name">${doc.name}</div>
					<div class="doc-info">Chunks: ${doc.chunkCount} | Uploaded: ${new Date(doc.uploadDate).toLocaleDateString()}</div>
				`;
				ragDocsList.appendChild(docDiv);

				ragDocuments[doc.name] = {"name": doc.name.replace(/\..+/g, ""), "id": doc.id, "chunkSize": doc.chunkSize, "chunkOverlap": doc.chunkOverlap, "isSelected": false};
			});

			// add click event to documents
			document.querySelectorAll(".document-item").forEach(doc => {
				doc.addEventListener('click', function() {
					selectedDoc = ragDocuments[doc.firstElementChild.innerText];
					// remove border color to all documents
					ragDocsList.childNodes.forEach(e => {
						e.style.borderColor = "#333";
					});

					if (!selectedDoc["isSelected"]) {
						Object.values(ragDocuments).forEach(value => {
							value["isSelected"] = false;
						});

						// update docBadge and add border color
						ragDocBadge.innerText = selectedDoc["name"];
						ragDocBadge.className = "status-ready badge"
						doc.style.borderColor = "#82f43a";

						selectedDoc["isSelected"] = true
						ragDocId.value = selectedDoc["id"]
						ragChunkSize.value = selectedDoc["chunkSize"];
						ragChunkOverlap.value = selectedDoc["chunkOverlap"];
						ragUploadInput.disabled = true;
						ragChunkSize.disabled = true;
						ragChunkOverlap.disabled = true;
						ragUploadBtn.disabled = true;
						ragUploadBtn.style.backgroundColor = "#2a2a2a";
						ragUploadBtn.style.border = "1px solid #444";
					}
					else {
						ragDocBadge.className = "status-not-ready badge"
						selectedDoc["isSelected"] = false
						ragDocBadge.innerText = "None";
						ragDocId.value = ""
						ragUploadInput.disabled = false;
						ragChunkSize.disabled = false;
						ragChunkOverlap.disabled = false;
						ragUploadBtn.disabled = false;
						ragUploadBtn.style.backgroundColor = "#82f43a";
						ragUploadBtn.style.border = "";
					}
				});
			});
			addMessageToUI("system", "Loaded " + data.documents.length + " documents.")
		} else {
			ragDocsList.innerHTML = "<p class=\"no-documents\">No documents uploaded yet.</p>";
			addMessageToUI("system", "No documents loaded.")
		}

		ragDocBadge.className = "status-not-ready badge"
		ragDocBadge.innerText = "None";
		ragDocId.value = ""
		ragUploadInput.disabled = false;
		ragChunkSize.disabled = false;
		ragChunkOverlap.disabled = false;
		ragUploadBtn.disabled = false;
		ragUploadBtn.style.backgroundColor = "#82f43a";
		ragUploadBtn.style.border = "";
	} catch (error) {
		addMessageToUI("system", "Failed to refresh documents:" + error);
	}
}

// upload a document
async function uploadDocument() {
	const fileInput = ragUploadInput;
	const file = fileInput.files[0];

	if (!file) {
		updateRagUI("Error");
		addMessageToUI("system", "Select document to upload.");
		return;
	}

	if (!await saveDocConfig()) {
		return;
	}

	const formData = new FormData();
	formData.append("file", file);

	const loadingId = "loading-" + Date.now();
	addLoadingToUI(loadingId);

	try {
		const response = await fetch("/api/rag/upload", {
			method: "POST",
			body: formData
		});

		const data = await response.json();

		if (data.success) {
			const loadingElement = document.getElementById(loadingId);
			if (loadingElement) {
				loadingElement.remove();
			}
			addMessageToUI("system", "Document uploaded successfully.")
			fileInput.value = ""; // Clear file input
			await refreshRagDocuments();
			updateRagUI("Ready")
		} else {
			const loadingElement = document.getElementById(loadingId);
			if (loadingElement) {
				loadingElement.remove();
			}
			addMessageToUI("system", "Failed to upload document: " + (data.error || "Unknown error"))
			updateRagUI("Error")
		}
	} catch (error) {
		updateRagUI("Error")
		addMessageToUI("system", "Error uploading document: " + error.message)
	}
}

// clear all current documents
async function clearAllDocuments() {
	setTimeout(() => {
		messageInput.focus();
	}, 10);
	const answer = await systemSession("Are you sure you want to clear all documents? This cannot be undone. (y/n)");

	if (answer.includes("y")) {
		try {
			const response = await fetch("/api/rag/clear", {
				method: "POST"
			});

			const data = await response.json();

			if (data.success) {
				addMessageToUI("system", "All documents cleared successfully.")
				await refreshRagDocuments();
			} else {
				addMessageToUI("system", "Failed to clear documents: " + data.error);
			}
		} catch (error) {
			addMessageToUI("system", "Error clearing documents: " + error.message);
		}
	}
	else {
		addMessageToUI("system", "Operation canceled.")
	}
}

//=== Event Listeners ===\\
sendBtn.addEventListener("click", sendMessage);

const send = event => checkSend(event);
messageInput.addEventListener("keydown", send);

messageInput.addEventListener("input", function() {
	this.style.height = "auto";
	this.style.height = Math.min(this.scrollHeight, 150) + "px";
});

newChatBtn.addEventListener("click", startNewChat);
saveConfigBtn.addEventListener("click", saveConfig);

// RAG Event Listeners
ragUploadBtn.addEventListener("click", uploadDocument);
ragClearBtn.addEventListener("click", clearAllDocuments);

//=== Initialize on load ===\\
init();

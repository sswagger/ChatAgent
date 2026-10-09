// Chat configuration
let config = {
\tsystemPrompt: '',
\tsystemPrompt: "",
\tbufferSize: 10,
\ttemperature: 0.7
};

// RAG configuration
let ragConfig = {
\tenabled: true,
\turl: "",
\tapiKey: "",
\tcollectionName: "documents",
\ttopK: 4,
\tsimilarityThreshold: 0.25,
\tready: false
};

let chatId = "default";
let tokenCount = 0;
let userMessages = []
let userMessageIndex = 0;

// DOM Elements
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

// RAG DOM Elements
const ragEnabledCheckbox = document.getElementById("rag-enabled");
const ragUploadInput = document.getElementById("rag-upload");
const ragUploadBtn = document.getElementById("rag-upload-btn");
const ragRefreshBtn = document.getElementById("rag-refresh-btn");
const ragClearBtn = document.getElementById("rag-clear-btn");
const ragDocsList = document.getElementById("rag-documents-list");
const ragDocCount = document.getElementById("rag-doc-count");
const ragChunkCount = document.getElementById("rag-chunk-count");
const ragStatus = document.getElementById("rag-status");
const ragStatusBadge = document.getElementById("rag-status-badge");
const ragModeToggle = document.getElementById("rag-mode-toggle");

// Initialize
async function init() {
\ttry {
\t\t// Load regular config
\t\tconst configResponse = await fetch("/api/config");
\t\tconst configData = await configResponse.json();
\t\tconfig.systemPrompt = configData.systemPrompt;
\t\tconfig.bufferSize = configData.bufferSize;
\t\tconfig.temperature = configData.temperature || 0.7;
\t\tsystemPromptInput.value = config.systemPrompt;
\t\tbufferSizeInput.value = config.bufferSize;
\t\ttemperatureInput.value = config.temperature;

\t\t// Load RAG config
\t\tawait loadRagConfig();

\t\tawait startNewChat()
\t\taddMessageToUI("system", "System initialized with configured system prompt.");
\t} catch (error) {
\t\tconsole.error("Failed to load config:", error);
\t}
}

async function loadRagConfig() {
\ttry {
\t\tconst response = await fetch("/api/rag/config");
\t\tconst data = await response.json();
\t\tragConfig.enabled = data.enabled;
\t\tragConfig.url = data.url;
\t\tragConfig.collectionName = data.collectionName;
\t\tragConfig.topK = data.topK;
\t\tragConfig.similarityThreshold = data.similarityThreshold;
\t\tragConfig.ready = data.ready;

\t\tragEnabledCheckbox.checked = ragConfig.enabled;
\t\tupdateRagUI();
\t\tawait refreshRagDocuments();
\t\tawait refreshRagStats();
\t} catch (error) {
\t\tconsole.error("Failed to load RAG config:", error);
\t}
}

async function saveRagConfig() {
\ttry {
\t\tawait fetch("/api/rag/config", {
\t\t\tmethod: "POST",
\t\t\theaders: { "Content-Type": "application/json" },
\t\t\tbody: JSON.stringify({
\t\t\t\tenabled: ragEnabledCheckbox.checked,
\t\t\t\ttopK: ragConfig.topK,
\t\t\t\tsimilarityThreshold: ragConfig.similarityThreshold
\t\t\t})
\t\t});
\t\talert("RAG configuration saved!");
\t} catch (error) {
\t\talert(`Failed to save RAG configuration: ${error.message}`);
\t}
}

// Save configuration
async function saveConfig() {
\tconst newSystemPrompt = systemPromptInput.value;
\tconst newBufferSize = parseInt(bufferSizeInput.value);
\tconst newTemperature = parseFloat(temperatureInput.value);

\tif (newBufferSize < 1 || newBufferSize > 50) {
\t\talert("Buffer size must be between 1 and 50");
\t\treturn;
\t}

\tif (newTemperature < 0 || newTemperature > 2) {
\t\talert("Temperature must be between 0 and 2");
\t\treturn;
\t}

\ttry {
\t\t// Save regular config
\t\tawait fetch("/api/config", {
\t\t\tmethod: "POST",
\t\t\theaders: { "Content-Type": "application/json" },
\t\t\tbody: JSON.stringify({
\t\t\t\tsystemPrompt: newSystemPrompt,
\t\t\t\tbufferSize: newBufferSize,
\t\t\t\ttemperature: newTemperature
\t\t\t})
\t\t});

\t\t// Save RAG config
\t\tawait saveRagConfig();

\t\tconfig.systemPrompt = newSystemPrompt;
\t\tconfig.bufferSize = newBufferSize;
\t\tconfig.temperature = newTemperature;

\t\talert("Configuration saved! Start a new chat for system prompt changes to take effect.");
\t} catch (error) {
\t\talert(`Failed to save configuration: ${error.message}`);
\t}
}

// Send message to API
async function sendMessage() {
\tconst message = messageInput.value.trim();
\tif (!message) return;

\tuserMessages.push(message);
\tif (userMessages.length > 5) {
\t\tuserMessages.shift();
\t}
\tuserMessageIndex = userMessages.length;

\t// Add user message to UI
\taddMessageToUI("user", message);
\tmessageInput.value = "";
\tmessageInput.style.height = "auto";

\t// Show loading indicator
\tconst loadingId = "loading-" + Date.now();
\taddLoadingToUI(loadingId);

\ttry {
\t\tconst response = await fetch("/api/chat", {
\t\t\tmethod: "POST",
\t\t\theaders: { "Content-Type": "application/json" },
\t\t\tbody: JSON.stringify({
\t\t\t\tmessage: message,
\t\t\t\tchatId: chatId,
\t\t\t\tuseRag: ragModeToggle && ragModeToggle.checked
\t\t\t})
\t\t});

\t\tconst data = await response.json();

\t\tif (!response.ok) {
\t\t\tthrow new Error(data.error || "Failed to get response")
\t\t}

\t\t// Remove loading indicator
\t\tconst loadingElement = document.getElementById(loadingId);
\t\tif (loadingElement) {
\t\t\tloadingElement.remove();
\t\t}

\t\taddMessageToUI("assistant", data.response);

\t\t// Update token count
\t\ttokenCount = data.tokenCount;
\t\tupdateTokenDisplay(tokenCount);

\t\t// Show sources if available
\t\tif (data.sources && data.sources.length > 0) {
\t\t\tconst sourcesDiv = document.createElement("div");
\t\t\tsourcesDiv.className = "rag-sources";
\t\t\tsourcesDiv.innerHTML = "<strong>Sources:</strong> " + data.sources.map((s, i) => `${i+1}. ${s.metadata.source}`).join(", ");
\t\t\tchatMessages.appendChild(sourcesDiv);
\t\t\tscrollToBottom();
\t\t}

\t} catch (error) {
\t\tconst loadingElement = document.getElementById(loadingId);
\t\tif (loadingElement) {
\t\t\tloadingElement.remove();
\t\t}
\t\taddMessageToUI("system", `Error: ${error.message}`);
\t}
}

// Add message to UI
function addMessageToUI(role, content) {
\tconst messageDiv = document.createElement("div");
\tmessageDiv.className = `message ${role}`;
\tmessageDiv.id = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

\tconst contentDiv = document.createElement("div");
\tcontentDiv.className = "message-content";

\t// parse response from llm
\tconsole.log(content)

\tcontentDiv.innerHTML = marked.parse(content);

\tmessageDiv.appendChild(contentDiv);
\tchatMessages.appendChild(messageDiv);
\tscrollToBottom();
}

// Add loading indicator to UI
function addLoadingToUI(id) {
\tconst messageDiv = document.createElement("div");
\tmessageDiv.className = "message assistant";
\tmessageDiv.id = id;

\tconst contentDiv = document.createElement("div");
\tcontentDiv.className = "message-content";
\tcontentDiv.innerHTML = "<span class=\"loading\"></span>";

\tmessageDiv.appendChild(contentDiv);
\tchatMessages.appendChild(messageDiv);
\tscrollToBottom();
}

// Scroll to bottom of chat
function scrollToBottom() {
\tchatMessages.scrollTop = chatMessages.scrollHeight;
}

// Update token display
function updateTokenDisplay(count) {
\ttokenCountDisplay.textContent = count;
\t// Scale bar: assume max 4096 tokens
\tconst percentage = Math.min((count / 4096) * 100, 100);
\ttokenBar.style.width = `${percentage}%`;
}

// Start new chat
async function startNewChat() {
\ttry {
\t\tconst response = await fetch("/api/new-chat", {
\t\t\tmethod: "POST",
\t\t\theaders: { "Content-Type": "application/json" },
\t\t\tbody: JSON.stringify({ chatId: chatId })
\t\t});

\t\tconst data = await response.json();

\t\tif (response.ok) {
\t\t\tchatMessages.innerHTML = "";
\t\t\taddMessageToUI("system", "Started new chat session.");
\t\t\tchatId = data.chatId;
\t\t\ttokenCount = data.tokenCount;
\t\t\tupdateTokenDisplay(tokenCount);
\t\t}
\t} catch (error) {
\t\taddMessageToUI("system", `Error starting new chat: ${error.message}`);
\t}
}

// RAG Functions
function updateRagUI() {
\tif (ragConfig.ready) {
\t\tragStatusBadge.textContent = "Ready";
\t\tragStatusBadge.style.backgroundColor = "#4CAF50";
\t\tragStatus.textContent = "Ready";
\t} else {
\t\tragStatusBadge.textContent = "Not Ready";
\t\tragStatusBadge.style.backgroundColor = "#f44336";
\t\tragStatus.textContent = "Initializing...";
\t}
}

async function refreshRagDocuments() {
\ttry {
\t\tconst response = await fetch("/api/rag/documents");
\t\tconst data = await response.json();

\t\tragDocsList.innerHTML = "";

\t\tif (data.documents && data.documents.length > 0) {
\t\t\tdata.documents.forEach(doc => {
\t\t\t\tconst docDiv = document.createElement("div");
\t\t\t\tdocDiv.className = "document-item";
\t\t\t\tdocDiv.innerHTML = `
\t\t\t\t\t<span class="doc-name">${doc.name}</span>
\t\t\t\t\t<span class="doc-info">Chunks: ${doc.chunkCount} | Uploaded: ${new Date(doc.uploadDate).toLocaleDateString()}</span>
\t\t\t\t\t<button class="btn-danger btn-small" onclick="deleteDocument('${doc.id}')">Delete</button>
\t\t\t\t`;
\t\t\t\tragDocsList.appendChild(docDiv);
\t\t\t});
\t\t} else {
\t\t\tragDocsList.innerHTML = "<p class=\"no-documents\">No documents uploaded yet.</p>";
\t\t}
\t} catch (error) {
\t\tconsole.error("Failed to refresh documents:", error);
\t}
}

async function refreshRagStats() {
\ttry {
\t\tconst response = await fetch("/api/rag/stats");
\t\tconst data = await response.json();

\t\tragDocCount.textContent = data.documents || 0;
\t\tragChunkCount.textContent = data.totalChunks || 0;
\t\tupdateRagUI();
\t} catch (error) {
\t\tconsole.error("Failed to refresh stats:", error);
\t}
}

function deleteDocument(docId) {
\tif (!confirm("Are you sure you want to delete this document?")) return;

\tfetch("/api/rag/document/" + docId, {
\t\tmethod: "DELETE"
\t})
\t.then(response => response.json())
\t.then(data => {
\t\tif (data.success) {
\t\t\talert("Document deleted successfully");
\t\t\trefreshRagDocuments();
\t\t\trefreshRagStats();
\t\t} else {
\t\t\talert("Failed to delete document: " + data.error);
\t\t}
\t})
\t.catch(error => {
\t\talert("Error deleting document: " + error.message);
\t});
}

async function uploadDocument() {
\tconst fileInput = document.getElementById("rag-upload");
\tconst file = fileInput.files[0];

\tif (!file) {
\t\talert("Please select a file to upload");
\t\treturn;
\t}

\tconst formData = new FormData();
\tformData.append("file", file);

\ttry {
\t\tconst response = await fetch("/api/rag/upload", {
\t\t\tmethod: "POST",
\t\t\tbody: formData
\t\t});

\t\tconst data = await response.json();

\t\tif (data.success) {
\t\t\talert("Document uploaded successfully!");
\t\t\tfileInput.value = ""; // Clear file input
\t\t\trefreshRagDocuments();
\t\t\trefreshRagStats();
\t\t} else {
\t\t\talert("Failed to upload document: " + (data.error || "Unknown error"));
\t\t}
\t} catch (error) {
\t\talert("Error uploading document: " + error.message);
\t}
}

async function clearAllDocuments() {
\tif (!confirm("Are you sure you want to clear all documents? This cannot be undone.")) return;

\ttry {
\t\tconst response = await fetch("/api/rag/clear", {
\t\t\tmethod: "POST"
\t\t});

\t\tconst data = await response.json();

\t\tif (data.success) {
\t\t\talert("All documents cleared successfully");
\t\t\trefreshRagDocuments();
\t\t\trefreshRagStats();
\t\t} else {
\t\t\talert("Failed to clear documents: " + data.error);
\t\t}
\t} catch (error) {
\t\talert("Error clearing documents: " + error.message);
\t}
}

// Event Listeners
sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (e) => {
\tif (e.key === "Enter" && !e.shiftKey) {
\t\te.preventDefault();
\t\tsendMessage();
\t}
\telse if (e.key === "ArrowUp" && !e.shiftKey) {
\t\te.preventDefault();
        if (userMessageIndex-1 >= 0) {
\t\t\tmessageInput.value = userMessages[--userMessageIndex]
\t\t}
\t\telse {
\t\t\tmessageInput.value = "End of history"
\t\t\tuserMessageIndex = -1
\t\t}
\t}
\telse if (e.key === "ArrowDown" && !e.shiftKey) {
\t\te.preventDefault();
\t\tif (userMessages[userMessageIndex+1] !== undefined) {
\t\t\tmessageInput.value = userMessages[++userMessageIndex];
\t\t}
\t\telse {
\t\t\tmessageInput.value = ""
\t\t\tuserMessageIndex = userMessages.length
\t\t}
\t}
});

messageInput.addEventListener("input", function() {
\tthis.style.height = "auto";
\tthis.style.height = Math.min(this.scrollHeight, 150) + "px";
});

newChatBtn.addEventListener("click", startNewChat);
saveConfigBtn.addEventListener("click", saveConfig);

// RAG Event Listeners
ragEnabledCheckbox.addEventListener("change", async (e) => {
\tragConfig.enabled = e.target.checked;
\tawait saveRagConfig();
});

ragUploadBtn.addEventListener("click", uploadDocument);
ragRefreshBtn.addEventListener("click", () => {
\trefreshRagDocuments();
\trefreshRagStats();
});
ragClearBtn.addEventListener("click", clearAllDocuments);

// Initialize on load
init();

"""
AI Chat Agent - Single File Implementation
Combines Flask API, Configuration Loader, and RAG Service

This file contains:
- Flask application with chat and RAG API endpoints
- Configuration loading and saving utilities
- RAG (Retrieval-Augmented Generation) service with document processing
"""

import os
import json
import uuid
import threading
import io
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Flask and web imports
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# RAG imports
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables for LLM configuration
load_dotenv()

# === Configuration Constants ===
CONFIG_FILE = 'config.json'
RAG_CONFIG_FILE = 'rag_config.json'
LOG_FILE = 'log.txt'

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

# RAG defaults
RAG_DEFAULTS = {
    'enabled': True,
    'url': os.getenv('CHROMA_DB_URL', 'http://chromadb:8000'),
    'apiKey': os.getenv('CHROMA_API_KEY', ''),
    'collectionName': os.getenv('CHROMA_COLLECTION', 'documents'),
    'topK': int(os.getenv('RAG_TOP_K', '4')),
    'similarityThreshold': float(os.getenv('RAG_THRESHOLD', '0.25')),
    'embeddingModel': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
    'chunkSize': int(os.getenv('CHUNK_SIZE', '500')),
    'chunkOverlap': int(os.getenv('CHUNK_OVERLAP', '50')),
    'documents': []
}

# === Threading Locks ===
config_lock = threading.Lock()
rag_lock = threading.Lock()

# === Global State ===
app = Flask(__name__, static_folder='static', static_url_path='/static')
config = {}
rag_service = None
chat_history = {}


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def load_config(config_file: str, default_config: dict) -> dict:
    """
    Load configuration from a JSON file with fallback to defaults.
    
    Args:
        config_file: Path to the configuration file
        default_config: Dictionary of default configuration values
        
    Returns:
        Dictionary containing loaded configuration merged with defaults
    """
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
    """
    Save configuration to a JSON file with thread-safe locking.
    
    Args:
        configuration: Dictionary of configuration values to save
        config_file: Path to the configuration file
    """
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
# RAG SERVICE
# =============================================================================

class RAGService:
    """
    Handles document upload, storage, and retrieval for Retrieval-Augmented Generation.
    
    This class manages:
    - Document parsing (PDF, DOCX, TXT, MD)
    - Vector storage using ChromaDB
    - Semantic search and retrieval
    - Configuration management
    """
    
    def __init__(self):
        """
        Initialize the RAG service by loading configuration and setting up
        the vector database components.
        """
        self.config = self._load_config()
        self.client = None
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self._initialize()
    
    def _load_config(self) -> dict:
        """
        Load RAG configuration from rag_config.json file.
        
        Returns:
            Dictionary containing RAG configuration with defaults
        """
        if os.path.exists(RAG_CONFIG_FILE):
            try:
                with open(RAG_CONFIG_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._get_defaults()
    
    def _get_defaults(self) -> dict:
        """
        Get default RAG configuration values.
        
        Returns:
            Dictionary with default RAG configuration
        """
        return RAG_DEFAULTS.copy()
    
    def _initialize(self):
        """
        Initialize RAG components including embeddings, text splitter,
        and ChromaDB vector store.
        """
        try:
            # Initialize embeddings (using OpenAI API)
            api_key = os.getenv("LLM_API_KEY", "")
            if api_key:
                self.embeddings = OpenAIEmbeddings(
                    model=self.config.get("embeddingModel", "text-embedding-3-small"),
                    api_key=api_key
                )
            else:
                # Fallback to local embeddings if no API key
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            
            # Initialize text splitter for chunking documents
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get("chunkSize", 500),
                chunk_overlap=self.config.get("chunkOverlap", 50)
            )
            
            # Initialize ChromaDB client
            chroma_path = os.getenv("CHROMA_PATH", "./chroma_data")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(path=chroma_path)
            
            # Get or create collection
            collection_name = self.config.get("collectionName", "documents")
            self.collection = self.client.get_or_create_collection(name=collection_name)
            
            # Initialize LangChain Chroma wrapper
            self.vector_store = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            self.config["ready"] = True
            
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            self.config["ready"] = False
            self.config["error"] = str(e)
    
    def upload_document(self, file_path: str, file_content: str = None, file_name: str = None) -> dict:
        """
        Upload and process a document for RAG retrieval.
        
        Args:
            file_path: Path to the file on disk (if file_content is None)
            file_content: Direct content string to process (if provided, overrides file_path)
            file_name: Name to use for the document (auto-detected if not provided)
            
        Returns:
            Dictionary with upload result containing success status and document info
        """
        try:
            if file_content is None:
                if not os.path.exists(file_path):
                    return {"success": False, "error": f"File not found: {file_path}"}
                
                with open(file_path, "rb") as f:
                    file_content = f.read()
            
            documents = self._parse_document(file_content, file_path, file_name)
            
            if not documents:
                return {"success": False, "error": "No content could be extracted from document"}
            
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                return {"success": False, "error": "No chunks could be created from document"}
            
            doc_id = str(uuid.uuid4())
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            self.vector_store.add_texts(texts=texts, ids=chunk_ids, metadatas=metadatas)
            
            doc_info = {
                "id": doc_id,
                "name": file_name or os.path.basename(file_path),
                "path": file_path,
                "chunkCount": len(chunks),
                "uploadDate": datetime.now().isoformat(),
                "size": len(file_content) if isinstance(file_content, str) else len(str(file_content))
            }
            
            with rag_lock:
                self.config["documents"].append(doc_info)
                self._save_config()
            
            return {
                "success": True,
                "docId": doc_id,
                "name": doc_info["name"],
                "chunks": len(chunks),
                "message": f"Successfully uploaded and chunked document"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _parse_document(self, content, file_path: str, file_name: str = None) -> List[Document]:
        """
        Parse document content based on file extension.
        
        Args:
            content: Document content (bytes or string)
            file_path: Path to the original file
            file_name: Name of the file
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        filename = file_name or os.path.basename(file_path).lower()
        
        try:
            if filename.endswith(".pdf"):
                documents = self._parse_pdf(content, filename)
            elif filename.endswith(".docx"):
                documents = self._parse_docx(content, filename)
            elif filename.endswith(".txt"):
                documents = self._parse_txt(content, filename)
            elif filename.endswith(".md"):
                documents = self._parse_md(content, filename)
            else:
                documents = self._parse_txt(content, filename)
        except Exception as e:
            documents = [Document(
                page_content=f"Error parsing document: {str(e)}",
                metadata={"source": filename, "error": str(e)}
            )]
        
        return documents
    
    def _parse_pdf(self, content, filename: str) -> List[Document]:
        """
        Parse PDF document content.
        
        Args:
            content: PDF file content as bytes or string
            filename: Name of the PDF file
            
        Returns:
            List containing a single Document with extracted text
        """
        try:
            import PyPDF2
            if isinstance(content, str):
                content = content.encode("utf-8", errors="ignore")
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or "" + "\n"
            
            return [Document(
                page_content=text,
                metadata={"source": filename, "type": "pdf"}
            )]
        except Exception as e:
            return [Document(
                page_content=f"PDF parsing error: {str(e)}",
                metadata={"source": filename, "type": "pdf", "error": str(e)}
            )]
    
    def _parse_docx(self, content, filename: str) -> List[Document]:
        """
        Parse DOCX document content.
        
        Args:
            content: DOCX file content as bytes or string
            filename: Name of the DOCX file
            
        Returns:
            List containing a single Document with extracted text
        """
        try:
            from docx import Document as DocxDocument
            if isinstance(content, str):
                content = content.encode("utf-8", errors="ignore")
            
            doc = DocxDocument(io.BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            return [Document(
                page_content=text,
                metadata={"source": filename, "type": "docx"}
            )]
        except Exception as e:
            return [Document(
                page_content=f"DOCX parsing error: {str(e)}",
                metadata={"source": filename, "type": "docx", "error": str(e)}
            )]
    
    def _parse_txt(self, content, filename: str) -> List[Document]:
        """
        Parse plain text file content.
        
        Args:
            content: Text content as bytes or string
            filename: Name of the text file
            
        Returns:
            List containing a single Document with the text content
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        return [Document(
            page_content=content,
            metadata={"source": filename, "type": "txt"}
        )]
    
    def _parse_md(self, content, filename: str) -> List[Document]:
        """
        Parse Markdown file content.
        
        Args:
            content: Markdown content as bytes or string
            filename: Name of the Markdown file
            
        Returns:
            List containing a single Document with the markdown content
        """
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        return [Document(
            page_content=content,
            metadata={"source": filename, "type": "markdown"}
        )]
    
    def retrieve_relevant_docs(self, query: str, top_k: int = None, threshold: float = None) -> List[dict]:
        """
        Retrieve relevant documents for a given query using semantic search.
        
        Args:
            query: Search query string
            top_k: Number of results to retrieve (uses config default if None)
            threshold: Minimum similarity threshold (uses config default if None)
            
        Returns:
            List of relevant documents with scores and metadata
        """
        try:
            if not self.config.get("enabled", True) or not self.config.get("ready", False):
                return []
            
            k = top_k or self.config.get("topK", 4)
            sim_threshold = threshold or self.config.get("similarityThreshold", 0.25)
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            relevant_docs = []
            for doc, score in results:
                similarity = 1 - score
                
                if similarity >= sim_threshold:
                    relevant_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "threshold": sim_threshold
                    })
            
            return relevant_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def query_documents(self, query: str, context_window: int = 3) -> Tuple[str, List[dict]]:
        """
        Query documents and return augmented context for the LLM.
        
        Args:
            query: User's question or query
            context_window: Number of previous context items (currently unused)
            
        Returns:
            Tuple of (augmented prompt string, list of source documents)
        """
        relevant = self.retrieve_relevant_docs(query)
        
        if not relevant:
            return "No relevant documents found.", []
        
        context_parts = []
        docs_info = []
        
        for doc in relevant:
            source = doc["metadata"].get("source", "Unknown")
            chunk_info = f"[Source: {source}]\n{doc['content'][:500]}..."
            context_parts.append(chunk_info)
            docs_info.append(doc)
        
        context = "\n\n---\n\n".join(context_parts)
        augmented_prompt = f"""Based on the following document context:

{context}

User Question: {query}

Please answer the question based on the provided document context. Cite specific sources when relevant.
"""
        
        return augmented_prompt, docs_info
    
    def list_documents(self) -> List[dict]:
        """
        List all uploaded documents.
        
        Returns:
            List of document information dictionaries
        """
        return self.config.get("documents", [])
    
    def delete_document(self, doc_id: str) -> dict:
        """
        Delete a document and its chunks from the vector store.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            Dictionary with deletion result containing success status
        """
        try:
            with rag_lock:
                doc_index = None
                for i, doc in enumerate(self.config["documents"]):
                    if doc["id"] == doc_id:
                        doc_index = i
                        break
                
                if doc_index is None:
                    return {"success": False, "error": f"Document not found: {doc_id}"}
                
                chunk_ids = [f"{doc_id}_{i}" for i in range(self.config["documents"][doc_index]["chunkCount"])]
                
                try:
                    self.collection.delete(ids=chunk_ids)
                except Exception as e:
                    print(f"Warning: Could not delete from collection: {e}")
                
                deleted_doc = self.config["documents"].pop(doc_index)
                self._save_config()
                
                return {
                    "success": True,
                    "message": f"Deleted document: {deleted_doc['name']}",
                    "deletedChunks": deleted_doc["chunkCount"]
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clear_all(self) -> dict:
        """
        Clear all documents from the vector store.
        
        Returns:
            Dictionary with clear result containing success status
        """
        try:
            with rag_lock:
                self.collection.delete(where={})
                self.config["documents"] = []
                self._save_config()
                
                return {
                    "success": True,
                    "message": "All documents cleared"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> dict:
        """
        Get RAG statistics.
        
        Returns:
            Dictionary with RAG statistics including document count and chunk count
        """
        docs = self.config.get("documents", [])
        total_chunks = sum(doc.get("chunkCount", 0) for doc in docs)
        
        return {
            "documents": len(docs),
            "totalChunks": total_chunks,
            "enabled": self.config.get("enabled", True),
            "ready": self.config.get("ready", False),
            "collection": self.config.get("collectionName", "documents")
        }
    
    def _save_config(self):
        """
        Save RAG configuration to rag_config.json file.
        """
        try:
            with open(RAG_CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")


def get_rag_service() -> RAGService:
    """
    Get or create the RAG service singleton instance.
    
    Returns:
        RAGService instance
    """
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


def reload_rag_config():
    """
    Reload RAG configuration and reset the service instance.
    
    Returns:
        New RAGService instance with reloaded configuration
    """
    global rag_service
    rag_service = None
    return get_rag_service()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken.
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Estimated number of tokens
    """
    try:
        if not text or not isinstance(text, str):
            return 0
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(str(text)) // 4 if text else 0


def get_total_tokens(messages: List[dict]) -> int:
    """
    Calculate the total token count for a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and/or 'content' keys
        
    Returns:
        Total estimated token count
    """
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            if 'role' in msg:
                total += estimate_tokens(msg['role'])
            if 'content' in msg:
                total += estimate_tokens(msg['content'])
    return total


def call_llm(messages: List[dict]) -> Tuple[dict, Optional[str]]:
    """
    Call the OpenAI-compatible LLM API.
    
    Args:
        messages: List of message dictionaries for the conversation
        
    Returns:
        Tuple of (response message dict, error string or None)
    """
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
    """
    Log API actions to the log file.
    
    Args:
        path: API endpoint path
        request_payload: Request data
        response_payload: Response data
        is_new_chat: Whether this is a new chat session (overwrites log)
        request_method: HTTP method used
        error: Error status code or "200" for success
    """
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


# =============================================================================
# FLASK API ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """
    Serve the main chat interface HTML page.
    
    Returns:
        Index.html static file
    """
    log_actions("/", "", "static/index.html", is_new_chat=True)
    return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """
    GET: Retrieve current user configuration.
    POST: Update user configuration.
    
    Accepts:
        - systemPrompt: System prompt for the AI
        - bufferSize: Number of messages to keep in context
        - temperature: LLM temperature setting
        
    Returns:
        GET: Current configuration dictionary
        POST: Success response
    """
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


def get_rag_service_instance() -> RAGService:
    """
    Get or create the RAG service instance (wrapper for singleton pattern).
    
    Returns:
        RAGService instance
    """
    global rag_service
    if rag_service is None:
        rag_service = get_rag_service()
    return rag_service


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat messages with optional RAG integration.
    
    Request body:
        - message: User's message
        - chatId: Chat session identifier
        - useRag: Whether to use RAG context
        
    Returns:
        Response with AI reply, token count, and chat ID
    """
    data = request.json
    message = data.get('message', '')
    chat_id = data.get('chatId', 'default')
    use_rag = data.get('useRag', False)

    if not message:
        json_response = {'error': 'Message is required'}
        log_actions("/api/chat", data, str(json_response), error="400")
        return jsonify(json_response), 400

    # Initialize chat history if needed
    if chat_id not in chat_history:
        chat_history[chat_id] = []

    # Add system prompt if this is a new chat
    if len(chat_history[chat_id]) == 0:
        prompt = config.get('systemPrompt', USER_CONFIG_DEFAULTS['systemPrompt'])
        # Add RAG instruction if enabled
        rag_service = get_rag_service_instance()
        if rag_service.config.get('enabled', False):
            prompt += "\n\nYou have access to uploaded documents. When the user asks about a topic, you can retrieve relevant information from the knowledge base and cite your sources. If no relevant context exists in the documents, say so honestly."
        chat_history[chat_id].append({'role': 'system', 'content': prompt})

    # Add user message
    chat_history[chat_id].append({'role': 'user', 'content': message})

    # Apply buffer size limit
    if len(chat_history[chat_id]) > config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']) + 1:
        chat_history[chat_id] = [chat_history[chat_id][0]] + chat_history[chat_id][-config.get('bufferSize', USER_CONFIG_DEFAULTS['bufferSize']):]

    # If RAG is enabled, retrieve relevant documents and augment the prompt
    json_response = {}  # Initialize to fix bug in original code
    if use_rag:
        rag_service = get_rag_service_instance()
        if rag_service.config.get('enabled', False) and rag_service.config.get('ready', False):
            try:
                augmented_prompt, docs = rag_service.query_documents(message)
                chat_history[chat_id][-1]['content'] = augmented_prompt
                json_response['sources'] = docs
            except Exception as e:
                log_actions("/api/chat/rag", data, str({'error': str(e)}), error="500")

    # Include get_capabilities function when MCP is enabled
    full_response, error = call_llm(chat_history[chat_id])

    if error:
        json_response = {'error': f'Failed to get response= {error}'}
        log_actions("/api/chat", data, str(json_response), error="500")
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
    log_actions("/api/chat", data, str(json_response))
    return jsonify(json_response)


@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """
    Start a new chat session.
    
    Request body:
        - chatId: Optional chat session identifier
        
    Returns:
        Success response with chat ID and token count
    """
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
    """
    Get the current token count for a chat session.
    
    Request body:
        - chatId: Chat session identifier
        
    Returns:
        Token count dictionary
    """
    data = request.json
    chat_id = data.get('chatId', 'default')

    if chat_id not in chat_history:
        json_response = jsonify({'tokenCount': 0})
        log_actions("/api/token-count", data, str(json_response))
        return json_response

    json_response = {'tokenCount': get_total_tokens(chat_history[chat_id])}
    log_actions("/api/token-count", data, str(json_response))
    return jsonify(json_response)


@app.route('/api/log', methods=['POST'])
def log_data():
    """
    Append data to the log file.
    
    Request body:
        - log: Log message to append
        
    Returns:
        Success response
    """
    data = request.json

    with open(LOG_FILE, 'a') as log:
        log.write(data.get("log") + "\n\n")

    json_response = {"success": True}
    log_actions('/api/log', data, str(json_response))
    return jsonify(json_response)


# =============================================================================
# RAG API ENDPOINTS
# =============================================================================

@app.route('/api/rag/upload', methods=['POST'])
def rag_upload():
    """
    Upload and process a document for RAG.
    
    Supports:
        - File upload via multipart form data
        - Direct content upload via JSON
        
    Returns:
        Upload result with document ID and chunk count
    """
    rag_service = get_rag_service_instance()
    
    if not rag_service.config.get('enabled', False):
        return jsonify({'success': False, 'error': 'RAG is not enabled'}), 400

    if 'file' in request.files:
        # Handle file upload
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = rag_service.upload_document(tmp_path, file_name=file.filename)
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
        return jsonify(result)
    
    return jsonify({'success': False, 'error': 'No file or content provided'}), 400


@app.route('/api/rag/documents', methods=['GET'])
def rag_list_documents():
    """
    List all uploaded documents.
    
    Returns:
        Dictionary with list of documents
    """
    rag_service = get_rag_service_instance()
    documents = rag_service.list_documents()
    return jsonify({'documents': documents})


@app.route('/api/rag/document/<doc_id>', methods=['DELETE'])
def rag_delete_document(doc_id):
    """
    Delete a specific document.
    
    Args:
        doc_id: ID of the document to delete
        
    Returns:
        Deletion result
    """
    rag_service = get_rag_service_instance()
    result = rag_service.delete_document(doc_id)
    return jsonify(result)


@app.route('/api/rag/clear', methods=['POST'])
def rag_clear():
    """
    Clear all documents from the RAG system.
    
    Returns:
        Clear result
    """
    rag_service = get_rag_service_instance()
    result = rag_service.clear_all()
    return jsonify(result)


@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
    """
    Get RAG statistics.
    
    Returns:
        Statistics including document count and chunk count
    """
    rag_service = get_rag_service_instance()
    stats = rag_service.get_stats()
    return jsonify(stats)


@app.route('/api/rag/config', methods=['GET', 'POST'])
def rag_config_endpoint():
    """
    GET: Retrieve current RAG configuration.
    POST: Update RAG configuration.
    
    Accepts:
        - enabled: Whether RAG is enabled
        - url: ChromaDB URL
        - collectionName: Vector collection name
        - topK: Number of results to retrieve
        - similarityThreshold: Minimum similarity threshold
        
    Returns:
        GET: Current configuration
        POST: Success response
    """
    global rag_service
    
    if request.method == 'GET':
        rag_service = get_rag_service_instance()
        return jsonify({
            'enabled': rag_service.config.get('enabled', True),
            'url': rag_service.config.get('url', ''),
            'collectionName': rag_service.config.get('collectionName', ''),
            'topK': rag_service.config.get('topK', 4),
            'similarityThreshold': rag_service.config.get('similarityThreshold', 0.25),
            'ready': rag_service.config.get('ready', False)
        })
    else:
        data = request.json
        if 'enabled' in data:
            rag_service = get_rag_service_instance()
            rag_service.config['enabled'] = bool(data['enabled'])
        
        if 'url' in data:
            rag_service = get_rag_service_instance()
            rag_service.config['url'] = data['url']
        
        if 'collectionName' in data:
            rag_service = get_rag_service_instance()
            rag_service.config['collectionName'] = data['collectionName']
        
        if 'topK' in data:
            rag_service = get_rag_service_instance()
            rag_service.config['topK'] = int(data['topK'])
        
        if 'similarityThreshold' in data:
            rag_service = get_rag_service_instance()
            rag_service.config['similarityThreshold'] = float(data['similarityThreshold'])
        
        # Reload service with new config
        rag_service = reload_rag_config()
        
        return jsonify({'success': True})


if __name__ == '__main__':
    # Load user config on startup
    load_user_config()
    # Run the Flask application
    app.run(host='0.0.0.0', port=config.get('port', USER_CONFIG_DEFAULTS['port']), debug=True)

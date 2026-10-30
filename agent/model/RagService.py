"""
RAG (Retrieval-Augmented Generation) Service Module
"""

import os
import json
import uuid
import io
from datetime import datetime
from typing import List, Tuple

# RAG imports
import threading
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Threading lock for thread-safe operations
rag_lock = threading.Lock()

# Configuration constants (imported from app.py)
RAG_CONFIG_FILE = 'rag_config.json'
RAG_DEFAULTS = {
	'enabled': True,
	'collectionName': 'documents',
	'topK': 5,
	'similarityThreshold': 0.9,
	'embeddingModel': '',
	'chunkSize': 500,
	'chunkOverlap': 30,
	'chromaPath': './chroma_data',
	'documents': []
}


class RAGService:
	"""
	Handles document upload, storage, and retrieval for Retrieval-Augmented Generation.
	"""

	def __init__(self, session: str):
		"""
		Initialize the RAG service by loading configuration and setting up
		the vector database components.
		"""
		self.session = session
		self.config = self._load_config()
		self.client = None
		self.vector_store = None
		self.embeddings = None
		self.text_splitter = None

		try:
			# Determine embedding model type
			embedding_model = self.config.get("embeddingModel", "text-embedding-3-small")

			# Use HuggingFace for sentence-transformers models
			if "sentence-transformers" in embedding_model:
				self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
			else:
				# Use OpenAI for other embedding models
				api_key = os.getenv("LLM_API_KEY", "")
				self.embeddings = OpenAIEmbeddings(
					model=embedding_model,
					api_key=api_key
				)

			# Initialize ChromaDB client
			chroma_path = self.config.get("chromaPath", "./chroma_data")

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
			self.config["ready"] = False
			self.config["error"] = str(e)

	def _load_config(self) -> dict:
		"""
		Load RAG configuration from rag_config.json file.

		Returns:
			Dictionary containing RAG configuration with defaults
		"""
		if os.path.exists(RAG_CONFIG_FILE):
			try:
				with open(RAG_CONFIG_FILE, 'r') as f:
					# load all json
					configs = json.load(f)

					return configs[self.session]
			except (json.JSONDecodeError, IOError, KeyError) as e:
				return self._get_defaults()
		else:
			return {"success": False, "error": f"Error loading config: config file not found"}


	def _get_defaults(self) -> dict:
		"""
		Get default RAG configuration values.

		Returns:
			Dictionary with default RAG configuration
		"""
		if os.path.exists(RAG_CONFIG_FILE):
			try:
				with open(RAG_CONFIG_FILE, 'r') as f:
					# load all json
					configs = json.load(f)

					return configs["default"]
			except (json.JSONDecodeError, IOError, KeyError) as e:
				return {"success": False, "error": f"Error loading config: no defaults set"}
		else:
			return {"success": False, "error": f"Error loading config: config file not found"}


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

			# Initialize text splitter for chunking documents
			self.text_splitter = RecursiveCharacterTextSplitter(
				chunk_size=self.config.get("chunkSize", 500),
				chunk_overlap=self.config.get("chunkOverlap", 50)
			)

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
				"size": len(file_content) if isinstance(file_content, str) else len(str(file_content)),
				"chunkSize": self.config.get("chunkSize", 500),
				"chunkOverlap": self.config.get("chunkOverlap", 50)
			}

			with rag_lock:
				self.config["documents"].append(doc_info)
				self.save_config()

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
				if score <= sim_threshold:
					relevant_docs.append({
						"content": doc.page_content,
						"metadata": doc.metadata,
						"score": score,
						"threshold": sim_threshold
					})

			return relevant_docs

		except Exception as e:
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
			return query, []

		context_parts = []
		docs_info = []

		for doc in relevant:
			source = doc["metadata"].get("source", "Unknown")
			chunk_info = f"[Source: {source}]\n{doc['content'][:500]}..."
			context_parts.append(chunk_info)
			docs_info.append(doc)

		context = "\n\n---\n\n".join(context_parts)
		augmented_prompt = f"""
			Based on the following document context:
			{context}

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
					return {"success": False, "error": f"Warning: Could not delete from collection: {e}"}

				deleted_doc = self.config["documents"].pop(doc_index)
				self.save_config()

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
				# Get all document IDs from the collection
				all_docs = self.collection.get()
				if all_docs and 'ids' in all_docs and all_docs["ids"]:
					self.collection.delete(ids=all_docs['ids'])
				else:
					# If no documents, just clear config
					self.config["documents"] = []
					self.save_config()

				self.config["documents"] = []
				self.save_config()

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

	def save_config(self):
		"""
		Save RAG configuration to rag_config.json file.
		"""
		if os.path.exists(RAG_CONFIG_FILE):
			try:
				with open(RAG_CONFIG_FILE, 'r') as f:
					# load all json
					configs = json.load(f)

					configs[self.session] = self.config
					with open(RAG_CONFIG_FILE, 'w') as file:
						json.dump(configs, file, indent=4)

			except (json.JSONDecodeError, IOError) as e:
				return {"success": False, "error": f"Error saving config: {e}"}
		else:
			return {"success": False, "error": f"Error saving config: config file not found"}

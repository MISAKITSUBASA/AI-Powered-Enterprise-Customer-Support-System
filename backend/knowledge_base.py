import os
import tempfile
from typing import List, Dict, Any, Optional, BinaryIO
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from dotenv import load_dotenv
from env_utils import get_openai_api_key

load_dotenv()
OPENAI_API_KEY = get_openai_api_key()

# Default paths for vector index storage
DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index.index")
DEFAULT_DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index_docs.npy")

class KnowledgeBase:
    """Vector database implementation using FAISS for semantic search of documents"""
    
    def __init__(self, index_path: Optional[str] = None):
        # Initialize with OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY
        )
        
        self.documents = []
        
        self.index_path = index_path if index_path else DEFAULT_INDEX_PATH
        self.documents_path = f"{self.index_path}_documents.npy" if index_path else DEFAULT_DOCUMENTS_PATH
        self.dimension = 1536  # Dimension of OpenAI ada-002 embeddings
        
        self._load_index_and_documents()
    
    def _load_index_and_documents(self):
        """Load existing FAISS index and document data from disk"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                
                if os.path.exists(self.documents_path):
                    self.documents = np.load(self.documents_path, allow_pickle=True).tolist()
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        except Exception:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
    
    def _get_loader_for_filetype(self, file_path: str, file_type: str):
        """Return appropriate document loader for each supported file type"""
        file_type = file_type.lower()
        
        if file_type == 'txt':
            return TextLoader(file_path)
        elif file_type == 'pdf':
            return PyPDFLoader(file_path)
        elif file_type in ['docx', 'doc']:
            return Docx2txtLoader(file_path)
        elif file_type in ['xlsx', 'xls']:
            return UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def add_document(self, file_content: BinaryIO, file_name: str, file_type: str, metadata: Dict[str, Any] = None) -> bool:
        """Process and add a document to the vector database with embeddings"""
        try:
            # Create temporary file for document processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            loader = self._get_loader_for_filetype(temp_path, file_type)
            
            # Load document with appropriate loader
            raw_documents = loader.load()
            
            # Add metadata to document chunks
            for doc in raw_documents:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata.update({
                    "file_name": file_name,
                    "file_type": file_type
                })
                if metadata:
                    doc.metadata.update(metadata)
            
            # Split into smaller chunks for better embedding
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(raw_documents)
            
            # Generate embeddings for document chunks
            texts = [doc.page_content for doc in documents]
            new_embeddings = self.embeddings.embed_documents(texts)
            
            # Add embeddings to FAISS index
            embeddings_array = np.array(new_embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            # Store document content and metadata
            original_count = len(self.documents)
            for i, doc in enumerate(documents):
                self.documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding_id": original_count + i
                })
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Persist to disk
            self._save_index_and_documents()
            
            return True
            
        except Exception:
            import traceback
            traceback.print_exc()
            return False
    
    def _save_index_and_documents(self):
        """Persist FAISS index and document data to disk"""
        try:
            index_dir = os.path.dirname(self.index_path)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
            
            faiss.write_index(self.index, self.index_path)
            
            np.save(self.documents_path, self.documents)
            
            return True
        except Exception:
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity"""
        if self.index.ntotal == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)
        
        # Normalize for cosine similarity
        query_embedding_normalized = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_normalized)
        
        # Search for similar vectors in FAISS index
        k = min(top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(query_embedding_normalized, k)
        
        # Prepare results with relevant metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                # Convert distance to similarity score
                similarity_score = float(max(0.0, 1.0 - (float(scores[0][i]) / 2.0)))
                
                doc = self.documents[idx]
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": similarity_score
                })
        
        # Sort by relevance and return top results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        return results

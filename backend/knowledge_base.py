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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define default paths for persistence
DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index.index")
DEFAULT_DOCUMENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index_docs.npy")

class KnowledgeBase:
    def __init__(self, index_path: Optional[str] = None):
        """Initialize the knowledge base with FAISS vector store and OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY
        )
        
        # Storage for document content and metadata
        self.documents = []
        self.document_embeddings = None
        
        # Create or load FAISS index
        self.index_path = index_path if index_path else DEFAULT_INDEX_PATH
        self.documents_path = f"{self.index_path}_documents.npy" if index_path else DEFAULT_DOCUMENTS_PATH
        self.dimension = 1536  # Dimension of OpenAI ada-002 embeddings
        
        # Try to load existing index and documents
        self._load_index_and_documents()
    
    def _load_index_and_documents(self):
        """Load index and documents from disk if they exist"""
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
        """Return appropriate document loader based on file type"""
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
        """
        Add a document to the knowledge base
        Returns True if successful, False otherwise
        """
        try:
            # Create a temporary file to use with document loaders
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            # Get the appropriate loader
            loader = self._get_loader_for_filetype(temp_path, file_type)
            
            # Load and split the document
            raw_documents = loader.load()
            
            # Add file metadata to each chunk
            for doc in raw_documents:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata.update({
                    "file_name": file_name,
                    "file_type": file_type
                })
                if metadata:
                    doc.metadata.update(metadata)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(raw_documents)
            
            # Get embeddings for each document chunk
            texts = [doc.page_content for doc in documents]
            new_embeddings = self.embeddings.embed_documents(texts)
            
            # Add to FAISS index
            embeddings_array = np.array(new_embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            # Store documents with their metadata
            original_count = len(self.documents)
            for i, doc in enumerate(documents):
                self.documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding_id": original_count + i  # Track the corresponding embedding index
                })
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Save the index and documents
            self._save_index_and_documents()
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False
    
    def _save_index_and_documents(self):
        """Save the index and documents to disk"""
        try:
            # Create directory if it doesn't exist
            index_dir = os.path.dirname(self.index_path)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save documents
            np.save(self.documents_path, self.documents)
            
            return True
        except Exception:
            return False
    
    def get_document_count(self) -> int:
        """Return the number of document chunks in the knowledge base"""
        return len(self.documents)
    
    def get_document_sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return a sample of documents for verification purposes"""
        if not self.documents:
            return []
        
        import random
        sample_size = min(n, len(self.documents))
        sample_indices = random.sample(range(len(self.documents)), sample_size)
        
        return [self.documents[i] for i in sample_indices]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant documents
        Returns list of dicts with content, metadata, and relevance score
        """
        if self.index.ntotal == 0:
            return []  # No documents in the index
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Normalize the embedding
        query_embedding_normalized = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding_normalized)
        
        # Search the index
        k = min(top_k * 2, self.index.ntotal)  # Get more results initially for better filtering
        scores, indices = self.index.search(query_embedding_normalized, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no more results
                # Convert distance score to similarity score (higher is better)
                # FAISS L2 distance: smaller is closer, so we invert and normalize
                # Convert numpy.float32 to Python float to ensure JSON serialization works
                similarity_score = float(max(0.0, 1.0 - (float(scores[0][i]) / 2.0)))
                
                doc = self.documents[idx]
                results.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": similarity_score
                })
        
        # Sort by score and take top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        return results
    
    def search_by_filename(self, filename: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find documents by filename - useful for debugging
        Returns matching document chunks
        """
        matches = []
        for doc in self.documents:
            if doc["metadata"].get("file_name", "").lower() == filename.lower():
                matches.append(doc)
                if len(matches) >= limit:
                    break
        
        return matches

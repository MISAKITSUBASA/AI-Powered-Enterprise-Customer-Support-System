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
                print(f"Loading existing index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                
                if os.path.exists(self.documents_path):
                    print(f"Loading documents from {self.documents_path}")
                    self.documents = np.load(self.documents_path, allow_pickle=True).tolist()
                    print(f"Loaded {len(self.documents)} documents")
            else:
                print(f"No existing index found at {self.index_path}, creating new index")
                self.index = faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            print(f"Error loading index or documents: {str(e)}")
            print("Creating new index")
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
                chunk_overlap=100
            )
            documents = text_splitter.split_documents(raw_documents)
            
            # Get embeddings for each document chunk
            texts = [doc.page_content for doc in documents]
            new_embeddings = self.embeddings.embed_documents(texts)
            
            # Add to FAISS index
            faiss.normalize_L2(np.array(new_embeddings).astype('float32'))
            self.index.add(np.array(new_embeddings).astype('float32'))
            
            # Store documents with their metadata
            for doc in documents:
                self.documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Save the index and documents
            self._save_index_and_documents()
            
            return True
            
        except Exception as e:
            print(f"Error adding document: {str(e)}")
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
            print(f"Saved index to {self.index_path}")
            
            # Save documents
            np.save(self.documents_path, self.documents)
            print(f"Saved {len(self.documents)} documents to {self.documents_path}")
            
            return True
        except Exception as e:
            print(f"Error saving index or documents: {str(e)}")
            return False
    
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
        scores, indices = self.index.search(query_embedding_normalized, min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no more results
                results.append({
                    "content": self.documents[idx]["content"],
                    "metadata": self.documents[idx]["metadata"],
                    "score": float(scores[0][i])
                })
        
        return results

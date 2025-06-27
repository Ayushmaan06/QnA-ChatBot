# Advanced LLM Chat Application with PDF Support - Streamlit Version
# Using Groq API, HuggingFace Embeddings, FAISS Vector Store, and Streamlit

import streamlit as st
import os
import pickle
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

# Core libraries
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 Advanced LLM Chat with PDF Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GroqLLM:
    """Groq API integration for LLM responses"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, messages: List[dict], model: str = "Gemma2-9b-it") -> str:
        """Generate response using Groq API"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error: Failed to get response from Groq API (Status: {response.status_code})"
                
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return f"Error: {str(e)}"

class HuggingFaceEmbeddings:
    """HuggingFace embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a lightweight but effective model"""
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            with st.spinner("Loading HuggingFace embedding model..."):
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Loaded HuggingFace model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            st.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding for a single text"""
        try:
            if self.model is None:
                self._load_model()
            
            embedding = self.model.encode([text])[0]
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return np.zeros(self.embedding_dim, dtype='float32')
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for multiple texts"""
        try:
            if self.model is None:
                self._load_model()
            
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dim), dtype='float32')

class FAISSVectorStore:
    """FAISS vector store for similarity search"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        self.texts = []
        self.metadata = []
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadata: List[dict] = None):
        """Add texts and their embeddings to the vector store"""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store texts and metadata
            self.texts.extend(texts)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(texts))
            
            logger.info(f"Added {len(texts)} texts to vector store. Total: {len(self.texts)}")
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            st.error(f"Error adding texts to vector store: {e}")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float, dict]]:
        """Search for similar texts"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.texts):
                    results.append((
                        self.texts[idx],
                        float(score),
                        self.metadata[idx]
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        return {
            "total_texts": len(self.texts),
            "embedding_dimension": self.embedding_dim,
            "index_size": self.index.ntotal
        }

class PDFProcessor:
    """Process PDF files and extract text"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Extract text
            reader = PdfReader(tmp_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            logger.info(f"Extracted text from PDF: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return f"Error processing PDF: {str(e)}"
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]

class StreamlitChatBot:
    """Main chatbot class for Streamlit"""
    
    def __init__(self):
        # Initialize session state
        if 'groq_llm' not in st.session_state:
            st.session_state.groq_llm = None
        if 'embeddings' not in st.session_state:
            st.session_state.embeddings = None
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'pdf_loaded' not in st.session_state:
            st.session_state.pdf_loaded = False
        if 'pdf_filename' not in st.session_state:
            st.session_state.pdf_filename = ""
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    def initialize_system(self, groq_api_key: str) -> bool:
        """Initialize the chatbot system"""
        try:
            with st.spinner("Initializing system..."):
                # Initialize Groq LLM
                st.session_state.groq_llm = GroqLLM(groq_api_key)
                
                # Initialize embeddings
                st.session_state.embeddings = HuggingFaceEmbeddings()
                
                # Initialize vector store
                st.session_state.vector_store = FAISSVectorStore(
                    st.session_state.embeddings.embedding_dim
                )
                
                st.session_state.system_initialized = True
                logger.info("✅ System initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            st.error(f"System initialization failed: {e}")
            return False
    
    def load_pdf(self, pdf_file) -> str:
        """Load and process PDF file"""
        if not st.session_state.system_initialized:
            return "❌ Please initialize the system first by entering your Groq API key"
        
        if pdf_file is None:
            return "❌ No PDF file provided"
        
        try:
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                pdf_processor = PDFProcessor()
                pdf_text = pdf_processor.extract_text_from_pdf(pdf_file)
                
                if pdf_text.startswith("Error"):
                    return pdf_text
                
                # Chunk the text
                chunks = pdf_processor.chunk_text(pdf_text)
                
                if not chunks:
                    return "❌ No text could be extracted from the PDF"
                
                # Create embeddings for chunks
                embeddings = st.session_state.embeddings.embed_texts(chunks)
                
                # Create metadata for chunks
                metadata = [
                    {
                        "source": pdf_file.name,
                        "chunk_id": i,
                        "timestamp": datetime.now().isoformat()
                    }
                    for i in range(len(chunks))
                ]
                
                # Add to vector store
                st.session_state.vector_store.add_texts(chunks, embeddings, metadata)
                
                # Update status
                st.session_state.pdf_loaded = True
                st.session_state.pdf_filename = pdf_file.name
                
                return f"✅ Successfully loaded PDF: {pdf_file.name}\n📄 Processed {len(chunks)} text chunks\n🔍 Ready for questions about the document!"
                
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return f"❌ Error loading PDF: {str(e)}"
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from PDF if available"""
        if not st.session_state.pdf_loaded or st.session_state.vector_store.index.ntotal == 0:
            return ""
        
        try:
            # Create query embedding
            query_embedding = st.session_state.embeddings.embed_text(query)
            
            # Search for relevant chunks
            results = st.session_state.vector_store.similarity_search(query_embedding, k=k)
            
            if not results:
                return ""
            
            # Debug logging
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            for i, (text, score, metadata) in enumerate(results):
                logger.info(f"Result {i+1}: Score={score:.4f}, Text={text[:100]}...")
            
            # Format context - use lower threshold for cosine similarity
            context_parts = []
            for text, score, metadata in results:
                # For cosine similarity, scores > 0.1 are usually relevant
                # Include more context by lowering threshold
                if score > 0.1:  # Lower threshold for better recall
                    context_parts.append(f"[Relevance: {score:.3f}] {text}")
            
            # If no results above threshold, include top result anyway
            if not context_parts and results:
                text, score, metadata = results[0]
                context_parts.append(f"[Relevance: {score:.3f}] {text}")
                logger.info(f"No results above threshold, included top result with score {score:.3f}")
            
            if context_parts:
                context = "\n\n--- RELEVANT DOCUMENT CONTEXT ---\n"
                context += "\n\n".join(context_parts)
                context += "\n--- END CONTEXT ---\n"
                logger.info(f"Returning context with {len(context_parts)} chunks")
                return context
            
            logger.info("No relevant context found")
            return ""
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""
    
    def chat_with_bot(self, message: str) -> str:
        """Chat with the bot"""
        if not st.session_state.system_initialized:
            return "❌ Please initialize the system first by entering your Groq API key"
        
        if not message.strip():
            return "Please enter a message"
        
        try:
            # Get relevant context from PDF if available
            pdf_context = self.get_relevant_context(message)
            
            # Prepare messages for Groq API
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an advanced AI assistant with access to chat history and document context.

Instructions:
1. Always provide helpful, accurate, and detailed responses
2. When document context is provided in the user's message, prioritize using that information
3. If you see "--- RELEVANT DOCUMENT CONTEXT ---" in the user's message, use that content to answer their question
4. Maintain conversation flow by referencing previous messages when relevant
5. If asked about document content and context is provided, explicitly mention you're referencing the uploaded PDF
6. Be conversational but professional
7. When using document context, cite specific information from the provided sections

Current Status: {f"Document '{st.session_state.pdf_filename}' is loaded and available for questions" if st.session_state.pdf_loaded else "No document currently loaded."}
"""
                }
            ]
            
            # Add chat history (last 10 exchanges to manage token limit)
            recent_history = st.session_state.chat_history[-20:] if len(st.session_state.chat_history) > 20 else st.session_state.chat_history
            for chat in recent_history:
                messages.extend([
                    {"role": "user", "content": chat["user"]},
                    {"role": "assistant", "content": chat["assistant"]}
                ])
            
            # Add current message with context
            current_content = message
            context_info = ""
            if pdf_context:
                current_content = f"{pdf_context}\n\nUser Question: {message}"
                context_info = "📄 Using PDF context"
            else:
                context_info = "💭 No PDF context found"
            
            messages.append({"role": "user", "content": current_content})
            
            # Log what we're sending to the LLM
            logger.info(f"Sending to LLM: {context_info}")
            logger.info(f"Message length: {len(current_content)} characters")
            if pdf_context:
                logger.info(f"Context included: {len(pdf_context)} characters")
            
            # Get response from Groq
            response = st.session_state.groq_llm.generate_response(messages)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "user": message,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "has_pdf_context": bool(pdf_context)
            })
            
            # Log the interaction
            logger.info(f"User: {message[:100]}...")
            logger.info(f"Bot: {response[:100]}...")
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        stats = {
            "system_initialized": st.session_state.system_initialized,
            "pdf_loaded": st.session_state.pdf_loaded,
            "pdf_filename": st.session_state.pdf_filename,
            "chat_exchanges": len(st.session_state.chat_history),
        }
        
        if st.session_state.vector_store:
            stats.update(st.session_state.vector_store.get_stats())
        
        return stats

def main():
    """Main Streamlit application"""
    
    # Initialize chatbot
    chatbot = StreamlitChatBot()
    
    # Header
    st.title("🤖 Advanced LLM Chat Application")
    st.markdown("""
    **Features:**
    - 🚀 **Groq API** for fast LLM responses  
    - 🤗 **HuggingFace Embeddings** for semantic understanding
    - 📄 **PDF Support** with intelligent document search
    - 🔍 **FAISS Vector Store** for similarity search
    - 💬 **Chat History** with context awareness
    """)
    
    # Sidebar for system configuration
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter your Groq API key (gsk_...)",
            help="Get your API key from console.groq.com"
        )
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            if groq_api_key:
                if chatbot.initialize_system(groq_api_key):
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ System initialization failed!")
            else:
                st.error("Please enter your Groq API key")
        
        st.divider()
        
        # PDF Upload section
        st.header("📄 PDF Upload")
        uploaded_pdf = st.file_uploader(
            "Upload PDF Document",
            type="pdf",
            help="Upload a PDF to enable document-based conversations"
        )
        
        if uploaded_pdf and st.session_state.system_initialized:
            if st.button("📤 Process PDF"):
                result = chatbot.load_pdf(uploaded_pdf)
                if result.startswith("✅"):
                    st.success(result)
                else:
                    st.error(result)
        
        st.divider()
        
        # System Statistics
        st.header("📊 System Stats")
        stats = chatbot.get_system_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chat Exchanges", stats.get("chat_exchanges", 0))
            st.metric("System Status", "✅ Ready" if stats.get("system_initialized") else "❌ Not Ready")
        
        with col2:
            st.metric("PDF Status", "✅ Loaded" if stats.get("pdf_loaded") else "❌ None")
            st.metric("Vector Chunks", stats.get("total_texts", 0))
        
        if stats.get("pdf_filename"):
            st.info(f"📄 Current PDF: {stats['pdf_filename']}")
        
        # Debug section
        if st.checkbox("🔍 Debug Mode"):
            st.subheader("Debug Information")
            
            if st.session_state.pdf_loaded:
                test_query = st.text_input("Test Context Retrieval:", placeholder="Enter a query to test PDF context retrieval")
                if test_query:
                    context = chatbot.get_relevant_context(test_query, k=3)
                    if context:
                        st.success("✅ Context retrieved successfully!")
                        with st.expander("View Retrieved Context"):
                            st.text(context)
                    else:
                        st.warning("⚠️ No context retrieved - this might be the issue!")
                        
                        # Show vector store stats
                        vector_stats = st.session_state.vector_store.get_stats()
                        st.json(vector_stats)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            chatbot.clear_chat_history()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system by entering your Groq API key in the sidebar.")
        st.stop()
    
    # Chat history display
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.write(chat["user"])
            
            # Assistant message
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
                
                # Show if PDF context was used
                if chat.get("has_pdf_context", False):
                    st.caption("📄 Response used PDF context")
    
    # Chat input
    user_input = st.chat_input("Ask me anything... I can use uploaded PDF context!")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.chat_with_bot(user_input)
            st.write(response)
            
            # Show if PDF context was used
            if st.session_state.pdf_loaded and st.session_state.vector_store.index.ntotal > 0:
                pdf_context = chatbot.get_relevant_context(user_input)
                if pdf_context:
                    st.caption("📄 Response used PDF context")
    
    # Example questions
    if not st.session_state.chat_history:
        st.subheader("💡 Example Questions")
        
        examples = [
            "Hello! What can you help me with?",
            "What's the main topic of the uploaded document?",
            "Can you summarize the key points from the PDF?",
            "What does the document say about [specific topic]?",
            "Compare the information in the document with current trends"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    # Simulate clicking the example
                    st.session_state.example_clicked = example
                    st.rerun()
        
        # Handle example clicks
        if hasattr(st.session_state, 'example_clicked'):
            example = st.session_state.example_clicked
            del st.session_state.example_clicked
            
            # Display user message
            with st.chat_message("user"):
                st.write(example)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.chat_with_bot(example)
                st.write(response)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Advanced LLM Chat Application • Built with Streamlit, Groq API, HuggingFace & FAISS
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

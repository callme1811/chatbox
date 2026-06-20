import os
import re
import json
import pickle
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SimpleVectorStore:
    def __init__(self, index_file="data/vector_index.pkl"):
        self.index_file = index_file
        self.documents = []  # List of dicts: {id, text, file_name, title, vector}
        self.vectors = None  # numpy array of vectors for fast operations
        self.load_index()

    def add_documents(self, docs):
        """Docs is a list of dicts with 'text', 'file_name', 'title', 'vector'"""
        for doc in docs:
            doc['id'] = len(self.documents)
            self.documents.append(doc)
        
        self.update_vectors()
        self.save_index()

    def update_vectors(self):
        if not self.documents:
            self.vectors = None
            return
        self.vectors = np.array([doc['vector'] for doc in self.documents], dtype=np.float32)

    def search(self, query_vector, k=5):
        """Returns top k documents and their cosine similarity scores"""
        if self.vectors is None or len(self.documents) == 0:
            return []
        
        # Calculate cosine similarity
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
            
        doc_norms = np.linalg.norm(self.vectors, axis=1)
        # Avoid division by zero
        doc_norms[doc_norms == 0] = 1e-10
        
        scores = np.dot(self.vectors, query_vector) / (doc_norms * q_norm)
        
        # Sort indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])
            })
        return results

    def save_index(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        # We don't want to save large numpy arrays directly in text, pickle is fast and works
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Saved vector index with {len(self.documents)} items to {self.index_file}")

    def load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    self.documents = pickle.load(f)
                self.update_vectors()
                print(f"Loaded vector index with {len(self.documents)} items from {self.index_file}")
            except Exception as e:
                print(f"Error loading vector index: {e}, starting fresh.")
                self.documents = []
                self.vectors = None

    def clear(self):
        self.documents = []
        self.vectors = None
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        print("Cleared vector index")


class RAGEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.vector_store = SimpleVectorStore(os.path.join(data_dir, "vector_index.pkl"))
        self.api_configured = False
        self.configure_genai()

    def configure_genai(self, api_key=None):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            
        if api_key:
            genai.configure(api_key=api_key)
            self.api_configured = True
            print("Gemini API configured successfully.")
            return True
        else:
            self.api_configured = False
            print("Gemini API key not found. Please set GEMINI_API_KEY in environment or .env file.")
            return False

    def parse_pdf(self, file_path):
        """Reads a PDF and returns its full text and pages"""
        reader = PdfReader(file_path)
        pages_content = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_content.append(text)
        return "\n".join(pages_content)

    def parse_txt(self, file_path):
        """Reads a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def split_into_articles(self, text, file_name):
        """
        Splits Vietnamese legal text into chunks.
        It tries to split by Article (Điều X) since it preserves legal context.
        If no articles are found, it falls back to character-based chunking.
        """
        # Regular expression to match "Điều X." or "Điều X:" or "Điều X" at start of line or after newlines
        # e.g., "Điều 1.", "Điều 12."
        pattern = r'(?:\n|^)(Điều\s+\d+[\.\s\:])'
        
        parts = re.split(pattern, text)
        
        chunks = []
        if len(parts) <= 1:
            # Fallback to character chunking
            print(f"No articles found in {file_name}. Falling back to character chunking.")
            chunk_size = 1200
            overlap = 200
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "file_name": file_name,
                        "title": f"Đoạn văn trong {file_name}"
                    })
                start += chunk_size - overlap
            return chunks

        # The first part is prologue metadata
        prologue = parts[0].strip()
        if prologue and len(prologue) > 100:
            chunks.append({
                "text": prologue,
                "file_name": file_name,
                "title": f"Phần mở đầu - {file_name}"
            })

        # Process the split articles
        i = 1
        while i < len(parts):
            article_header = parts[i].strip()
            article_content = parts[i+1].strip() if i+1 < len(parts) else ""
            
            # Combine header with content
            full_article_text = f"{article_header} {article_content}"
            
            # Extract article name if it exists (usually the first sentence of the article)
            first_line = article_content.split('\n')[0] if article_content else ""
            title = f"{article_header} {first_line[:50]}..." if len(first_line) > 50 else f"{article_header} {first_line}"
            
            # If the article content is too long, we might want to split it further
            # but usually a single article is between 500-3000 characters, which is perfect for LLM context.
            if len(full_article_text) > 3000:
                # Sub-split into chunks of ~1500 chars with 200 overlap, keeping article header context
                start = 0
                sub_idx = 1
                while start < len(full_article_text):
                    end = min(start + 1500, len(full_article_text))
                    sub_text = full_article_text[start:end]
                    # Prepend context header if we are not at the start
                    if start > 0:
                        sub_text = f"({article_header} - Tiếp tục) ... {sub_text}"
                    chunks.append({
                        "text": sub_text.strip(),
                        "file_name": file_name,
                        "title": f"{article_header} (Mục {sub_idx}) - {file_name}"
                    })
                    start += 1300
                    sub_idx += 1
            else:
                chunks.append({
                    "text": full_article_text,
                    "file_name": file_name,
                    "title": f"{title} - {file_name}"
                })
            i += 2

        return chunks

    def process_file_and_generate_embeddings(self, file_path):
        """Parses a file, chunks it, calls Gemini to get embeddings, and returns documents"""
        if not self.api_configured:
            raise ValueError("Gemini API key is not configured. Cannot generate embeddings.")

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        print(f"Processing {file_name}...")
        if ext == '.pdf':
            text = self.parse_pdf(file_path)
        elif ext in ['.txt', '.md']:
            text = self.parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        chunks = self.split_into_articles(text, file_name)
        print(f"Split {file_name} into {len(chunks)} chunks.")
        
        # Batch generate embeddings to save API network overhead
        docs = []
        batch_size = 50  # embedding API supports batching
        
        for idx in range(0, len(chunks), batch_size):
            batch = chunks[idx:idx+batch_size]
            texts = [c['text'] for c in batch]
            
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                
                embeddings = response['embedding']
                for j, emb in enumerate(embeddings):
                    doc = batch[j]
                    doc['vector'] = emb
                    docs.append(doc)
            except Exception as e:
                print(f"Error generating embedding batch {idx}-{idx+len(batch)}: {e}")
                # Fallback for single requests in case batch fails
                for c in batch:
                    try:
                        res = genai.embed_content(
                            model="models/text-embedding-004",
                            content=c['text'],
                            task_type="retrieval_document"
                        )
                        c['vector'] = res['embedding'][0] if isinstance(res['embedding'][0], list) else res['embedding']
                        docs.append(c)
                    except Exception as single_err:
                        print(f"Failed to embed single chunk: {single_err}")
        
        return docs

    def build_index(self):
        """Indexes all files in the data directory if they are not already indexed"""
        if not self.api_configured:
            print("Cannot build index: Gemini API not configured.")
            return False
            
        # Clear existing index to avoid duplicate content
        self.vector_store.clear()
        
        all_docs = []
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.isdir(file_path):
                continue
            if os.path.splitext(file_name)[1].lower() in ['.pdf', '.txt', '.md']:
                try:
                    docs = self.process_file_and_generate_embeddings(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error indexing file {file_name}: {e}")
                    
        if all_docs:
            self.vector_store.add_documents(all_docs)
            print(f"Index successfully built with {len(all_docs)} segments.")
            return True
        else:
            print("No documents were indexed.")
            return False

    def query(self, user_query, k=5):
        """Performs RAG search and returns response generator & source chunks"""
        if not self.api_configured:
            raise ValueError("Gemini API key is not configured.")
            
        if len(self.vector_store.documents) == 0:
            # Let's try building index if vector store is empty but files exist
            files = [f for f in os.listdir(self.data_dir) if os.path.splitext(f)[1].lower() in ['.pdf', '.txt', '.md']]
            if files:
                print("Vector store is empty, rebuilding index from files...")
                self.build_index()
            else:
                raise ValueError("No law documents indexed. Please upload or download law documents first.")

        # 1. Embed query
        query_response = genai.embed_content(
            model="models/text-embedding-004",
            content=user_query,
            task_type="retrieval_query"
        )
        query_vector = query_response['embedding']
        
        # 2. Search
        search_results = self.vector_store.search(query_vector, k=k)
        
        if not search_results:
            return "Không tìm thấy thông tin luật liên quan trong cơ sở dữ liệu.", []

        # 3. Formulate Prompt
        context_parts = []
        sources = []
        for i, res in enumerate(search_results):
            doc = res['document']
            score = res['score']
            context_parts.append(f"--- TRÍCH DẪN NGUỒN {i+1} (Độ phù hợp: {score:.2f}, Nguồn: {doc['title']}) ---\n{doc['text']}")
            sources.append({
                "title": doc["title"],
                "file_name": doc["file_name"],
                "text": doc["text"][:300] + "...",
                "score": score
            })
            
        context = "\n\n".join(context_parts)
        
        prompt = f"""Bạn là một trợ lý pháp lý ảo (AI Chatbot) chuyên nghiệp về Luật pháp Y tế, Khám chữa bệnh, Bảo hiểm y tế và Dược phẩm tại Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác, đáng tin cậy và chuyên nghiệp DỰA HOÀN TOÀN vào các đoạn trích lục văn bản luật được cung cấp dưới đây.

Quy tắc trả lời câu hỏi:
1. Trả lời bằng tiếng Việt, ngôn từ lịch sự, khách quan và chuyên nghiệp.
2. Sử dụng đúng thông tin trong phần "Thông tin các Điều luật cung cấp". Không tự ý suy diễn, suy đoán hoặc thêm các thông tin chưa có trong tài liệu.
3. Nếu thông tin cung cấp không chứa câu trả lời cho câu hỏi của người dùng, hãy trả lời trung thực: "Dựa vào các văn bản luật hiện tại, tôi không tìm thấy thông tin cụ thể về câu hỏi của bạn." Không tự bịa câu trả lời.
4. Ở cuối câu trả lời, hãy liệt kê danh sách trích dẫn cụ thể các Điều luật đã sử dụng dưới dạng danh sách bullet points rõ ràng (ví dụ: "* Điều 12, Luật Khám bệnh, chữa bệnh 2023").

Thông tin các Điều luật cung cấp:
{context}

Câu hỏi của người dùng: {user_query}

Hãy trả lời chi tiết và rõ ràng theo đúng cấu trúc:"""

        # 4. Generate with streaming
        # We use gemini-1.5-flash by default as it is fast and robust
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        
        try:
            response_stream = model.generate_content(prompt, stream=True)
            return response_stream, sources
        except Exception as e:
            print(f"Error calling Gemini generation: {e}")
            raise e

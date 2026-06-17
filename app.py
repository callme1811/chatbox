
import os
import re
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

APP_TITLE = "Chat hỏi đáp pháp luật về khám chữa bệnh"

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "chunks.pkl"

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"

MIN_SCORE = 0.45
DEFAULT_TOP_K = 5


SYSTEM_PROMPT = """
Bạn là chatbot hỏi đáp pháp luật về khám chữa bệnh tại Việt Nam.

NHIỆM VỤ:
- Trả lời câu hỏi của người dùng dựa trên các văn bản pháp luật đã được nạp vào hệ thống RAG.
- Văn bản có thể gồm: Luật Khám bệnh, chữa bệnh; Luật Bảo hiểm y tế; Luật Dược; nghị định; thông tư và tài liệu pháp luật y tế liên quan.

NGUYÊN TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong NGỮ CẢNH VĂN BẢN LUẬT được cung cấp.
2. Không tự bịa điều luật, số điều, mức phạt, ngày ban hành, quyền lợi hoặc quy định nếu tài liệu không nêu.
3. Nếu tài liệu không có căn cứ, trả lời đúng câu:
   "Tôi chưa tìm thấy căn cứ trong tài liệu đã nạp để trả lời câu hỏi này."
4. Trả lời bằng tiếng Việt.
5. Câu trả lời phải có cấu trúc rõ ràng.
6. Bắt buộc trích dẫn nguồn theo tên file và số trang.

ĐỊNH DẠNG OUTPUT BẮT BUỘC:

## Trả lời
Trả lời trực tiếp câu hỏi của người dùng dựa trên văn bản luật.

## Căn cứ từ văn bản luật
Liệt kê căn cứ đã dùng:
- [Nguồn 1: tên file, trang X]
- [Nguồn 2: tên file, trang Y]

## Giải thích ngắn gọn
Giải thích dễ hiểu, không suy diễn ngoài tài liệu.

## Lưu ý
Câu trả lời chỉ có giá trị tham khảo, không thay thế tư vấn pháp lý chính thức.
"""


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


def get_api_key() -> str:
    """
    Trên Streamlit Cloud: Settings -> Secrets:
    GEMINI_API_KEY = "AIza..."
    Local dev có thể dùng biến môi trường GEMINI_API_KEY.
    Không đọc .env để tránh lộ key lên Git.
    """
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GEMINI_API_KEY", "")

    return str(key).strip()


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chunk_id(source: str, page: int, text: str) -> str:
    raw = f"{source}-{page}-{text[:120]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def read_pdf(path: Path) -> List[Chunk]:
    result = []
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        st.warning(f"Không đọc được PDF {path.name}: {e}")
        return result

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            result.append(
                Chunk(
                    text=text,
                    source=path.name,
                    page=page_number,
                    chunk_id=make_chunk_id(path.name, page_number, text),
                )
            )
    return result


def read_txt(path: Path) -> List[Chunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"Không đọc được TXT {path.name}: {e}")
        return []

    text = clean_text(text)
    if not text:
        return []

    return [Chunk(text=text, source=path.name, page=1, chunk_id=make_chunk_id(path.name, 1, text))]


def split_text(text: str, chunk_size: int = 850, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        part = " ".join(words[start:end]).strip()
        if part:
            chunks.append(part)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def list_data_files() -> List[Path]:
    ensure_dirs()
    return sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in [".pdf", ".txt"]])


def data_fingerprint() -> str:
    """
    Nếu dữ liệu đổi, fingerprint đổi -> app tự tạo lại index.
    """
    h = hashlib.sha256()
    files = list_data_files()
    for path in files:
        stat = path.stat()
        h.update(path.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def load_documents() -> List[Chunk]:
    ensure_dirs()
    all_chunks = []

    for path in list_data_files():
        if path.suffix.lower() == ".pdf":
            pages = read_pdf(path)
        elif path.suffix.lower() == ".txt":
            pages = read_txt(path)
        else:
            continue

        for page_chunk in pages:
            for idx, part in enumerate(split_text(page_chunk.text), start=1):
                all_chunks.append(
                    Chunk(
                        text=part,
                        source=page_chunk.source,
                        page=page_chunk.page,
                        chunk_id=make_chunk_id(page_chunk.source, page_chunk.page, f"{idx}-{part}"),
                    )
                )
    return all_chunks


def embed_texts(client, texts: List[str], task_type: str) -> np.ndarray:
    vectors = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini không trả về embedding.")

        vectors.append(response.embeddings[0].values)

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_index(client) -> Tuple[int, int]:
    chunks = load_documents()
    if not chunks:
        raise RuntimeError("Không tìm thấy file PDF/TXT trong thư mục data/.")

    embeddings = embed_texts(
        client=client,
        texts=[chunk.text for chunk in chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    ensure_dirs()
    faiss.write_index(index, str(INDEX_FILE))

    meta = {
        "fingerprint": data_fingerprint(),
        "chunks": chunks,
    }
    with META_FILE.open("wb") as f:
        pickle.dump(meta, f)

    return len(chunks), embeddings.shape[1]


def load_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return None, None, None

    index = faiss.read_index(str(INDEX_FILE))

    with META_FILE.open("rb") as f:
        meta = pickle.load(f)

    if isinstance(meta, dict):
        return index, meta.get("chunks", []), meta.get("fingerprint")

    # Tương thích với chunks.pkl bản cũ.
    return index, meta, None


def reset_index():
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if META_FILE.exists():
        META_FILE.unlink()


def ensure_index_ready(client):
    """
    Tự động tạo/cập nhật index khi:
    - Chưa có index.
    - File data thay đổi.
    """
    if client is None:
        return False

    files = list_data_files()
    if not files:
        return False

    index, chunks, old_fp = load_index()
    current_fp = data_fingerprint()

    if index is not None and chunks and old_fp == current_fp:
        return True

    with st.spinner("Đang tự động tạo/cập nhật chỉ mục RAG từ tài liệu..."):
        n_chunks, dim = build_index(client)
        st.success(f"Đã tự động tạo/cập nhật chỉ mục RAG: {n_chunks} đoạn, vector {dim} chiều.")

    return True


def lexical_score(question: str, text: str) -> float:
    q_words = set(re.findall(r"\w+", question.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))

    if not q_words or not t_words:
        return 0.0

    return len(q_words.intersection(t_words)) / len(q_words)


def retrieve(client, question: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[Chunk, float]]:
    index, chunks, _ = load_index()
    if index is None or not chunks:
        raise RuntimeError("Chưa có chỉ mục RAG. Hệ thống chưa tạo được index từ tài liệu.")

    q_vec = embed_texts(client=client, texts=[question], task_type="RETRIEVAL_QUERY")
    scores, ids = index.search(q_vec, top_k * 2)

    results = []
    for pos, idx in enumerate(ids[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        vector_score = float(scores[0][pos])
        keyword_score = lexical_score(question, chunk.text)
        final_score = 0.8 * vector_score + 0.2 * keyword_score
        results.append((chunk, final_score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def format_context(contexts: List[Tuple[Chunk, float]]) -> str:
    blocks = []
    for idx, (chunk, score) in enumerate(contexts, start=1):
        blocks.append(
            f"""
[Nguồn {idx}]
Tên file: {chunk.source}
Trang: {chunk.page}
Điểm liên quan: {score:.3f}
Nội dung văn bản:
{chunk.text}
"""
        )
    return "\n\n".join(blocks)


def no_evidence_answer(reason: str) -> str:
    return f"""
## Trả lời
Tôi chưa tìm thấy căn cứ trong tài liệu đã nạp để trả lời câu hỏi này.

## Căn cứ từ văn bản luật
Không có căn cứ phù hợp trong tài liệu đã nạp.

## Giải thích ngắn gọn
{reason}

## Lưu ý
Câu trả lời chỉ có giá trị tham khảo, không thay thế tư vấn pháp lý chính thức.
"""


def answer_question(client, question: str, contexts: List[Tuple[Chunk, float]]) -> str:
    if not contexts:
        return no_evidence_answer("Hệ thống không truy xuất được đoạn văn bản nào liên quan đến câu hỏi.")

    best_score = contexts[0][1]
    if best_score < MIN_SCORE:
        return no_evidence_answer(
            f"Câu hỏi có thể nằm ngoài phạm vi tài liệu hiện có. Điểm liên quan cao nhất chỉ đạt {best_score:.3f}, thấp hơn ngưỡng an toàn {MIN_SCORE}."
        )

    prompt = f"""
{SYSTEM_PROMPT}

NGỮ CẢNH VĂN BẢN LUẬT:
{format_context(contexts)}

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

YÊU CẦU RIÊNG:
- Trả lời đúng trọng tâm câu hỏi.
- Bắt buộc dùng định dạng OUTPUT đã yêu cầu.
- Bắt buộc ghi rõ nguồn, tên file và trang.
- Không dùng kiến thức bên ngoài NGỮ CẢNH VĂN BẢN LUẬT.
- Nếu tài liệu chỉ có căn cứ một phần, hãy nói rõ phần nào có căn cứ, phần nào chưa có căn cứ.
"""

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1, top_p=0.8),
    )

    if not response.text:
        return no_evidence_answer("Gemini không tạo được câu trả lời.")

    return response.text


def delete_data_file(path: Path):
    if path.exists():
        path.unlink()
    reset_index()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sources(contexts: List[Tuple[Chunk, float]]):
    if not contexts:
        return

    with st.expander("Các đoạn văn bản được dùng làm căn cứ"):
        for idx, (chunk, score) in enumerate(contexts, start=1):
            st.markdown(f"**Nguồn {idx}: {chunk.source}, trang {chunk.page} — điểm liên quan {score:.3f}**")
            st.write(chunk.text)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
    ensure_dirs()
    init_session_state()

    st.title("⚖️ Chat hỏi đáp pháp luật về khám chữa bệnh")
    st.caption("Input: văn bản luật PDF/TXT | RAG + Streamlit + Gemini 2.5 | Output: trả lời dựa trên văn bản luật")

    api_key = get_api_key()
    client = get_client(api_key)

    with st.sidebar:
        st.header("Cấu hình")

        if api_key:
            st.success("Đã nhận GEMINI_API_KEY từ Streamlit Secrets.")
        else:
            st.error("Chưa có GEMINI_API_KEY. Vào Streamlit Cloud → Settings → Secrets để thêm key.")

        top_k = st.slider("Số đoạn truy xuất", min_value=3, max_value=10, value=DEFAULT_TOP_K)

        st.write(f"Model trả lời: `{GEN_MODEL}`")
        st.write(f"Model embedding: `{EMBED_MODEL}`")
        st.write(f"Ngưỡng chống trả lời ngoài tài liệu: `{MIN_SCORE}`")

        st.divider()
        st.subheader("Văn bản luật")

        files = list_data_files()
        if files:
            st.write("Văn bản đang có trong thư mục data/:")
            for file_path in files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📄 {file_path.name}")
                with col2:
                    if st.button("Xóa", key=f"delete-{file_path.name}"):
                        delete_data_file(file_path)
                        st.rerun()
        else:
            st.warning("Chưa có văn bản luật nào trong data/.")

        uploaded_files = st.file_uploader("Thêm file PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = DATA_DIR / uploaded_file.name
                save_path.write_bytes(uploaded_file.getbuffer())
            reset_index()
            st.success(f"Đã lưu {len(uploaded_files)} file. Hệ thống sẽ tự tạo lại index.")
            st.rerun()

        st.divider()

        if st.button("Xóa lịch sử chat"):
            st.session_state.messages = []
            st.session_state.last_contexts = []
            st.rerun()

        if st.button("Xóa index RAG"):
            reset_index()
            st.success("Đã xóa index RAG. Hệ thống sẽ tự tạo lại khi cần.")

    if client is None:
        st.warning("Thiếu GEMINI_API_KEY nên chưa thể tạo index hoặc trả lời.")
        st.stop()

    try:
        ensure_index_ready(client)
    except Exception as e:
        st.error(f"Lỗi khi tự động tạo/cập nhật chỉ mục RAG: {e}")
        st.stop()

    st.subheader("Đặt câu hỏi pháp luật")

    with st.expander("Ví dụ câu hỏi"):
        st.markdown("- Người bệnh có những quyền gì khi khám chữa bệnh?")
        st.markdown("- Bảo hiểm y tế thanh toán chi phí khám chữa bệnh như thế nào?")
        st.markdown("- Luật Dược quy định gì về thuốc kê đơn?")
        st.markdown("- Bệnh án điện tử có vai trò gì trong chuyển đổi số y tế?")

    render_chat_history()

    question = st.chat_input("Nhập câu hỏi liên quan đến pháp luật khám chữa bệnh...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Đang truy xuất văn bản luật liên quan..."):
                    contexts = retrieve(client, question, top_k=top_k)

                with st.spinner("Gemini 2.5 đang tạo câu trả lời dựa trên văn bản luật..."):
                    answer = answer_question(client, question, contexts)
            except Exception as e:
                contexts = []
                answer = no_evidence_answer(f"Lỗi hệ thống khi xử lý câu hỏi: {e}")

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_contexts = contexts

    render_sources(st.session_state.last_contexts)

    st.divider()
    st.info(
        "Ứng dụng chỉ trả lời dựa trên các văn bản luật đã nạp. "
        "Nếu tài liệu không có căn cứ, hệ thống sẽ từ chối trả lời thay vì tự suy diễn."
    )


if __name__ == "__main__":
    main()

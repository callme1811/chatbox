import hashlib
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

# Dùng model nhẹ hơn để đỡ lỗi quota.
GEN_MODEL = "gemini-2.5-flash"

FALLBACK_GEN_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

MIN_SCORE = 0.25
DEFAULT_TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 8
MAX_CONTEXT_CHARS = 12000

SYSTEM_PROMPT = """
Bạn là chatbot hỏi đáp pháp luật về khám chữa bệnh tại Việt Nam.

NHIỆM VỤ:
- Trả lời câu hỏi của người dùng dựa trên các văn bản pháp luật đã được nạp vào hệ thống RAG.
- Văn bản có thể gồm: Luật Khám bệnh, chữa bệnh; Luật Bảo hiểm y tế; Luật Dược; nghị định; thông tư và tài liệu pháp luật y tế liên quan.

NGUYÊN TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong NGỮ CẢNH VĂN BẢN LUẬT được cung cấp.
2. Không tự bịa điều luật, số điều, mức phạt, ngày ban hành, quyền lợi hoặc quy định nếu tài liệu không nêu.
3. Nếu tài liệu không có căn cứ, trả lời đúng ý:
   "Tôi chưa tìm thấy căn cứ trong tài liệu đã nạp để trả lời câu hỏi này."
4. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu.
5. Không trả lời quá ngắn hoặc chung chung. Nếu có căn cứ, phải giải thích đủ ý theo căn cứ.
6. Bắt buộc trích dẫn nguồn theo tên file và số trang.
7. Nếu ngữ cảnh chỉ có tiêu đề chương/mục mà không có nội dung điều khoản cụ thể, phải nói rõ tài liệu chưa đủ nội dung chi tiết.
8. Nếu có nhiều căn cứ liên quan, tổng hợp theo từng ý; không chỉ nêu một câu kết luận.

ĐỊNH DẠNG OUTPUT BẮT BUỘC:

## Trả lời
- Trả lời trực tiếp câu hỏi.
- Viết thành 4 đến 8 gạch đầu dòng hoặc 2 đến 4 đoạn ngắn.
- Mỗi ý quan trọng phải gắn với căn cứ trong tài liệu.
- Nếu có điều kiện, ngoại lệ, phạm vi áp dụng thì nêu rõ.

## Căn cứ từ văn bản luật
Liệt kê căn cứ đã dùng:
- [Nguồn 1: tên file, trang X] Tóm tắt ngắn nội dung căn cứ.
- [Nguồn 2: tên file, trang Y] Tóm tắt ngắn nội dung căn cứ.

## Giải thích ngắn gọn
Giải thích dễ hiểu hơn cho người không chuyên, khoảng 3 đến 6 câu.

## Lưu ý
Câu trả lời chỉ có giá trị tham khảo, không thay thế tư vấn pháp lý chính thức.
"""


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


def get_api_key() -> str:
    """
    Chỉ nên đặt GEMINI_API_KEY trong Streamlit Secrets.
    Local có thể dùng biến môi trường, nhưng không commit .env lên GitHub.
    """
    key = ""
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
    raw = f"{source}-{page}-{text[:180]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def read_pdf(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        st.warning(f"Không đọc được PDF {path.name}: {exc}")
        return chunks

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""

        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source=path.name,
                    page=page_number,
                    chunk_id=make_chunk_id(path.name, page_number, text),
                )
            )

    return chunks


def read_txt(path: Path) -> List[Chunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        st.warning(f"Không đọc được TXT {path.name}: {exc}")
        return []

    text = clean_text(text)
    if not text:
        return []

    return [
        Chunk(
            text=text,
            source=path.name,
            page=1,
            chunk_id=make_chunk_id(path.name, 1, text),
        )
    ]


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
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
    return sorted(path for path in DATA_DIR.glob("*") if path.suffix.lower() in {".pdf", ".txt"})


def data_fingerprint() -> str:
    h = hashlib.sha256()

    for path in list_data_files():
        stat = path.stat()
        h.update(path.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))

    return h.hexdigest()


def load_documents() -> List[Chunk]:
    all_chunks: List[Chunk] = []

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


def is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "quota" in text.lower()
        or "rate limit" in text.lower()
    )


def is_overload_error(exc: Exception) -> bool:
    text = str(exc)
    return "503" in text or "UNAVAILABLE" in text or "overload" in text.lower()


def friendly_error_answer(exc: Exception) -> str:
    if is_quota_error(exc):
        return """
## Trả lời
Hệ thống AI đang tạm thời vượt giới hạn sử dụng, nên chưa thể tạo câu trả lời.

## Căn cứ từ văn bản luật
Chưa thể tạo câu trả lời ở thời điểm hiện tại vì Gemini API đã hết quota hoặc vượt giới hạn theo phút/ngày.

## Giải thích ngắn gọn
Đây là lỗi quota của API, không phải do câu hỏi hoặc tài liệu pháp luật. Bạn có thể thử lại sau vài phút, đổi API key thuộc project khác, hoặc bật billing cho project Google Cloud.

## Lưu ý
Ứng dụng vẫn hoạt động, nhưng cần quota Gemini còn khả dụng để tạo câu trả lời.
""".strip()

    if is_overload_error(exc):
        return """
## Trả lời
Hệ thống AI đang quá tải tạm thời, vui lòng thử lại sau vài giây.

## Căn cứ từ văn bản luật
Chưa thể tạo câu trả lời ở thời điểm hiện tại vì dịch vụ Gemini đang quá tải.

## Giải thích ngắn gọn
Đây là lỗi tạm thời từ Gemini, không phải do tài liệu hoặc câu hỏi của bạn.

## Lưu ý
Bạn có thể gửi lại câu hỏi sau vài giây.
""".strip()

    return """
## Trả lời
Hệ thống gặp lỗi khi xử lý câu hỏi.

## Căn cứ từ văn bản luật
Chưa thể tạo câu trả lời ở thời điểm hiện tại.

## Giải thích ngắn gọn
Vui lòng kiểm tra API key, quota Gemini, file tài liệu và thử lại.

## Lưu ý
Câu trả lời chỉ có giá trị tham khảo, không thay thế tư vấn pháp lý chính thức.
""".strip()


def _embed_one(client, text: str, task_type: str) -> List[float]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    embeddings = getattr(response, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("Gemini không trả về embedding.")
    return embeddings[0].values


def _embed_batch(client, texts: List[str], task_type: str) -> List[List[float]]:
    """
    Thử batch trước, nếu SDK/API không nhận batch thì fallback từng đoạn.
    """
    try:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings = getattr(response, "embeddings", None) or []
        if len(embeddings) == len(texts):
            return [item.values for item in embeddings]
    except Exception:
        pass

    return [_embed_one(client, text, task_type) for text in texts]


def embed_texts(client, texts: List[str], task_type: str) -> np.ndarray:
    if not texts:
        raise ValueError("Danh sách văn bản để embedding đang rỗng.")

    vectors: List[List[float]] = []

    progress = None
    if len(texts) > EMBED_BATCH_SIZE:
        progress = st.progress(0, text="Đang tạo embedding...")

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(client, batch, task_type))

        if progress:
            progress.progress(
                min((start + len(batch)) / len(texts), 1.0),
                text="Đang tạo embedding...",
            )

    if progress:
        progress.empty()

    arr = np.array(vectors, dtype="float32")
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise RuntimeError("Embedding trả về không hợp lệ.")

    faiss.normalize_L2(arr)
    return arr


def build_index(client) -> Tuple[int, int]:
    chunks = load_documents()
    if not chunks:
        raise RuntimeError("Không tìm thấy nội dung đọc được trong thư mục data/. Hãy thêm file PDF/TXT có text.")

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
        "embed_model": EMBED_MODEL,
        "dim": int(embeddings.shape[1]),
    }

    with META_FILE.open("wb") as file:
        pickle.dump(meta, file)

    return len(chunks), int(embeddings.shape[1])


def reset_index() -> None:
    for path in (INDEX_FILE, META_FILE):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def load_index() -> Tuple[Optional[faiss.Index], List[Chunk], Optional[str]]:
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return None, [], None

    try:
        index = faiss.read_index(str(INDEX_FILE))
        with META_FILE.open("rb") as file:
            meta = pickle.load(file)
    except Exception:
        reset_index()
        return None, [], None

    if isinstance(meta, dict):
        chunks = meta.get("chunks", []) or []
        fingerprint = meta.get("fingerprint")
    else:
        chunks = meta or []
        fingerprint = None

    if index.ntotal != len(chunks):
        reset_index()
        return None, [], None

    return index, chunks, fingerprint


def ensure_index_ready(client) -> bool:
    if client is None:
        return False

    if not list_data_files():
        return False

    index, chunks, old_fp = load_index()
    current_fp = data_fingerprint()

    if index is not None and chunks and old_fp == current_fp:
        return True

    with st.spinner("Đang tự động tạo/cập nhật chỉ mục RAG từ tài liệu..."):
        n_chunks, dim = build_index(client)
        st.success(f"Đã tạo/cập nhật chỉ mục RAG: {n_chunks} đoạn, vector {dim} chiều.")

    return True


def lexical_score(question: str, text: str) -> float:
    q_words = set(re.findall(r"\w+", question.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))

    if not q_words or not t_words:
        return 0.0

    return len(q_words & t_words) / len(q_words)


def retrieve(client, question: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[Chunk, float]]:
    index, chunks, _ = load_index()

    if index is None or not chunks:
        raise RuntimeError("Chưa có chỉ mục RAG. Hệ thống chưa tạo được index từ tài liệu.")

    safe_top_k = max(1, min(top_k * 2, index.ntotal))
    q_vec = embed_texts(client=client, texts=[question], task_type="RETRIEVAL_QUERY")
    scores, ids = index.search(q_vec, safe_top_k)

    results: List[Tuple[Chunk, float]] = []

    for pos, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]
        vector_score = float(scores[0][pos])
        keyword_score = lexical_score(question, chunk.text)
        final_score = 0.75 * vector_score + 0.25 * keyword_score
        results.append((chunk, final_score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def format_context(contexts: List[Tuple[Chunk, float]]) -> str:
    blocks: List[str] = []
    used_chars = 0

    for idx, (chunk, score) in enumerate(contexts, start=1):
        block = f"""
[Nguồn {idx}]
Tên file: {chunk.source}
Trang: {chunk.page}
Điểm liên quan: {score:.3f}
Nội dung văn bản:
{chunk.text}
""".strip()

        if used_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        used_chars += len(block)

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
""".strip()


def generate_with_fallback(client, prompt: str) -> str:
    last_error: Optional[Exception] = None

    for model_name in FALLBACK_GEN_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.85,
                    max_output_tokens=1400,
                ),
            )
            text = getattr(response, "text", "") or ""
            if text.strip():
                return text.strip()

        except Exception as exc:
            last_error = exc

            # Quota thì không thử tiếp nhiều lần để khỏi tốn request.
            if is_quota_error(exc):
                raise exc

            # Quá tải thì thử model fallback.
            if is_overload_error(exc):
                time.sleep(1)
                continue

            raise exc

    if last_error:
        raise last_error

    return ""


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
- Trả lời đúng trọng tâm câu hỏi, nhưng không được trả lời quá ngắn.
- Phải khai thác tối đa các đoạn ngữ cảnh được cung cấp.
- Ưu tiên nêu rõ: quy định chính, chủ thể áp dụng, điều kiện, quyền/nghĩa vụ, hệ quả pháp lý nếu tài liệu có nêu.
- Bắt buộc dùng định dạng OUTPUT đã yêu cầu.
- Bắt buộc ghi rõ nguồn, tên file và trang cho từng căn cứ.
- Không dùng kiến thức bên ngoài NGỮ CẢNH VĂN BẢN LUẬT.
- Nếu tài liệu chỉ có căn cứ một phần, hãy nói rõ phần nào có căn cứ, phần nào chưa có căn cứ.
- Nếu chỉ tìm được tiêu đề chương/mục, không được suy diễn nội dung điều luật; hãy nói tài liệu chưa đủ chi tiết.
""".strip()

    text = generate_with_fallback(client, prompt)
    if not text:
        return no_evidence_answer("Gemini không tạo được câu trả lời.")

    return text


def safe_upload_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w\-.()\sÀ-ỹ]", "_", name, flags=re.UNICODE)
    return name.strip() or "uploaded_file"


def delete_data_file(path: Path) -> None:
    if path.exists() and path.parent.resolve() == DATA_DIR.resolve():
        path.unlink()
    reset_index()


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_contexts", [])


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_sources(contexts: List[Tuple[Chunk, float]]) -> None:
    if not contexts:
        return

    with st.expander("Các đoạn văn bản được dùng làm căn cứ"):
        for idx, (chunk, score) in enumerate(contexts, start=1):
            st.markdown(f"**Nguồn {idx}: {chunk.source}, trang {chunk.page} — điểm liên quan {score:.3f}**")
            st.write(chunk.text)


def render_sidebar() -> int:
    with st.sidebar:
        st.header("Cấu hình")

        api_key = get_api_key()
        if api_key:
            st.success("Đã nhận GEMINI_API_KEY.")
        else:
            st.error("Chưa có GEMINI_API_KEY. Hãy thêm trong Streamlit Secrets.")

        top_k = st.slider(
            "Số đoạn truy xuất",
            min_value=3,
            max_value=8,
            value=DEFAULT_TOP_K,
        )

        st.write(f"Model trả lời: `{GEN_MODEL}`")
        st.write(f"Model embedding: `{EMBED_MODEL}`")
        st.write(f"Ngưỡng căn cứ: `{MIN_SCORE}`")

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

        uploaded_files = st.file_uploader(
            "Thêm file PDF/TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            saved = 0
            for uploaded_file in uploaded_files:
                save_name = safe_upload_name(uploaded_file.name)
                save_path = DATA_DIR / save_name
                save_path.write_bytes(uploaded_file.getbuffer())
                saved += 1

            reset_index()
            st.success(f"Đã lưu {saved} file. Hệ thống sẽ tự tạo lại index.")
            st.rerun()

        st.divider()

        if st.button("Xóa lịch sử chat"):
            st.session_state.messages = []
            st.session_state.last_contexts = []
            st.rerun()

        if st.button("Xóa index RAG"):
            reset_index()
            st.success("Đã xóa index RAG. Hệ thống sẽ tự tạo lại khi cần.")

    return top_k


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
    ensure_dirs()
    init_session_state()

    st.title("⚖️ Chat hỏi đáp pháp luật về khám chữa bệnh")
    st.caption("RAG + Streamlit + Gemini | Tự động tạo index | Trả lời dựa trên tài liệu đã nạp")

    top_k = render_sidebar()

    api_key = get_api_key()
    client = get_client(api_key)

    if client is None:
        st.warning("Thiếu GEMINI_API_KEY nên chưa thể tạo index hoặc trả lời.")
        st.stop()

    try:
        ready = ensure_index_ready(client)
    except Exception as exc:
        st.error(friendly_error_answer(exc))
        st.stop()

    if not ready:
        st.warning("Hãy thêm ít nhất một file PDF/TXT vào thư mục data/ hoặc upload ở thanh bên.")
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
            contexts: List[Tuple[Chunk, float]] = []

            try:
                with st.spinner("Đang truy xuất văn bản luật liên quan..."):
                    contexts = retrieve(client, question, top_k=top_k)

                with st.spinner("Gemini đang tạo câu trả lời dựa trên văn bản luật..."):
                    answer = answer_question(client, question, contexts)

            except Exception as exc:
                st.exception(exc)
                answer = friendly_error_answer(exc)

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

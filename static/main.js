document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const apiKeyInput = document.getElementById("api-key-input");
    const saveKeyBtn = document.getElementById("save-key-btn");
    const apiStatus = document.getElementById("api-status");
    const docList = document.getElementById("doc-list");
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const reindexBtn = document.getElementById("reindex-btn");
    const chatHistory = document.getElementById("chat-history");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const welcomeScreen = document.getElementById("welcome-screen");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const toggleSidebarBtn = document.getElementById("toggle-sidebar");
    const sidebar = document.querySelector(".sidebar");
    const suggestionsContainer = document.getElementById("suggestions-container");

    let isStreaming = false;

    // Toast Notifications
    function showToast(message, type = "success") {
        const container = document.getElementById("toast-container");
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let icon = "fa-circle-check";
        if (type === "error") icon = "fa-circle-xmark";
        if (type === "warning") icon = "fa-triangle-exclamation";
        
        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add("fade-out");
            toast.addEventListener("animationend", () => toast.remove());
        }, 4000);
    }

    // Toggle Sidebar on mobile
    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener("click", () => {
            sidebar.classList.toggle("collapsed");
        });
    }

    // API Key settings
    async function checkApiKeyStatus() {
        try {
            const res = await fetch("/api/settings/apikey");
            const data = await res.json();
            
            const dot = apiStatus.querySelector(".status-dot");
            const text = apiStatus.querySelector(".status-text");
            
            if (data.configured) {
                dot.className = "status-dot dot-active";
                text.innerText = `Đã cấu hình (${data.preview})`;
                apiKeyInput.value = "";
                document.getElementById("db-status-text").innerText = "Sẵn sàng tra cứu";
            } else {
                dot.className = "status-dot dot-inactive";
                text.innerText = "Chưa cấu hình API Key";
                document.getElementById("db-status-text").innerText = "Chưa thiết lập API Key";
                showToast("Vui lòng cấu hình Gemini API Key để bắt đầu sử dụng Chat RAG!", "warning");
            }
        } catch (err) {
            console.error("Error checking API status:", err);
        }
    }

    saveKeyBtn.addEventListener("click", async () => {
        const key = apiKeyInput.value.trim();
        if (!key) {
            showToast("Vui lòng nhập khóa API hợp lệ", "error");
            return;
        }

        try {
            saveKeyBtn.disabled = true;
            saveKeyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
            
            const res = await fetch("/api/settings/apikey", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: key })
            });
            
            const data = await res.json();
            
            if (res.ok) {
                showToast("Cấu hình API Key thành công!", "success");
                await checkApiKeyStatus();
                // Refresh docs to reflect indexed status if it changed
                loadDocuments();
            } else {
                showToast(data.detail || "Không thể cấu hình API Key", "error");
            }
        } catch (err) {
            showToast("Lỗi kết nối đến máy chủ", "error");
            console.error(err);
        } finally {
            saveKeyBtn.disabled = false;
            saveKeyBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i>';
        }
    });

    // Documents Management
    async function loadDocuments() {
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            
            if (data.length === 0) {
                docList.innerHTML = `
                    <div class="loading-docs" style="font-size: 11px;">
                        <i class="fa-solid fa-box-open" style="font-size: 18px; display: block; margin-bottom: 6px;"></i>
                        Chưa có tài liệu nào.
                    </div>`;
                return;
            }
            
            docList.innerHTML = "";
            data.forEach(doc => {
                const item = document.createElement("div");
                item.className = "doc-item";
                
                const sizeKb = (doc.size / 1024).toFixed(1);
                const isPdf = doc.name.toLowerCase().endsWith(".pdf");
                const iconClass = isPdf ? "fa-file-pdf" : "fa-file-lines";
                
                const badgeText = doc.indexed ? `${doc.chunks} đoạn` : "Chưa lập chỉ mục";
                const badgeClass = doc.indexed ? "doc-badge" : "doc-badge unindexed";
                
                item.innerHTML = `
                    <i class="fa-solid ${iconClass} doc-icon"></i>
                    <div class="doc-info">
                        <div class="doc-name" title="${doc.name}">${doc.name}</div>
                        <div class="doc-meta">
                            <span>${sizeKb} KB</span>
                            <span class="${badgeClass}">${badgeText}</span>
                        </div>
                    </div>
                    <button class="doc-delete-btn" data-name="${doc.name}" title="Xóa tài liệu">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                `;
                
                docList.appendChild(item);
            });

            // Bind delete events
            document.querySelectorAll(".doc-delete-btn").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const name = btn.getAttribute("data-name");
                    if (confirm(`Bạn có chắc chắn muốn xóa tài liệu: ${name}?`)) {
                        await deleteDocument(name);
                    }
                });
            });
        } catch (err) {
            console.error("Error loading documents:", err);
            docList.innerHTML = '<div class="loading-docs text-error"><i class="fa-solid fa-circle-exclamation"></i> Lỗi tải danh sách</div>';
        }
    }

    async function deleteDocument(name) {
        try {
            const res = await fetch(`/api/documents/${name}`, { method: "DELETE" });
            const data = await res.json();
            
            if (res.ok) {
                showToast(`Đã xóa tài liệu ${name}`, "success");
                loadDocuments();
            } else {
                showToast(data.detail || "Không thể xóa tài liệu", "error");
            }
        } catch (err) {
            showToast("Lỗi kết nối máy chủ", "error");
        }
    }

    // Trigger full reindexing
    reindexBtn.addEventListener("click", async () => {
        try {
            reindexBtn.disabled = true;
            reindexBtn.querySelector("i").className = "fa-solid fa-rotate fa-spin";
            
            const res = await fetch("/api/documents/reindex", { method: "POST" });
            const data = await res.json();
            
            if (res.ok) {
                showToast("Đang tiến hành lập chỉ mục trong nền...", "success");
                // Poll document status every 3 seconds to see chunk updates
                let pollCount = 0;
                const interval = setInterval(async () => {
                    await loadDocuments();
                    pollCount++;
                    if (pollCount >= 10) clearInterval(interval); // stop polling after 30s
                }, 3000);
            } else {
                showToast(data.detail || "Không thể chạy lập chỉ mục", "error");
            }
        } catch (err) {
            showToast("Lỗi kết nối máy chủ", "error");
        } finally {
            setTimeout(() => {
                reindexBtn.disabled = false;
                reindexBtn.querySelector("i").className = "fa-solid fa-rotate";
            }, 3000);
        }
    });

    // File upload triggers
    uploadZone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", async () => {
        if (fileInput.files.length > 0) {
            await uploadFile(fileInput.files[0]);
            fileInput.value = ""; // Clear file picker
        }
    });

    // Drag and drop upload zone
    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.style.borderColor = "var(--accent-cyan)";
            uploadZone.style.background = "rgba(6, 182, 212, 0.05)";
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.style.borderColor = "rgba(99, 102, 241, 0.25)";
            uploadZone.style.background = "none";
        }, false);
    });

    uploadZone.addEventListener("drop", async (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            await uploadFile(files[0]);
        }
    });

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            showToast(`Đang tải lên: ${file.name}...`, "warning");
            
            const res = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                if (data.status === "success") {
                    showToast(data.message, "success");
                } else {
                    showToast(data.message, "warning");
                }
                loadDocuments();
            } else {
                showToast(data.detail || "Tải lên thất bại", "error");
            }
        } catch (err) {
            showToast("Lỗi kết nối máy chủ khi tải lên", "error");
            console.error(err);
        }
    }

    // Clear Chat History
    clearChatBtn.addEventListener("click", () => {
        if (confirm("Bạn có chắc muốn xóa toàn bộ lịch sử trò chuyện này?")) {
            // Remove all bubbles except welcome screen
            const messages = chatHistory.querySelectorAll(".message");
            messages.forEach(m => m.remove());
            welcomeScreen.style.display = "flex";
        }
    });

    // Auto-resize textarea input height
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = (chatInput.scrollHeight - 4) + "px";
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Chat suggestions selection
    suggestionsContainer.addEventListener("click", (e) => {
        const chip = e.target.closest(".suggestion-chip");
        if (chip) {
            chatInput.value = chip.innerText;
            chatInput.dispatchEvent(new Event("input"));
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Send question & stream response
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const query = chatInput.value.trim();
        if (!query || isStreaming) return;
        
        // Clear input field
        chatInput.value = "";
        chatInput.style.height = "auto";
        
        // Hide welcome screen
        welcomeScreen.style.display = "none";
        
        // Append User bubble
        appendMessage(query, "user");
        
        // Append AI Bubble placeholder with loader
        const aiBubbleId = `ai-msg-${Date.now()}`;
        const aiBubbleElement = appendMessage("", "ai", aiBubbleId);
        
        isStreaming = true;
        let sources = [];
        let fullAnswerText = "";
        
        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Lỗi khi xử lý câu hỏi");
            }
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            
            let buffer = "";
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                const lines = buffer.split("\n\n");
                // Save last incomplete line back to buffer
                buffer = lines.pop();
                
                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const jsonStr = line.slice(6);
                        try {
                            const data = JSON.parse(jsonStr);
                            
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            
                            if (data.sources) {
                                sources = data.sources;
                            } else if (data.text) {
                                fullAnswerText += data.text;
                                updateAIBubble(aiBubbleElement, fullAnswerText);
                            } else if (data.done) {
                                // Done streaming, finalize block and render sources
                                renderCitations(aiBubbleElement, sources);
                            }
                        } catch (parseErr) {
                            console.error("Error parsing stream chunk:", parseErr);
                        }
                    }
                }
            }
            
        } catch (err) {
            console.error("Chat error:", err);
            showToast(err.message, "error");
            updateAIBubble(aiBubbleElement, `<span style="color:#ef4444;"><i class="fa-solid fa-circle-exclamation"></i> Lỗi: ${err.message}</span>`);
        } finally {
            isStreaming = false;
        }
    });

    function appendMessage(text, sender, id = null) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message message-${sender}`;
        if (id) messageDiv.id = id;
        
        const avatarIcon = sender === "user" ? "fa-user" : "fa-scale-balanced";
        const avatarLabel = sender === "user" ? "U" : "AI";
        
        const textFormatted = sender === "user" ? escapeHtml(text) : renderMarkdown(text);
        
        messageDiv.innerHTML = `
            <div class="message-avatar" title="${sender === "user" ? "Bạn" : "Trợ lý luật y tế"}">
                <i class="fa-solid ${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    ${sender === "ai" && !text ? '<div class="typing-dots"><span></span><span></span><span></span></div>' : textFormatted}
                </div>
            </div>
        `;
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return messageDiv;
    }

    function updateAIBubble(messageDiv, text) {
        const bubble = messageDiv.querySelector(".message-bubble");
        bubble.innerHTML = renderMarkdown(text);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Mini Markdown-like renderer
    function renderMarkdown(text) {
        if (!text) return "";
        let html = text;
        
        // Escape HTML tags to prevent XSS except the ones we explicitly create
        html = escapeHtml(html);
        
        // Bold formatting
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        
        // Bullet points
        // Match line starting with * or - and space
        html = html.replace(/(?:^|\n)[\*\-]\s+(.*?)(?=\n|$)/g, "\n<li>$1</li>");
        // Wrap adjacent li tags in ul
        html = html.replace(/(<li>.*?<\/li>)+/gs, (match) => `<ul>${match}</ul>`);
        
        // Paragraph newlines (preserve paragraphs)
        html = html.split("\n\n").map(p => {
            p = p.trim();
            if (!p) return "";
            if (p.startsWith("<ul") || p.startsWith("<li")) return p;
            return `<p>${p.replace(/\n/g, "<br>")}</p>`;
        }).join("");
        
        return html;
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function renderCitations(messageDiv, sources) {
        if (!sources || sources.length === 0) return;
        
        const contentDiv = messageDiv.querySelector(".message-content");
        const citationsDiv = document.createElement("div");
        citationsDiv.className = "citations";
        
        const uniqueId = `citations-list-${Date.now()}`;
        
        // Generate list items
        let listItemsHtml = "";
        sources.forEach((src, idx) => {
            listItemsHtml += `
                <div class="citation-item">
                    <div class="citation-title">
                        <span>Nguồn ${idx+1}: ${src.title}</span>
                        <span class="citation-score">Độ trùng khớp: ${(src.score * 100).toFixed(0)}%</span>
                    </div>
                    <div class="citation-snippet">${escapeHtml(src.text)}</div>
                </div>
            `;
        });
        
        citationsDiv.innerHTML = `
            <div class="citation-header" id="header-${uniqueId}">
                <i class="fa-solid fa-chevron-right"></i>
                <span>Xem nguồn tham khảo (${sources.length} trích lục luật)</span>
            </div>
            <div class="citation-list" id="${uniqueId}" style="display: none;">
                ${listItemsHtml}
            </div>
        `;
        
        contentDiv.appendChild(citationsDiv);
        
        // Bind collapse event
        const header = document.getElementById(`header-${uniqueId}`);
        const list = document.getElementById(uniqueId);
        
        header.addEventListener("click", () => {
            const isOpen = list.style.display !== "none";
            if (isOpen) {
                list.style.display = "none";
                header.classList.remove("open");
            } else {
                list.style.display = "flex";
                header.classList.add("open");
                // Scroll down a bit to show sources
                setTimeout(() => {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }, 50);
            }
        });
        
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Initialize Page
    checkApiKeyStatus();
    loadDocuments();
});

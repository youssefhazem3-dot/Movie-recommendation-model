const chatWindow = document.querySelector("#chatWindow");
const chatForm = document.querySelector("#chatForm");
const feedbackForm = document.querySelector("#feedbackForm");
const typingIndicator = document.querySelector("#typingIndicator");
const feedbackPanel = document.querySelector("#feedbackPanel");
const feedbackCloseBtn = document.querySelector("#feedbackCloseBtn");
const openFeedbackBtn = document.querySelector("#openFeedbackBtn");
const pendingReviewBar = document.querySelector("#pendingReviewBar");
const pendingReviewText = document.querySelector("#pendingReviewText");
const pendingMovieTitle = document.querySelector("#pendingMovieTitle");
const pendingMovieRating = document.querySelector("#pendingMovieRating");
const feedbackMovieId = document.querySelector("#feedbackMovieId");
const feedbackMovieTitle = document.querySelector("#feedbackMovieTitle");

function scrollChatToBottom() {
    if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function appendMessage(role, message) {
    if (!chatWindow || !typingIndicator) {
        return;
    }

    const messageEl = document.createElement("div");
    messageEl.className = `chat-message ${role}`;
    messageEl.innerHTML = `
        <span>${role === "user" ? "User" : "Model"}</span>
        <p>${escapeHtml(message)}</p>
    `;

    chatWindow.insertBefore(messageEl, typingIndicator);
    scrollChatToBottom();
}

function setTyping(isTyping) {
    if (!typingIndicator) {
        return;
    }

    typingIndicator.classList.toggle("is-hidden", !isTyping);
    scrollChatToBottom();
}

function setButtonLoading(form, isLoading, loadingText, defaultText) {
    const button = form?.querySelector("button");
    if (!button) {
        return;
    }

    button.disabled = isLoading;
    button.textContent = isLoading ? loadingText : defaultText;
}

function showFeedbackPanel(movie) {
    if (!feedbackPanel || !movie) {
        return;
    }

    feedbackPanel.classList.remove("is-hidden");
    feedbackPanel.setAttribute("aria-hidden", "false");
    showPendingReviewBar(movie);
    pendingMovieTitle.textContent = movie.title;
    pendingMovieRating.textContent = `Predicted rating: ${Number(movie.predicted_rating).toFixed(2)}/5`;
    feedbackMovieId.value = movie.movie_id;
    feedbackMovieTitle.value = movie.title;
}

function hideFeedbackPanel() {
    if (!feedbackPanel) {
        return;
    }

    feedbackPanel.classList.add("is-hidden");
    feedbackPanel.setAttribute("aria-hidden", "true");
}

function showPendingReviewBar(movie) {
    if (!pendingReviewBar || !pendingReviewText || !movie) {
        return;
    }

    pendingReviewText.textContent = `Ready to review ${movie.title}?`;
    pendingReviewBar.classList.remove("is-hidden");
}

function hidePendingReviewBar() {
    if (pendingReviewBar) {
        pendingReviewBar.classList.add("is-hidden");
    }
}

if (chatWindow) {
    scrollChatToBottom();
}

if (chatForm) {
    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const textarea = chatForm.querySelector("textarea[name='message']");
        const message = textarea.value.trim();

        if (!message) {
            return;
        }

        appendMessage("user", message);
        textarea.value = "";
        setTyping(true);
        setButtonLoading(chatForm, true, "Sending", "Send");

        try {
            const response = await fetch(chatForm.dataset.apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            setTyping(false);
            appendMessage("assistant", data.assistant_message || data.error);

            if (data.ok && data.movie) {
                showFeedbackPanel(data.movie);
            }
        } catch (error) {
            setTyping(false);
            appendMessage("assistant", "Something went wrong while I was preparing the recommendation.");
        } finally {
            setButtonLoading(chatForm, false, "Sending", "Send");
        }
    });
}

if (feedbackForm) {
    feedbackForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const formData = new FormData(feedbackForm);
        const payload = Object.fromEntries(formData.entries());

        setTyping(true);
        setButtonLoading(feedbackForm, true, "Saving", "Submit Feedback");

        try {
            const response = await fetch(feedbackForm.dataset.apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            setTyping(false);
            if (data.ok) {
                appendMessage("user", data.user_message);
                appendMessage("assistant", data.assistant_message);
                feedbackForm.reset();
                hideFeedbackPanel();
                hidePendingReviewBar();
            } else {
                appendMessage("assistant", data.error || "I could not save that feedback.");
            }
        } catch (error) {
            setTyping(false);
            appendMessage("assistant", "Something went wrong while saving your feedback.");
        } finally {
            setButtonLoading(feedbackForm, false, "Saving", "Submit Feedback");
        }
    });
}

if (feedbackCloseBtn) {
    feedbackCloseBtn.addEventListener("click", hideFeedbackPanel);
}

if (openFeedbackBtn) {
    openFeedbackBtn.addEventListener("click", () => {
        if (!feedbackPanel) {
            return;
        }

        feedbackPanel.classList.remove("is-hidden");
        feedbackPanel.setAttribute("aria-hidden", "false");
    });
}

if (feedbackPanel) {
    feedbackPanel.addEventListener("click", (event) => {
        if (event.target === feedbackPanel) {
            hideFeedbackPanel();
        }
    });
}

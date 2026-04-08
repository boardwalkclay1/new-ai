// frontend/app.js
const API_BASE = "http://localhost:8000";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

let history = [];

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = content;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMessage("user", text);
  history.push({ role: "user", content: text });

  sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history })
    });
    if (!res.ok) {
      addMessage("assistant", `Error: ${res.status} ${res.statusText}`);
    } else {
      const data = await res.json();
      const reply = data.reply || "";
      addMessage("assistant", reply);
      history.push({ role: "assistant", content: reply });
    }
  } catch (e) {
    addMessage("assistant", "Network error talking to backend.");
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

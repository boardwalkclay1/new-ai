// frontend/app.js
const API_BASE = "http://localhost:8000";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

const stackSelect = document.getElementById("stack-select");
const appNameInput = document.getElementById("app-name");
const scaffoldBtn = document.getElementById("scaffold-btn");

let history = [];

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "assistant");
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
  scaffoldBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history })
    });
    const data = await res.json();
    const reply = data.reply || "";
    addMessage("assistant", reply);
    history.push({ role: "assistant", content: reply });
  } catch (e) {
    addMessage("assistant", "Error talking to backend.");
  } finally {
    sendBtn.disabled = false;
    scaffoldBtn.disabled = false;
  }
}

async function scaffoldApp() {
  const stack = stackSelect.value;
  const name = appNameInput.value.trim() || "boardwalk-app";

  addMessage("user", `Scaffold a ${stack} app named ${name}.`);

  scaffoldBtn.disabled = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/scaffold`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stack, name })
    });
    const data = await res.json();
    const msg = data.message || JSON.stringify(data);
    addMessage("assistant", msg);
  } catch (e) {
    addMessage("assistant", "Error scaffolding app.");
  } finally {
    scaffoldBtn.disabled = false;
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
scaffoldBtn.addEventListener("click", scaffoldApp);

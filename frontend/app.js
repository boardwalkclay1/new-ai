// frontend/app.js
const API_BASE = "http://localhost:8000";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

const stackSelect = document.getElementById("stack-select");
const appNameInput = document.getElementById("app-name");
const scaffoldBtn = document.getElementById("scaffold-btn");

const dbNameInput = document.getElementById("db-name");
const dbSchemaInput = document.getElementById("db-schema");
const dbBtn = document.getElementById("db-btn");

const r2NameInput = document.getElementById("r2-name");
const r2Btn = document.getElementById("r2-btn");

const apiProjectInput = document.getElementById("api-project");
const apiSpecInput = document.getElementById("api-spec");
const apiBtn = document.getElementById("api-btn");

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "assistant");
  div.textContent = content;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function scaffoldApp() {
  const stack = stackSelect.value;
  const name = appNameInput.value.trim() || "boardwalk-app";

  scaffoldBtn.disabled = true;
  addMessage("user", `Scaffold ${stack} app: ${name}`);

  try {
    const res = await fetch(`${API_BASE}/api/scaffold`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stack, name })
    });
    const data = await res.json();
    addMessage("assistant", data.message + " @ " + data.path);
  } catch (e) {
    addMessage("assistant", "Error scaffolding app.");
  } finally {
    scaffoldBtn.disabled = false;
  }
}

async function createDbSchema() {
  const name = dbNameInput.value.trim() || "appdb";
  const schema = dbSchemaInput.value.trim();
  if (!schema) {
    addMessage("assistant", "DB schema is empty.");
    return;
  }

  dbBtn.disabled = true;
  addMessage("user", `Create DB schema: ${name}`);

  try {
    const res = await fetch(`${API_BASE}/api/db/schema`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, schema })
    });
    const data = await res.json();
    addMessage("assistant", data.message + " @ " + data.path);
  } catch (e) {
    addMessage("assistant", "Error creating DB schema.");
  } finally {
    dbBtn.disabled = false;
  }
}

async function createR2Config() {
  const bucket_name = r2NameInput.value.trim() || "boardwalk-bucket";

  r2Btn.disabled = true;
  addMessage("user", `Create R2 config: ${bucket_name}`);

  try {
    const res = await fetch(`${API_BASE}/api/cloudflare/r2`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bucket_name })
    });
    const data = await res.json();
    addMessage("assistant", data.message + " @ " + data.path);
  } catch (e) {
    addMessage("assistant", "Error creating R2 config.");
  } finally {
    r2Btn.disabled = false;
  }
}

async function generateApi() {
  const project = apiProjectInput.value.trim() || "app";
  const spec = apiSpecInput.value.trim();
  if (!spec) {
    addMessage("assistant", "API spec is empty.");
    return;
  }

  apiBtn.disabled = true;
  addMessage("user", `Generate API backend for: ${project}`);

  try {
    const res = await fetch(`${API_BASE}/api/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project, spec })
    });
    const data = await res.json();
    addMessage("assistant", data.message + " @ " + data.path);
  } catch (e) {
    addMessage("assistant", "Error generating API backend.");
  } finally {
    apiBtn.disabled = false;
  }
}

function logNote() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMessage("user", text);
  addMessage("assistant", "Note logged. (LLM wiring can be added later.)");
}

sendBtn.addEventListener("click", logNote);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    logNote();
  }
});

scaffoldBtn.addEventListener("click", scaffoldApp);
dbBtn.addEventListener("click", createDbSchema);
r2Btn.addEventListener("click", createR2Config);
apiBtn.addEventListener("click", generateApi);

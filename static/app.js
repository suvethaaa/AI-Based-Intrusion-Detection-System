const form = document.querySelector("#upload-form");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const counts = document.querySelector("#counts");
const preview = document.querySelector("#preview");
const startLive = document.querySelector("#start-live");
const startDemo = document.querySelector("#start-demo");
const stopLive = document.querySelector("#stop-live");
const liveMessage = document.querySelector("#live-message");
const liveState = document.querySelector("#live-state");
const liveMode = document.querySelector("#live-mode");
const liveTotal = document.querySelector("#live-total");
const liveNormal = document.querySelector("#live-normal");
const liveAttack = document.querySelector("#live-attack");
const liveEvents = document.querySelector("#live-events");
const interfaceInput = document.querySelector("#interface");

function numberCell(value) {
  return Number(value).toFixed(5);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "Analyzing traffic...";
  results.hidden = true;

  const formData = new FormData(form);
  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    message.textContent = payload.error || "Prediction failed.";
    return;
  }

  message.textContent = `${payload.total} records processed.`;
  counts.textContent = `Normal: ${payload.counts.normal || 0} | Attack: ${payload.counts.attack || 0}`;
  preview.innerHTML = "";

  payload.preview.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.label}</td>
      <td>${numberCell(row.rf_attack_probability)}</td>
      <td>${numberCell(row.vae_reconstruction_error)}</td>
      <td>${numberCell(row.hybrid_score)}</td>
      <td><span class="badge ${row.prediction}">${row.prediction}</span></td>
    `;
    preview.appendChild(tr);
  });

  results.hidden = false;
});

async function refreshLiveStatus() {
  const response = await fetch("/api/live/status");
  const payload = await response.json();
  const status = payload.status;
  liveState.textContent = status.running ? `Running on ${status.interface}` : "Stopped";
  liveMode.textContent = `Mode: ${status.mode}`;
  liveTotal.textContent = `Total: ${status.total}`;
  liveNormal.textContent = `Normal: ${status.normal}`;
  liveAttack.textContent = `Attack: ${status.attack}`;
  liveMessage.textContent = status.error || liveMessage.textContent;
  liveEvents.innerHTML = "";

  payload.events.forEach((event) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${event.time}</td>
      <td>${event.source}</td>
      <td>${event.destination}</td>
      <td>${event.protocol}</td>
      <td>${numberCell(event.hybrid_score)}</td>
      <td><span class="badge ${event.prediction}">${event.prediction}</span></td>
      <td><button type="button" class="mini" data-ip="${event.source}">Command</button></td>
    `;
    liveEvents.appendChild(tr);
  });
}

async function startMonitor(mode) {
  liveMessage.textContent = mode === "demo" ? "Starting demo stream..." : "Starting live capture...";
  const response = await fetch("/api/live/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ interface: interfaceInput.value.trim() || null, mode }),
  });
  const payload = await response.json();
  liveMessage.textContent = payload.error || payload.message;
  refreshLiveStatus();
}

startLive.addEventListener("click", () => startMonitor("live"));
startDemo.addEventListener("click", () => startMonitor("demo"));

stopLive.addEventListener("click", async () => {
  const response = await fetch("/api/live/stop", { method: "POST" });
  const payload = await response.json();
  liveMessage.textContent = payload.message;
  refreshLiveStatus();
});

liveEvents.addEventListener("click", async (event) => {
  if (!event.target.matches("button[data-ip]")) {
    return;
  }
  const response = await fetch("/api/block-command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ip: event.target.dataset.ip }),
  });
  const payload = await response.json();
  liveMessage.textContent = payload.command || payload.error;
});

setInterval(refreshLiveStatus, 2000);
refreshLiveStatus();

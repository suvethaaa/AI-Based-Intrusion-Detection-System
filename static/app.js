const form = document.querySelector("#upload-form");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const counts = document.querySelector("#counts");
const preview = document.querySelector("#preview");

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

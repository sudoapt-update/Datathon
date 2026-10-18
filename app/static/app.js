const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("file").files[0];
  const apiKey = document.getElementById("api-key").value.trim();
  if (!file) return;

  const body = new FormData();
  body.append("file", file);
  statusEl.textContent = "Загрузка…";
  resultEl.classList.add("hidden");

  const start = await fetch("/api/v1/analyze", {
    method: "POST",
    headers: { "X-API-Key": apiKey },
    body,
  });
  if (!start.ok) {
    statusEl.textContent = `Ошибка загрузки: ${start.status}`;
    return;
  }
  const { job_id } = await start.json();
  statusEl.textContent = `Обработка job ${job_id}…`;
  await poll(job_id, apiKey);
});

async function poll(jobId, apiKey) {
  const deadline = Date.now() + 5 * 60 * 1000;
  while (Date.now() < deadline) {
    const res = await fetch(`/api/v1/jobs/${jobId}`, {
      headers: { "X-API-Key": apiKey },
    });
    const job = await res.json();
    if (job.status === "processing") {
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (job.status === "failed") {
      statusEl.textContent = `Сбой: ${job.error || "неизвестная ошибка"}`;
      return;
    }
    render(job);
    return;
  }
  statusEl.textContent = "Превышен лимит ожидания 5 минут";
}

function render(job) {
  statusEl.textContent = "Готово";
  resultEl.classList.remove("hidden");
  const ok = job.verdict === "no_violation";
  const title = ok
    ? "Нарушение не обнаружено"
    : "Обнаружены признаки возможных нарушений";
  const items = (job.violations || [])
    .map((v) => {
      const frames = (v.evidence_frames || [])
        .map(
          (f) =>
            `<img alt="кадр-доказательство" src="data:image/jpeg;base64,${f.image_base64}" />`
        )
        .join("");
      const rule = [v.rule_document, v.rule_section].filter(Boolean).join(", ");
      return `<article class="violation">
        <h3>${escapeHtml(v.type)}</h3>
        <p class="meta">Таймкод: ${v.timecode_sec} с${rule ? ` · ${escapeHtml(rule)}` : ""}</p>
        <p>${escapeHtml(v.description)}</p>
        ${frames}
      </article>`;
    })
    .join("");
  resultEl.innerHTML = `<p class="verdict ${ok ? "ok" : "bad"}">${title}</p>${items || "<p>Список событий пуст.</p>"}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

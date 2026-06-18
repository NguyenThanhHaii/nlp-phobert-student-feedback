const healthText = document.getElementById("healthText");
const statusDot = document.querySelector(".status-dot");
const feedbackText = document.getElementById("feedbackText");
const taskSelect = document.getElementById("taskSelect");
const modelSelect = document.getElementById("modelSelect");
const normalizeCheck = document.getElementById("normalizeCheck");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const resultBox = document.getElementById("resultBox");

async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    healthText.textContent = data.status === "ok" ? "Đang chạy" : "Không rõ";
    statusDot.classList.add("ok");
  } catch (err) {
    healthText.textContent = "Không kết nối";
    statusDot.classList.add("fail");
  }
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function renderScores(scores) {
  if (!scores) return "";

  const items = Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .map(([label, score]) => `<li><span>${label}</span><strong>${formatPercent(score)}</strong></li>`)
    .join("");

  return `<ul class="score-list">${items}</ul>`;
}

function renderResult(data) {
  const normalizedBlock = data.normalize
    ? `<h3>Text sau chuẩn hóa</h3><div class="text-block">${escapeHtml(data.normalized_text || data.used_text)}</div>`
    : "";

  const warnings = data.warnings && data.warnings.length
    ? data.warnings.map((item) => `<div class="warning">${escapeHtml(item)}</div>`).join("")
    : "";

  const predictions = data.predictions.map((pred) => `
    <div class="prediction">
      <h3>${pred.task === "sentiment" ? "Sentiment" : "Topic"}</h3>
      <div>
        <span class="badge">${pred.model_type}</span>
        <span class="badge">${pred.model_name}</span>
      </div>
      <div class="label">${escapeHtml(pred.label)}</div>
      <div>Độ tin cậy: <strong>${formatPercent(pred.confidence)}</strong></div>
      ${renderScores(pred.scores)}
    </div>
  `).join("");

  resultBox.classList.remove("muted");
  resultBox.innerHTML = `
    <h3>Text sử dụng để dự đoán</h3>
    <div class="text-block">${escapeHtml(data.used_text)}</div>
    ${normalizedBlock}
    ${predictions}
    ${warnings}
    <p class="muted">${escapeHtml(data.disclaimer)}</p>
  `;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function predict() {
  const text = feedbackText.value.trim();

  if (!text) {
    resultBox.classList.add("muted");
    resultBox.textContent = "Vui lòng nhập phản hồi trước khi dự đoán.";
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Đang dự đoán...";
  resultBox.classList.add("muted");
  resultBox.textContent = "Đang xử lý...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        task: taskSelect.value,
        model: modelSelect.value,
        normalize: normalizeCheck.checked,
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    resultBox.classList.add("muted");
    resultBox.innerHTML = `<div class="warning">Lỗi: ${escapeHtml(err.message)}</div>`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Dự đoán";
  }
}

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    feedbackText.value = button.getAttribute("data-example");
  });
});

clearBtn.addEventListener("click", () => {
  feedbackText.value = "";
  resultBox.classList.add("muted");
  resultBox.textContent = "Chưa có kết quả. Nhập phản hồi và bấm “Dự đoán”.";
});

predictBtn.addEventListener("click", predict);
feedbackText.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    predict();
  }
});

checkHealth();

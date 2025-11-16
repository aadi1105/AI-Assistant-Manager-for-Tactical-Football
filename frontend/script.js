const homeSelect = document.getElementById("home-select");
const awaySelect = document.getElementById("away-select");
const homeImg = document.getElementById("home-img");
const awayImg = document.getElementById("away-img");
const btn = document.getElementById("generate-btn");

function showLoading() {
  const loader = document.createElement("div");
  loader.className = "loading-overlay";
  loader.innerHTML = `
    <div class="loading-spinner"></div>
    <p>Generating Reports...</p>
  `;
  document.body.appendChild(loader);
}

function hideLoading() {
  document.querySelectorAll(".loading-overlay").forEach(l => l.remove());
}

async function generateReport(home, away, teamTab, homeTab, awayTab) {
  try {
    showLoading();

    const res = await fetch(`http://127.0.0.1:5000/pregenerate?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);

    if (!res.ok || res.headers.get("content-type")?.includes("application/json")) {
      const data = await res.json().catch(() => ({}));
      hideLoading();
      showPopup(`<h2>Failed to generate report</h2><p>${data.error || "Unknown error occurred."}</p>`);
      return;
    }

    const blob = await res.blob();
    const pdfUrl = URL.createObjectURL(blob);

    // TEAM REPORT
    teamTab.location.href = pdfUrl;

    // PLAYER REPORTS (open from assets folder)
    const homeFile = `http://127.0.0.1:5000/reports/Scout_Report_${home.charAt(0).toUpperCase() + home.slice(1)}.pdf`;
    const awayFile = `http://127.0.0.1:5000/reports/Scout_Report_${away.charAt(0).toUpperCase() + away.slice(1)}.pdf`;

    homeTab.location.href = homeFile;
    awayTab.location.href = awayFile;

    // ✅ Redirect to index.html after reports are generated
    setTimeout(() => {
      window.location.href = `index.html?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`;
    }, 2000);


  } catch (err) {
    console.error("Error:", err);
    showPopup("<h2>Failed to generate report</h2><p>Could not connect to server.</p>");
  } finally {
    hideLoading();
  }
}

function showPopup(innerHTML) {
  if(innerHTML.includes('blob:')){
        window.open(innerHTML, "_blank");
        return;
  }
  // Remove existing popup if any
  document.querySelectorAll(".popup-overlay").forEach(p => p.remove());

  const overlay = document.createElement("div");
  overlay.className = "popup-overlay";
  overlay.innerHTML = `
    <div class="popup-box">
      <button class="popup-close">×</button>
      <div class="popup-content">${innerHTML}</div>
    </div>
  `;
  document.body.appendChild(overlay);
  document.body.classList.add("popup-active");

  overlay.querySelector(".popup-close").addEventListener("click", () => {
    overlay.remove();
    document.body.classList.remove("popup-active");
  });
}



function updateDropdowns() {
  const home = homeSelect.value;
  const away = awaySelect.value;

  [...awaySelect.options].forEach(opt => {
    opt.disabled = opt.value !== "" && opt.value === home;
  });

  [...homeSelect.options].forEach(opt => {
    opt.disabled = opt.value !== "" && opt.value === away;
  });

  homeImg.classList.toggle("hidden", home === "");
  awayImg.classList.toggle("hidden", away === "");

  if (home) homeImg.src = `assets/badges/${home}.png`;
  if (away) awayImg.src = `assets/badges/${away}.png`;

  btn.classList.toggle("hidden", !(home && away));
}

homeSelect.addEventListener("change", updateDropdowns);
awaySelect.addEventListener("change", updateDropdowns);

btn?.addEventListener("click", () => {
  const home = homeSelect.value;
  const away = awaySelect.value;
  if (!home || !away) return;

  // Pre-open all 3 tabs immediately (bypasses popup blocking)
  const teamTab = window.open("", "_blank");
  const homeTab = window.open("", "_blank");
  const awayTab = window.open("", "_blank");

  // Generate the reports asynchronously
  generateReport(home, away, teamTab, homeTab, awayTab);
});




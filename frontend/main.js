// (Paste over your current main.js — only the summary popup placement was added / fixed.)
// Keep the rest of your original main.js intact. Below is your main.js with a safe global summary popup handler added.

let events = []
let tileData = []
// 1. Get the 'cookie' query parameter from the URL
const urlParams = new URLSearchParams(window.location.search);
const home = urlParams.get('home');
const away = urlParams.get('away');

// ---------- Team badges ----------
const homeTeam = home || '';
const awayTeam = away || '';
const homeBadge = document.getElementById('home-badge');
const awayBadge = document.getElementById('away-badge');
if(homeTeam) homeBadge.src = `assets/badges/${homeTeam}.png`;
if(awayTeam) awayBadge.src = `assets/badges/${awayTeam}.png`;

// ---------- Timer (countdown) ----------
const timerEl = document.getElementById('timer');
const timerBtn = document.getElementById('timer-btn');
let remaining = 90*60; // seconds (countdown from 90 minutes)
let tick = null;

function renderTime(s){
  const m = Math.floor(s/60).toString().padStart(2,'0');
  const sec = (s%60).toString().padStart(2,'0');
  timerEl.textContent = `${m}:${sec}`;
}

timerBtn.addEventListener('click', () => {
  if(tick){
      clearInterval(tick); tick = null;
      timerBtn.textContent = 'Start Timer';
      timerBtn.classList.remove('stop');
      timerBtn.classList.add('start');
      return;
  }
  timerBtn.textContent = 'Stop';
  timerBtn.classList.remove('start');
  timerBtn.classList.add('stop');
  tick = setInterval(() => {
      if(remaining <= 0){ clearInterval(tick); tick=null; return; }
      remaining--; renderTime(remaining);
  }, 1000);
});
renderTime(remaining);

// ---------- Events input + floating ----------
const inputEl = document.getElementById('event-input');
const leftContent = document.querySelector('.left-content');
const eventsList = document.getElementById('events-list');
const eventsPanel = document.getElementById('events-panel');
const eventsToggle = document.getElementById('events-toggle');

function readEvents(){
  try { return events || []; }
  catch { return []; }
}
function writeEvents(arr){ events=arr; }

function refreshEventsList(){
  const arr = readEvents().slice().reverse(); // most recent first
  eventsList.innerHTML = '';
  for(const e of arr){
      const li = document.createElement('li');
      li.textContent = e; li.className='event-item';
      eventsList.appendChild(li);
  }
}

function withinFiveWords(text){
  const words = text.trim().split(' ').filter(Boolean);
  return words.length > 0 && words.length <= 5;
}

function spawnFloating(text){
  const span = document.createElement('span');
  span.className = 'floaty';
  span.textContent = text;
  const box = leftContent.getBoundingClientRect();
  const x = Math.random() * (box.width - 160) + 20; // padding
  const y = Math.random() * (box.height - 160) + 40;
  span.style.left = x + 'px';
  span.style.top = y + 'px';
  leftContent.appendChild(span);
  requestAnimationFrame(() => span.classList.add('rise'));
  setTimeout(() => span.remove(), 1600);
}

inputEl.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter'){
      const currRemaining = document.getElementById('timer').innerText;

      const toSec = t => t.split(":").reduce((m,s)=>+m*60+ +s);
      const from90 = t => 5400 - toSec(t);
      const toStr = s => `${(s/60|0).toString().padStart(2,"0")}:${(s%60).toString().padStart(2,"0")}`;
      const val = inputEl.value.trim()+' - '+toStr(from90(currRemaining));
      if(!withinFiveWords(val)){
        inputEl.classList.add('shake');
        setTimeout(()=>inputEl.classList.remove('shake'), 400);
        return;
      }
      spawnFloating(val);
      const keyDownEvents = readEvents();
      keyDownEvents.push(val);
      writeEvents(keyDownEvents);
      refreshEventsList();
      inputEl.value = '';
  }
});

eventsToggle.addEventListener('click', ()=>{
  eventsPanel.classList.toggle('open');
});

// If a new report was generated, ensure events are reset (already done on welcome)
refreshEventsList();

const tiles = document.querySelectorAll('.tile');

tiles.forEach( (tile, index) => {
  tile.addEventListener('click', () => {
    // Create popup wrapper
    const popup = document.createElement('div');
    popup.className = 'tile-popup';
    popup.innerHTML = `
      <button class="popup-close">X</button>
      <div class="popup-content">${tileData[index]}</div>
      <br><br>
    `;

    document.body.appendChild(popup);
    document.body.classList.add('popup-active');

    // Close button
    popup.querySelector('.popup-close').addEventListener('click', () => {
      popup.remove();
      document.body.classList.remove('popup-active');
    });

    const reinforceBtn = popup.querySelector('.reinforce');
    if (reinforceBtn) {
        reinforceBtn.addEventListener('click', async (e) => {
            const idx = parseInt(e.currentTarget.dataset.idx, 10);
            const matchEvents = readEvents();
            const currRemaining = document.getElementById('timer').innerText;
            const toSec = t => t.split(":").reduce((m,s)=>+m*60+ +s);
            const from90 = t => 5400 - toSec(t);
            const toStr = s => `${(s/60|0).toString().padStart(2,"0")}:${(s%60).toString().padStart(2,"0")}`;

            try {
                const res = await fetch("http://127.0.0.1:5000/reinforce", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    index: idx,
                    matchEvents: matchEvents,
                    timeStamp: toStr(from90(currRemaining))
                })
                });

                const reportData = await res.json();

                let titleOne = document.getElementById('title-0');
                let titleTwo = document.getElementById('title-1');
                let titleThree = document.getElementById('title-2');
                let titleFour = document.getElementById('title-3');
                let titleFive = document.getElementById('title-4');
                let titleSix = document.getElementById('title-5');
                let titleSeven = document.getElementById('title-6');

                titleOne.innerText = reportData[0];
                titleTwo.innerText = reportData[1];
                titleThree.innerText = reportData[2];
                titleFour.innerText = reportData[3];
                titleFive.innerText = reportData[4];
                titleSix.innerText = reportData[5];
                titleSeven.innerText = reportData[6];

            } catch (err) {
                console.error("Reinforce request failed:", err);
            }
        });
    }


  });

});

// index.js - load after DOM ready
document.addEventListener("DOMContentLoaded", async () => {
  // read home/away from query params
  const params = new URLSearchParams(window.location.search);
  const home = params.get("home");
  const away = params.get("away");
  if (!home || !away) {
    console.error("Missing home/away on index.html");
    return;
  }

  // show team badges if you want
  const homeBadge = document.getElementById("home-badge");
  const awayBadge = document.getElementById("away-badge");
  if (homeBadge) homeBadge.src = `assets/badges/${home}.png`;
  if (awayBadge) awayBadge.src = `assets/badges/${away}.png`;

  // ------- FIX 4: correct predictions URL -------
  const predUrl = `http://127.0.0.1:5000/reports/predictions_${home}_${away}.json`;

  let data = null;
  try {
    const res = await fetch(predUrl);
    if (!res.ok) throw new Error("predictions not found");
    data = await res.json();
  } catch (e) {
    console.error("Could not load predictions:", e);
    return;
  }

  const modelKeys = Object.keys(data.models);

  initSimulation(data, modelKeys);

  // Add a call at load to fetch minute 0 predictions
  fetchLivePredictions(0, readEvents(), home, away);
});

function fmtTime(s) {
  const m = Math.floor(s/60).toString().padStart(2,'0');
  const sec = (s%60).toString().padStart(2,'0');
  return `${m}:${sec}`;
}

function initSimulation(data, modelKeys) {
  const slider = document.getElementById("time-slider");
  const timeInput = document.getElementById("time-input"); // may be null if input not present — guarded below
  const timeLabel = document.getElementById("slider-time"); // small elapsed / slider time
  const playBtn = document.getElementById("play-btn");
  const jumpBtn = document.getElementById("jump-goal");
  const bigTimerEl = document.getElementById("timer");

  if (!slider || !timeLabel || !playBtn || !bigTimerEl) {
    console.error("initSimulation: missing required DOM elements (slider/timeLabel/playBtn/bigTimer).");
    return;
  }

  const duration = Number(data?.match_meta?.duration_sec || 90 * 60); // seconds
  slider.max = duration;
  slider.step = Number(data.time_step || 1);

  // Helper to format seconds -> "MM:SS"
  function fmtTimeLocal(s) {
    const m = Math.floor(s / 60).toString().padStart(2, "0");
    const sec = (s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  }

  // When slider moves: update small elapsed label, big remaining timer, sync minute input, and update UI
  slider.addEventListener("input", () => {
    const t = Math.max(0, Math.min(Number(slider.value) || 0, duration)); // seconds elapsed
    timeLabel.textContent = fmtTimeLocal(t); // elapsed shown in small label

    // update the main visible timer to show remaining time
    const remaining = Math.max(0, duration - t);
    bigTimerEl.textContent = fmtTimeLocal(remaining);

    // sync minute input (if present) — show rounded down minutes
    if (timeInput) {
      timeInput.value = Math.floor(t / 60);
    }

    // call app logic to update tiles / charts / text for this timestamp
    try {
      updateUIForTime(t, data, modelKeys);
    } catch (err) {
      console.error("updateUIForTime error:", err);
    }

    // LIVE MODEL UPDATE (AI insights) — send minute and builtin events to backend
    fetchLivePredictions(Math.floor(t / 60), readEvents(), home, away);
  });

  // If user types minutes into the number input, jump the slider to that minute
  if (timeInput) {
    // ensure input min/max are sane
    timeInput.min = 0;
    timeInput.max = Math.floor(duration / 60);

    timeInput.addEventListener("input", () => {
      // clamp and coerce to integer minutes
      let minutes = Number(timeInput.value);
      if (isNaN(minutes)) minutes = 0;
      minutes = Math.max(0, Math.min(Math.floor(minutes), Math.floor(duration / 60)));

      const seconds = minutes * 60;
      slider.value = seconds;
      // trigger the slider handler (keeps everything in sync)
      slider.dispatchEvent(new Event("input"));
    });
  }

  // Play / Pause behaviour (plays through slider forward)
  let playing = false;
  let intervalId = null;
  playBtn.addEventListener("click", () => {
    if (!playing) {
      playing = true;
      playBtn.textContent = "Pause";

      // playback speed: increase seconds per tick. Adjust 250ms interval for speed.
      const playbackTickMs = 300; // shorter -> faster playback
      intervalId = setInterval(() => {
        const cur = Number(slider.value) || 0;
        const step = Number(data.time_step || 1);
        if (cur >= duration) {
          clearInterval(intervalId);
          playing = false;
          playBtn.textContent = "Play";
          return;
        }
        slider.value = Math.min(duration, cur + step);
        slider.dispatchEvent(new Event("input"));
      }, playbackTickMs);
    } else {
      // pause
      playing = false;
      playBtn.textContent = "Play";
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    }
  });

  // Jump to next event (if defined in data.events). Finds first event.time > current slider value.
  const events = Array.isArray(data.events) ? data.events : [];
  jumpBtn.addEventListener("click", () => {
    const cur = Number(slider.value) || 0;
    const next = events.find(e => (Number(e.time) || 0) > cur);
    if (next) {
      // jump slightly before event so UI has context
      const jumpTo = Math.max(0, (Number(next.time) || 0) - 30);
      slider.value = Math.min(duration, jumpTo);
      slider.dispatchEvent(new Event("input"));
    } else {
      // no next event: move to end
      slider.value = duration;
      slider.dispatchEvent(new Event("input"));
    }
  });

  // Initialize values (start at 0)
  slider.value = 0;
  if (timeInput) timeInput.value = 0;
  timeLabel.textContent = fmtTimeLocal(0);
  bigTimerEl.textContent = fmtTimeLocal(duration);
  // trigger initial UI update
  slider.dispatchEvent(new Event("input"));
}

let lastMinute = 0;
function updateUIForTime(t, data, modelKeys) {
  let tidx = 0;
  modelKeys.forEach((key, idx) => {
    const modelObj = data.models[key];
    const values = modelObj.values;
    const step = data.time_step || 1;
    const index = Math.min(values.length - 1, Math.floor(t / step));
    const prob = values[index] || 0;

    const tile = document.getElementById(`tile-${tidx}`);
    tidx+=1;
    if (!tile) return;

    tile.textContent = modelObj.label || key + `${Math.round(prob*100)}%`;

    const threshold = modelObj.threshold || 0.5;
    if (prob >= threshold) {
      tile.classList.add('alert');
      showTileAlert(tile, `${modelObj.label} ${(prob*100).toFixed(0)}%`);
    } else {
      tile.classList.remove('alert');
      clearTileAlert(tile);
    }
  });

  // summary
  const summaryEl = document.getElementById('summary-tile');
  if (summaryEl) {
    let sum = 0, n = 0;
    modelKeys.forEach(k => {
      const v = data.models[k].values[Math.min(Math.floor(t / (data.time_step||1)), data.models[k].values.length - 1)] || 0;
      sum += v; n++;
    });
    const avg = n ? sum / n : 0;
    summaryEl.textContent = `Combined: ${(avg*100).toFixed(0)}%`;
  }
}

function showTileAlert(tile, text) {
  const now = Date.now();
  const id = tile.id;
  if (lastAlertTs[id] && now - lastAlertTs[id] < 1200) return;
  lastAlertTs[id] = now;
  let badge = tile.querySelector('.tile-alert');
  if (!badge) {
    badge = document.createElement('div'); badge.className = 'tile-alert';
    tile.appendChild(badge);
  }
  badge.textContent = text;
  badge.style.opacity = 1;
  setTimeout(()=> badge.style.opacity = 0, 1200);
}

function clearTileAlert(tile) {
  const badge = tile.querySelector('.tile-alert');
  if (badge) badge.style.opacity = 0;
}

async function fetchLivePredictions(minutes, rawEvents = [], homeArg = null, awayArg = null) {
  try {
    // Convert rawEvents (strings or objects) into structured events:
    // - If string contains " - MM:SS" use minutes from time part (MM)
    // - Else, assign current minute (passed in `minutes`)
    const structured = (rawEvents || []).map(ev => {
      if (!ev) return { event: "", minute: Number(minutes) };
      if (typeof ev === "object" && ev.event) {
        // already structured
        return { event: ev.event, minute: Number(ev.minute || minutes) };
      }
      if (typeof ev === "string") {
        // try to split "something - MM:SS"
        const parts = ev.split(" - ");
        if (parts.length > 1) {
          const timepart = parts[parts.length - 1].trim();
          if (timepart.includes(":")) {
            const mm = parseInt(timepart.split(":")[0], 10);
            if (!Number.isNaN(mm)) return { event: parts.slice(0, -1).join(" - ").trim(), minute: mm };
          }
        }
        // no explicit time: use provided current minute
        return { event: ev.trim(), minute: Number(minutes) };
      }
      // fallback
      return { event: String(ev), minute: Number(minutes) };
    });

    const payload = {
      min: Number(minutes),
      events: structured,
      home: homeArg || (new URLSearchParams(window.location.search).get('home') || window.home || null),
      away: awayArg || (new URLSearchParams(window.location.search).get('away') || window.away || null)
    };

    const res = await fetch("http://127.0.0.1:5000/live_predictions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      console.error("live_predictions responded with", res.status);
      return;
    }

    const data = await res.json();
    const models = data.models || [];
    const summary = data.summary || "";

    // Update the six model tiles based on returned paragraphs
    models.forEach((txt, i) => {
      if (i >= 6) return; // first six are model paragraphs
      const tile = document.getElementById(`tile-${i}`);
      if (!tile) return;

      // clear previous contents
      tile.innerHTML = "";

      // detect ALERTALERT prefix (allow whitespace)
      let isAlert = false;
      if (txt.trim().toUpperCase().startsWith("ALERTALERT")) {
        isAlert = true;
        txt = txt.replace(/^(ALERTALERT\s*)/i, "");
      }

      // extract heading (<h5>) and body
      let heading = "";
      let body = txt;
      const h5start = txt.indexOf("<h5>");
      const h5end = txt.indexOf("</h5>");
      if (h5start !== -1 && h5end !== -1 && h5end > h5start) {
        heading = txt.substring(h5start + 4, h5end).trim();
        body = txt.substring(h5end + 5).trim();
      } else {
        heading = "Model";
        body = txt.trim();
      }

      const headingEl = document.createElement("h5");
      headingEl.textContent = heading;
      const p = document.createElement("p");
      p.textContent = body.slice(0, 120) + (body.length > 120 ? "..." : "");

      if (isAlert) {
        tile.classList.add("alert");
        tile.style.boxShadow = "0 4px 0px rgba(235, 0, 0, 0.838)";
        tile.style.backgroundColor = "rgba(235, 0, 0, 0.138)";
      } else {
        tile.classList.remove("alert");
        tile.style.boxShadow = "";
        tile.style.backgroundColor = "";
      }

      tile.appendChild(headingEl);
      tile.appendChild(p);

      // store full paragraph for popup (ensures popup has content)
      tileData[i] = txt;
    });

    // update summary tile
    const summaryDiv = document.getElementById("summary-tile");
    if (summaryDiv) summaryDiv.innerHTML = summary;

  } catch (err) {
    console.error("Failed to fetch live predictions:", err);
  }
}

// --- GLOBAL SUMMARY TILE POPUP HANDLER (safe placement) ---
document.addEventListener("DOMContentLoaded", () => {
    const summaryTile = document.getElementById("summary-tile");
    if (!summaryTile) return;

    summaryTile.addEventListener("click", () => {
        const content = summaryTile.innerHTML;

        const popup = document.createElement("div");
        popup.className = "tile-popup";
        popup.innerHTML = `
            <button class="popup-close">X</button>
            <div class="popup-content">${content}</div>
        `;

        document.body.appendChild(popup);
        document.body.classList.add("popup-active");

        popup.querySelector(".popup-close").addEventListener("click", () => {
            popup.remove();
            document.body.classList.remove("popup-active");
        });
    });
});

// ============= ORIGINAL CODE PRESERVED =============
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, '<br>');
}


// All helper functions, upload, chat, auth, etc. remain EXACTLY as you had them
let currentChatId = null;  // ← ADD THIS LINE

async function sendMessage() {
  console.log("🔥 sendMessage() CALLED!");
  console.log("🔥 mode = ", document.getElementById("retrievalMode").value);

  const question = document.getElementById("userMessage").value;
  const retrievalMode = document.getElementById("retrievalMode").value;

  const payload = {
    messages: [{ role: "user", content: question }],
    retrieval_mode: retrievalMode,
    role: "User"
  };

const token = localStorage.getItem("AUTH_TOKEN");

  const res = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token && { "Authorization": `Bearer ${token}` })
    },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  console.log("Reply:", data.reply);
};

document.addEventListener('mousemove', e => {
  const layers = document.querySelectorAll('.layer');
  const x = (e.clientX / window.innerWidth - 0.5) * 2;
  const y = (e.clientY / window.innerHeight - 0.5) * 2;
  layers.forEach(l => {
    const depth = l.dataset.depth;
    l.style.transform = `translate3d(${x * depth * 30}px, ${y * depth * 30}px, 0)`;
  });
});

// const themeButtons = document.querySelectorAll('.theme-btn');
// themeButtons.forEach(btn => {
//   btn.addEventListener('click', () => {
//     document.body.classList.remove('theme-light', 'theme-dark', 'theme-nebula', 'theme-aurora');
//     document.body.classList.add(`theme-${btn.dataset.theme}`);
//     btn.style.transform = 'scale(1.4)';
//     setTimeout(() => (btn.style.transform = ''), 300);
//   });
// });

// const themeButtons = document.querySelectorAll('.theme-btn');
// const bgVideo = document.getElementById('bg-video');

// themeButtons.forEach(btn => {
//   btn.addEventListener('click', () => {
//     const theme = btn.dataset.theme;

//     // switch theme classes
//     document.body.classList.remove(
//       'theme-light',
//       'theme-dark',
//       'theme-nebula',
//       'theme-aurora'
//     );
//     document.body.classList.add(`theme-${theme}`);

//     // handle bg video for aurora only
//     if (theme === 'aurora') {
//       bgVideo.style.display = 'block';       // or add a class like 'show-video'
//       bgVideo.play();                        // optional
//     } else {
//       bgVideo.pause();                       // optional
//       bgVideo.style.display = 'none';        // or remove class
//     }

//     // button click animation
//     btn.style.transform = 'scale(1.4)';
//     setTimeout(() => (btn.style.transform = ''), 300);
//   });
// });


// ---------- helpers ----------
const $ = (s, p = document) => p.querySelector(s);
const $$ = (s, p = document) => [...p.querySelectorAll(s)];
const API_BASE = () => localStorage.getItem("API_BASE") || "http://localhost:8000";
const pages = ["home", "upload", "chat", "dash", "login", "account", "upgrade"];

// Add this definition to your ne.js file
function apiHeaders(contentType = "application/json") {
  const token = localStorage.getItem("AUTH_TOKEN");
  if (!token) {
    console.warn("No AUTH_TOKEN found – you will get 401");
    return { "Content-Type": contentType };
  }
  return {
    "Content-Type": contentType,
    "Authorization": `Bearer ${token}`   
  };
}

function show(id) {
  pages.forEach(p => $("#" + p).classList.remove("active"));
  $("#" + id).classList.add("active");
  pages.forEach(p => $("#" + p).style.display = (p === id ? 'block' : 'none'));
}

function openSettings() {
  $("#settingsModal").classList.remove("hidden");
  $("#apiBase").value = API_BASE();
}

function closeSettings() {
  $("#settingsModal").classList.add("hidden");
}

function saveSettings() {
  const v = $("#apiBase").value.trim();
  if (v) {
    localStorage.setItem("API_BASE", v);
    closeSettings();
    showToast("✅ API Base saved successfully!");
  }
}

window.show = show;
window.openSettings = openSettings;
window.closeSettings = closeSettings;
window.saveSettings = saveSettings;

// ---------- Upload with progress ----------
(function initUpload() {
  const dz = $("#dropzone");
  const input = $("#fileInput");
  const browse = $("#browseLink");
  const progress = $("#uploadProgress");
  const bar = $("#uploadBar");
  const fill = $("#uploadBarFill");
  const filesUl = $("#fileList");

  function setBar(pct) {
    bar.classList.remove("hidden");
    fill.style.width = pct + "%";
  }
  function clearBar() {
    bar.classList.add("hidden");
    fill.style.width = "0%";
  }

  async function refreshFiles() {
    try {
      const res = await fetch(API_BASE() + "/files", { cache: "no-store" });
      const data = await res.json();
      filesUl.innerHTML = "";
      (data.files || []).forEach(name => {
        const li = document.createElement("li");
        li.className = "flex items-center justify-between gap-2";
        li.innerHTML = `<span class="truncate">${name}</span>`;
        filesUl.appendChild(li);
      });
    } catch { }
  }

  async function uploadFiles(files) {
    if (!files || !files.length) return;
    const form = new FormData();
    [...files].forEach(f => form.append("files", f));
    progress.textContent = "Uploading…";
    setBar(3);
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", API_BASE() + "/upload", true);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.max(5, Math.round((e.loaded / e.total) * 100));
          setBar(pct);
        }
      };
      xhr.onload = () => (xhr.status >= 200 && xhr.status < 300) ? resolve() : reject();
      xhr.onerror = reject;
      xhr.send(form);
    })
      .then(() => {
        progress.textContent = "Uploaded complete!";
        setBar(100);
        setTimeout(() => {
          progress.textContent = "";
          clearBar();
        }, 2500);
        refreshFiles();
        loadHealth();
        showToast("File uploaded successfully!", "success");
      })
      .catch(() => {
        progress.textContent = "Upload failed or server offline.";
        setBar(0);
        setTimeout(() => {
          progress.textContent = "";
          clearBar();
        }, 3000);
      });
  }

  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("bg-white/5"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("bg-white/5"));
  dz.addEventListener("drop", e => { e.preventDefault(); dz.classList.remove("bg-white/5"); uploadFiles(e.dataTransfer.files); });
  browse.addEventListener("click", e => { e.preventDefault(); input.click(); });
  input.addEventListener("change", e => uploadFiles(e.target.files));
  $("#refreshFilesBtn").addEventListener("click", refreshFiles);
  refreshFiles();
})();

// ---------- Chat (markdown + typing + local history) ----------
(function initChat() {
  const input = $("#chatInput");
  const sendBtn = $("#sendBtn");
  const clearBtn = $("#clearChatBtn");
  const retrievalModeSel = document.getElementById("retrievalMode");
  const roled = document.getElementById("roleSel");
  const chatLog = $("#chatLog");


  function bubble(role, html) {
    const row = document.createElement("div");
    row.className = "flex " + (role === "user" ? "justify-end" : "justify-start");
    const b = document.createElement("div");
    b.className = "bubble " + (role === "user" ? "bubble-user" : "bubble-ai") + " msg";
    b.innerHTML = html;
    row.appendChild(b);
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
    return b;
  }

  function md(text) {
    try {
      const unsafe = marked.parse(text || "");
      return DOMPurify.sanitize(unsafe);
    } catch { return DOMPurify.sanitize(text); }
  }

  window.bubble = bubble;
  window.md = md;
  window.chatLog = chatLog;
  window.setTyping = setTyping;
  window.saveLocal = saveLocal;
  window.loadLocal = loadLocal;

  function setTyping(on) {
    if (on) {
      const el = bubble("ai", '<span class="type-cursor">Thinking</span>');
      el.dataset.typing = "1";
      return el;
    } else {
      const last = [...chatLog.querySelectorAll('[data-typing="1"]')].pop();
      if (last) last.remove();
    }
  }

  function saveLocal(role, content) {
    const hist = JSON.parse(localStorage.getItem("NEBULA_CHAT") || "[]");
    hist.push({ role, content, t: Date.now() });
    localStorage.setItem("NEBULA_CHAT", JSON.stringify(hist.slice(-100)));
  }

  function loadLocal() {
    chatLog.innerHTML = "";
    const hist = JSON.parse(localStorage.getItem("NEBULA_CHAT") || "[]");
    hist.forEach(m => bubble(m.role, md(m.content)));
    chatLog.scrollTop = chatLog.scrollHeight;
  }

//   async function send() {
//     const msg = (input.value || "").trim();
//     if (!msg) return;
//     bubble("user", md(msg));
//     saveLocal("user", msg);
//     input.value = "";

//     const typingEl = setTyping(true);
//     try {
//       const payload = {
//         messages: [{ content: msg }],
//         role: (roled?.value || "user"),
//         retrieval_mode: (retrievalModeSel?.value || "hybrid"),
//         chat_id: currentChatId
//       };

//       const r = await fetch(API_BASE() + "/chat", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(payload)
//       });
//       const data = await r.json();

//       setTyping(false);
//       const text = (data.reply || "…");
//       bubble("ai", md(text));
//       saveLocal("assistant", text);

//       if (data.citations && Array.isArray(data.citations) && data.citations.length) {
//         const c = document.createElement("div");
//         c.className = "text-xs text-white/60 mt-1";
//         c.innerHTML = "Sources: " + data.citations.map(s => '<span class="mr-2">• ' + DOMPurify.sanitize(String(s)) + '</span>').join("");
//         chatLog.lastElementChild.appendChild(c);
//         chatLog.scrollTop = chatLog.scrollHeight;
//       }
//     } catch (e) {
//       setTyping(false);
//       bubble("ai", md("Unauthorized. Please log in to continue.😉"));
//     }
//   }

// today now means 1:34 03-01-2025 
// async function send() {
//   const msg = (input.value || "").trim();
//   if (!msg) return;

//   bubble("user", md(msg));
//   saveLocal("user", msg);
//   input.value = "";

//   const typingEl = setTyping(true);

//   try {
//     const payload = {
//       messages: [{ content: msg }],
//       role: (roled?.value || "user"),
//       retrieval_mode: (retrievalModeSel?.value || "hybrid"),
//       chat_id: currentChatId
//     };

//     // THIS IS THE ONLY LINE YOU WERE MISSING
//     const token = localStorage.getItem("AUTH_TOKEN");
//     const headers = {
//       "Content-Type": "application/json",
//       ...(token && { "Authorization": `Bearer ${token}` })
//     };

//     const r = await fetch(API_BASE() + "/chat", {
//       method: "POST",
//       headers: headers,           // ← NOW SENDS JWT!
      
//       body: JSON.stringify(payload)
//     });

//     if (!r.ok) throw new Error("HTTP " + r.status);

//     const data = await r.json();

//     setTyping(false);
//     const text = data.reply || "...";
//     bubble("ai", md(text));
//     // After bubble("ai", ...) and before return
// if (chatLog.children.length <= 2) {  // first real message pair
//   const firstUserMsg = document.querySelector('.bubble-user .msg')?.textContent || "New Chat";
//   const shortTitle = firstUserMsg.substring(0, 32) + (firstUserMsg.length > 32 ? "..." : "");

//   const sidebarItem = document.querySelector(`.chat-item[data-id="${currentChatId}"] .chat-title`);
//   if (sidebarItem) {
//     sidebarItem.textContent = shortTitle;
//   }
// }
//     saveLocal("assistant", text);

//     // Update current chat ID if it was a new chat
//     if (data.chat_id && !currentChatId) {
//   currentChatId = data.chat_id;
//   if (sessionIdEl) sessionIdEl.value = currentChatId;
  
  
//   // ← UPDATE UI
//   loadChatList();  // ← Refresh sidebar to show real chat (not temp)
// }

//     if (data.citations?.length) {
//       // your existing citation code
//     }

//   } catch (e) {
//     console.error(e);
//     setTyping(false);
//     bubble("ai", md("Server error or unauthorized."));
//   }
// }

// async function send() {
//   const msg = (input.value || "").trim();
//   if (!msg) return;

//   bubble("user", md(msg));
//   saveLocal("user", msg);
//   input.value = "";

//   const typingEl = setTyping(true);

//   try {
//     const payload = {
//       messages: [{ content: msg }],
//       role: (roled?.value || "user"),
//       retrieval_mode: (retrievalModeSel?.value || "hybrid"),
//       chat_id: currentChatId
//     };

//     const token = localStorage.getItem("AUTH_TOKEN");
//     const headers = {
//       "Content-Type": "application/json",
//       ...(token && { "Authorization": `Bearer ${token}` })
//     };

//     const r = await fetch(API_BASE() + "/chat", {
//       method: "POST",
//       headers: headers,
//       body: JSON.stringify(payload)
//     });

//     if (!r.ok) throw new Error("HTTP " + r.status);

//     const data = await r.json();

//     setTyping(false);
//     const text = data.reply || "...";
//     const aiBubble = bubble("ai", md(text));
//     saveLocal("assistant", text);

//     // === THIS IS THE NEW PART: SHOW SOURCES BEAUTIFULLY ===
//     if (data.sources && data.sources.length > 0) {
//       let sourcesHTML = '<div class="sources-container">';
//       data.sources.forEach((src, i) => {
//         const rank = i + 1;
//         const pageInfo = src.page !== null && src.page !== undefined
//           ? `→ page ${src.page}`
//           : src.source.match(/\.(png|jpe?g|webp)$/i)
//             ? '→ (OCR from screenshot)'
//             : '';

//         sourcesHTML += `
//           <div class="source-block">
//             <div class="source-header">
//               <span class="source-rank">#${rank}</span>
//               <span class="source-name">${escapeHtml(src.source)}</span>
//               <span class="source-page">${pageInfo}</span>
//             </div>
//             <div class="source-text">${escapeHtml(src.text.trim())}</div>
//           </div>
//         `;
//       });
//       sourcesHTML += '</div>';

//       // Append directly below the AI message
//       const sourcesDiv = document.createElement("div");
//       sourcesDiv.className = "sources-wrapper";
//       sourcesDiv.innerHTML = sourcesHTML;
//       aiBubble.parentNode.appendChild(sourcesDiv);
//     }
//     // === END OF NEW PART ===

//     // Update chat title on first message
//     if (chatLog.children.length <= 4) {
//       const shortTitle = msg.substring(0, 35) + (msg.length > 35 ? "..." : "");
//       const sidebarItem = document.querySelector(`.chat-item[data-id="${currentChatId}"] .chat-title`);
//       if (sidebarItem) sidebarItem.textContent = shortTitle;
//     }

//     // Handle new chat_id from server
//     if (data.chat_id && !currentChatId) {
//       currentChatId = data.chat_id;
//       if (sessionIdEl) sessionIdEl.value = currentChatId;
//       loadChatList();
//     }

//   } catch (e) {
//     console.error(e);
//     setTyping(false);
//     bubble("ai", md("Server error or unauthorized."));
//   }
// }





async function send() {
  const msg = (input.value || "").trim();
  if (!msg) return;

  bubble("user", md(msg));
  saveLocal("user", msg);
  input.value = "";

  const typingEl = setTyping(true);

  try {
    const payload = {
      messages: [{ content: msg }],
      role: (roled?.value || "user"),
      retrieval_mode: (retrievalModeSel?.value || "hybrid"),
      chat_id: currentChatId
    };

    const token = localStorage.getItem("AUTH_TOKEN");
    const headers = {
      "Content-Type": "application/json",
      ...(token && { "Authorization": `Bearer ${token}` })
    };

    const r = await fetch(API_BASE() + "/chat", {
      method: "POST",
      headers: headers,
      body: JSON.stringify(payload)
    });

    if (!r.ok) throw new Error("HTTP " + r.status);

    const data = await r.json();

    setTyping(false);
    const text = data.reply || "...";
    bubble("ai", md(text));
    saveLocal("assistant", text);

    // === THIS IS THE ONLY CHANGE: SEND SOURCES TO SIDEBAR PANEL ===
    if (data.sources && data.sources.length > 0) {
      window.showSources(data.sources);  // This opens your right sidebar automatically!
    } else {
      // Optional: clear panel if no sources
      if (typeof window.showSources === "function") {
        window.showSources([]);
      }
    }
    // === END OF CHANGE ===

    // Update title & chat_id (your existing code)
    if (chatLog.children.length <= 4) {
      const shortTitle = msg.substring(0, 35) + (msg.length > 35 ? "..." : "");
      const sidebarItem = document.querySelector(`.chat-item[data-id="${currentChatId}"] .chat-title`);
      if (sidebarItem) sidebarItem.textContent = shortTitle;
    }

    if (data.chat_id && !currentChatId) {
      currentChatId = data.chat_id;
      const sessionIdEl = document.getElementById("sessionId");
      if (sessionIdEl) sessionIdEl.value = currentChatId;
      loadChatList();
    }

  } catch (e) {
    console.error(e);
    setTyping(false);
    bubble("ai", md("Server error or unauthorized."));
  }
}




  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
  clearBtn.addEventListener("click", () => {
    localStorage.removeItem("NEBULA_CHAT");
    chatLog.innerHTML = "";
  });

  loadLocal();




  
})();

// ---------- Dashboard / Health / Chart ----------
let statsChart;
// async function loadHealth() {
//   try {
//     const d = await fetch(API_BASE() + "/health", { cache: "no-store" }).then(r => r.json());
//     $("#health").textContent =
//       `${d.status} • ${d.chunks} chunks • ${d.files} files • emb=${d.emb_model} • bm25=${d.bm25} • ocr=${d.ocr}`;

//     const ctx = $("#statsChart");
//     const data = {
//       labels: ["Files", "Chunks", "BM25", "OCR"],
//       datasets: [{
//         label: "System Metrics",
//         data: [d.files || 0, d.chunks || 0, d.bm25 ? 1 : 0, d.ocr ? 1 : 0],
//       }]
//     };
//     const options = {
//       responsive: true,
//       plugins: { legend: { labels: { color: "#cfcfe6" } } },
//       scales: {
//         x: { ticks: { color: "#cfcfe6" }, grid: { color: "rgba(255,255,255,.08)" } },
//         y: { ticks: { color: "#cfcfe6" }, grid: { color: "rgba(255,255,255,.08)" } }
//       }
//     };
//     if (statsChart) statsChart.destroy();
//     statsChart = new Chart(ctx, { type: "bar", data, options });
//   } catch {
//     $("#health").textContent = "offline";
//   }
// }

async function loadHealth() {
  try {
    const d = await fetch(API_BASE() + "/health", { cache: "no-store" })
      .then(r => r.json());

    // Super clean & beautiful table-style with emojis
    $("#health").innerHTML = `
  <div class="health-table">
    <div class="health-row header">
      <div class="health-col icon">Status</div>
      <div class="health-col value">Value</div>
    </div>
    
    <div class="health-row">
      <div class="health-col icon">Server</div>
      <div class="health-col value"><span class="status-online">🟢 Online</span></div>
    </div>
    
    <div class="health-row">
      <div class="health-col icon">Documents</div>
      <div class="health-col value">${d.files} files</div>
    </div>
    
    <div class="health-row">
      <div class="health-col icon">Chunks</div>
      <div class="health-col value">${d.chunks.toLocaleString()}</div>
    </div>
    
    <div class="health-row">
      <div class="health-col icon">Vector DB</div>
      <div class="health-col value">${d.emb_model.toLocaleString()} vectors</div>
    </div>
    
   
    <div class="health-row">
      <div class="health-col icon">OCR</div>
      <div class="health-col value ${d.ocr ? 'text-green-400' : 'text-red-600'}">
        ${d.ocr ? 'Enabled' : 'Disabled'}
      </div>
    </div>
  </div>
`;

    // Keep your chart (slightly improved colors)
    const ctx = $("#statsChart");
    const data = {
      labels: ["Files", "Chunks", "Vector Index", "OCR"],
      datasets: [{
        label: "System Health",
        data: [d.files, d.chunks, d.emb_model, d.ocr ? 1 : 0],
        backgroundColor: ["#c084fc", "#a78bfa", "#e0aaff", "#c8a2ff"],
        borderRadius: 8,
        borderSkipped: false,
      }]
    };

    const options = {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: "#1e1b4b" }
      },
      scales: {
        y: { 
          beginAtZero: true,
          ticks: { color: "#e0e0ff" },
          grid: { color: "rgba(255,255,255,0.05)" }
        },
        x: { 
          ticks: { color: "#e0e0ff" },
          grid: { display: false }
        }
      }
    };

    if (statsChart) statsChart.destroy();
    statsChart = new Chart(ctx, { type: "bar", data, options });

  } catch (err) {
    $("#health").innerHTML = `
      <div class="health-grid">
        <div class="health-item">
          <span class="health-emoji">Server</span>
          <span class="health-value text-red-500 font-bold">Offline</span>
        </div>
      </div>
    `;
  }
}





$("#refreshBtn").addEventListener("click", e => { e.preventDefault(); loadHealth(); });

$("#resetBtn").addEventListener("click", async e => {
  e.preventDefault();
  if (!confirm("Reset vector index & metadata? This cannot be undone!")) return;
  showToast("Resetting index & metadata...", "info");
  try {
    const res = await fetch(API_BASE() + "/reset", { method: "POST" });
    if (res.ok) {
      showToast("Index & metadata reset successfully!", "success");
      loadHealth();
    } else {
      showToast("Reset failed (server error)", "error");
    }
  } catch {
    showToast("Reset failed – server offline", "error");
  }
});

// ---------- boot ----------
window.addEventListener("load", () => {
  pages.forEach(p => $("#" + p).style.display = ($("#" + p).classList.contains("active") ? 'block' : 'none'));
  loadHealth();
});

// ---------- Options Popup ----------
const optionsBtn = document.getElementById('optionsBtn');
const optionsPopup = document.getElementById('optionsPopup');
const optionItems = document.querySelectorAll('.option-item');

if (optionsBtn) {
  optionsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    optionsPopup.classList.toggle('hidden');
  });
}

optionItems.forEach(item => {
  item.addEventListener('click', () => {
    optionItems.forEach(i => i.classList.remove('active'));
    item.classList.add('active');
    optionsPopup.classList.add('hidden');
    const type = item.dataset.type;
    if (type === 'upload') {
      document.getElementById('chatInput').placeholder = "Upload file selected…";
    } else if (type === 'reasoning') {
      document.getElementById('chatInput').placeholder = "Reasoning mode enabled…";
    } else if (type === 'settings') {
      alert("Settings clicked!");
    }
  });
});

document.addEventListener('click', (e) => {
  if (optionsPopup && !optionsPopup.classList.contains('hidden')) {
    optionsPopup.classList.add('hidden');
  }
});

// ---------- Option Selection ----------
document.querySelectorAll('.option').forEach(option => {
  const cancelBtn = option.querySelector('.cancel-btn');
  if (cancelBtn) {
    option.addEventListener('click', (e) => {
      if (e.target === cancelBtn) return;
      option.classList.add('selected');
    });
    cancelBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      option.classList.remove('selected');
    });
  }
});

// ---------- Metrics ----------
const API = location.origin;

// async function loadMetrics() {
//   try {
//     // ← THIS IS THE ONLY LINE THAT WAS WRONG
//     const m = await fetch(API_BASE() + "/metrics", { cache: "no-store" })
//       .then(r => {
//         if (!r.ok) throw new Error("metrics 404");
//         return r.json();
//       });

//     const top = (m.top_queries || [])
//       .map(([q, c]) => `• ${q} <span class="text-gray-400">(${c}x)</span>`)
//       .join("<br>");

//     document.getElementById("metrics").innerHTML = `
//       <div class="metrics-grid">
//         <div><b>Total Searches</b>     ${m.search_count ?? 0}</div>
//         <div><b>Avg Latency</b>        ${m.avg_total_ms?.toFixed(0) ?? "-"} ms</div>
//         <div><b>Embed Latency</b>      ${m.avg_embed_ms?.toFixed(0) ?? "-"} ms</div>
//         <div><b>LLM Latency</b>        ${m.avg_llm_ms?.toFixed(0) ?? "-"} ms</div>
//         <div><b>Active Sessions</b>    ${m.sessions ?? 0}</div>
//         <div><b>Top Queries</b><br><span class="text-sm">${top || "—"}</span></div>
//         <div><b>Last Updated</b>       ${m.last_updated?.split("T")[1]?.slice(0,8) ?? "-"}</div>
//       </div>
//     `;
//   } catch (err) {
//     console.error("Metrics failed:", err);
//     document.getElementById("metrics").innerHTML = `<span class="text-red-500">Metrics offline or not implemented</span>`;
//   }
// }

async function loadMetrics() {
  try {
    const API = API_BASE();
    const res = await fetch(API + "/metrics", {
      cache: "no-store",
      headers: apiHeaders()
    });

    if (!res.ok) throw new Error("Failed to load metrics");

    const m = await res.json();
    console.log("Metrics:", m); // keep this — super helpful

    // Format top queries nicely
    const topQueriesHTML = (m.top_queries || [])
      .slice(0, 6)
      .map(([query, count], i) => `
        <div class="flex justify-between text-sm">
          <span class="truncate max-w-64">${i + 1}. ${escapeHtml(query)}</span>
          <span class="text-blue-400 font-medium">${count}x</span>
        </div>
      `)
      .join("") || '<span class="text-gray-500">No queries yet</span>';

    // Format time
    const lastTime = m.last_ingest_at
      ? new Date(m.last_ingest_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : "—";

    document.getElementById("metrics").innerHTML = `
      <div class="bg-gray-900/60 backdrop-blur border border-gray-800 rounded-xl p-5 shadow-xl">
        <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
          Live Analytics
          <span class="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full animate-pulse">
            LIVE
          </span>
        </h3>

        <div class="grid grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
          <!-- Total Searches -->
          <div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div class="text-gray-400">Total Searches</div>
            <div class="text-2xl font-bold text-white mt-1">${m.search_count || 0}</div>
          </div>

          <!-- Avg Total Latency -->
          <div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div class="text-gray-400">Avg Latency</div>
            <div class="text-2xl font-bold text-cyan-400 mt-1">
              ${m.avg_time_ms ? m.avg_time_ms.toFixed(1) + " ms" : "—"}
            </div>
          </div>

          <!-- Embed Latency -->
          <div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div class="text-gray-400">Embed + Search</div>
            <div class="text-xl font-bold text-orange-400 mt-1">
              ${m.avg_embed_ms ? m.avg_embed_ms.toFixed(0) + " ms" : "—"}
            </div>
          </div>

          <!-- LLM Latency -->
          <div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div class="text-gray-400">LLM Response</div>
            <div class="text-xl font-bold text-green-400 mt-1">
              ${m.avg_llm_ms ? m.avg_llm_ms.toFixed(0) + " ms" : "—"}
            </div>
          </div>

          <!-- Sessions -->
          <div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div class="text-gray-400">Active Sessions</div>
            <div class="text-2xl font-bold text-purple-400 mt-1">${m.sessions || 0}</div>
          </div>

          <!-- Last Updated -->
          <div class="bg-gray-800/50 rounded-lg p-3 border border-gray-700">
            <div class="text-gray-400">Last Activity</div>
            <div class="text-xl font-mono text-yellow-400 mt-1">${lastTime}</div>
          </div>
        </div>

        <!-- Top Queries -->
        <div class="mt-5">
          <div class="text-gray-300 font-medium mb-2">Top Queries</div>
          <div class="text-sm space-y-1.5">
            ${topQueriesHTML}
          </div>
        </div>
      </div>
    `;

  } catch (err) {
    console.error("Metrics error:", err);
    document.getElementById("metrics").innerHTML = `
      <div class="bg-red-900/30 border border-red-800 rounded-xl p-6 text-center">
        <p class="text-red-400">Analytics temporarily unavailable</p>
      </div>
    `;
  }
}

// Helper: prevent XSS in queries
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}



// ← THIS WAS MISSING – call it once + refresh every 15s
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();                    // ← first immediate load
  setInterval(loadMetrics, 15000);  // ← refresh every 15 seconds
});

// ---------- LOGIN PAGE LOGIC ----------
let authMode = 'login';

function setAuthMode(mode) {
  authMode = mode;
  if (mode === 'login') {
    $('#tabLogin').classList.add('text-nebula-300', 'border-b-2', 'border-nebula-400');
    $('#tabRegister').classList.remove('text-nebula-300', 'border-b-2', 'border-nebula-400');
    $('#authActionBtn').textContent = 'Sign In';
  } else {
    $('#tabRegister').classList.add('text-nebula-300', 'border-b-2', 'border-nebula-400');
    $('#tabLogin').classList.remove('text-nebula-300', 'border-b-2', 'border-nebula-400');
    $('#authActionBtn').textContent = 'Sign Up';
  }
}

$('#authActionBtn').addEventListener('click', async () => {
  const username = $('#authUser').value.trim();
  const password = $('#authPass').value.trim();
  if (!username || !password) {
    $('#authMsg').textContent = 'Please enter username and password.';
    return;
  }
  $('#authMsg').textContent = authMode === 'login' ? 'Signing in...' : 'Registering...';
  try {
    const endpoint = authMode === 'login' ? '/login' : '/register';
    const res = await fetch(API_BASE() + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (res.ok && data.access_token) {
      localStorage.setItem('AUTH_TOKEN', data.access_token);
      $('#authMsg').textContent = '✅ Authenticated!';
      show('home');
    } else {
      $('#authMsg').textContent = data.detail || 'Authentication failed.';
    }
  } catch {
    $('#authMsg').textContent = 'Server offline.';
  }
});

function googleAuth() {
  window.location.href = "http://localhost:8000/auth/google/login";
}

// Update Navbar based on auth state
function updateAuthNav() {
  const token = localStorage.getItem("AUTH_TOKEN");
  const container = document.getElementById("authNavContainer");
  if (!token) {
    container.innerHTML = `<button class="px-3 py-1.5 rounded-lg nav-link hover:text-nebula-200" onclick="show('login')">Login</button>`;
    return;
  }
  let userName = "User";
  let userPicture = null;
  let userEmail = "";
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    userName = payload.name || payload.email.split('@')[0];
    userEmail = payload.email || "";
    userPicture = payload.picture || null;
  } catch (e) { }
  container.innerHTML = `
    <div class="relative group">
      <button class="flex items-center gap-3 px-4 py-2 rounded-xl hover:bg-white/10 transition-all">
        ${userPicture ? `<img src="${userPicture}" class="w-8 h-8 rounded-full object-cover flex-shrink-0 border-4 border-nebula-500/80 shadow-2xl ring-2 ring-nebula-600/40" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex'" alt="${userName}">` : ''}
        <div class="${userPicture ? 'hidden' : 'flex'} w-9 h-9 rounded-full !bg-gradient-to-br !from-nebula-600 !via-nebula-500 !to-purple-600 flex items-center justify-center text-white font-bold text-lg border-4 border-nebula-400 shadow-2xl ring-4 ring-nebula-500/40 flex-shrink-0">${userName.charAt(0).toUpperCase()}</div>
        <svg class="w-4 h-4 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
        </svg>
      </button>
      <div class="absolute right-0 top-full mt-3 w-56 glass rounded-xl shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all border border-white/10">
        <div class="p-3 border-b border-white/10">
          <div class="text-xs text-white/60">Signed in as</div>
          <div class="font-medium truncate">${userName}</div>
          <div class="text-xs text-white/50 truncate">${userEmail}</div>
        </div>
        <button onclick="showAccountPage()" class="w-full text-left px-4 py-3 hover:bg-white/10">Your Account</button>
        <button onclick="showUpgradePage()" class="w-full text-left px-4 py-3 hover:bg-violet-500/80">Upgrade Plan</button>
        <button onclick="logout()" class="w-full text-left px-4 py-3 hover:bg-red-500/80 text-white-400">Logout</button>
      </div>
    </div>
  `;
  if (document.getElementById("account")) {
    document.getElementById("accountName").textContent = userName;
    document.getElementById("accountEmail").textContent = userEmail;
    document.getElementById("accountAvatar").textContent = userName.charAt(0).toUpperCase();
    if (userPicture) {
      document.getElementById("accountAvatar").innerHTML = `<img src="${userPicture}" class="w-full h-full rounded-full object-cover">`;
    }
  }
}

function showAccountPage() {
  const token = localStorage.getItem("AUTH_TOKEN");
  if (!token) return;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    document.getElementById("accountName").textContent = payload.name || "User";
    document.getElementById("accountEmail").textContent = payload.email || "";
    const avatar = document.getElementById("accountAvatar");
    if (payload.picture) {
      avatar.innerHTML = `<img src="${payload.picture}" class="w-full h-full rounded-full object-cover">`;
    } else {
      avatar.textContent = (payload.name || "U").charAt(0).toUpperCase();
    }
  } catch (e) { }
  show('account');
}

function showUpgradePage() {
  show('upgrade');
}

function logout() {
  localStorage.removeItem("AUTH_TOKEN");
  updateAuthNav();
  show('home');
  showToast("Logged out successfully! 😟", 'logout');
}

// Run on page load
document.addEventListener("DOMContentLoaded", () => {
  updateAuthNav();
  const params = new URLSearchParams(window.location.search);
  if (params.has("jwt")) {
    const jwt = params.get("jwt");
    localStorage.setItem("AUTH_TOKEN", jwt);
    currentChatId = null;
    window.history.replaceState({}, document.title, window.location.pathname);
    updateAuthNav();
    showToast("Logged in successfully! 😉", 'login');
  }
});

function showToast(message, type = "success") {
  const portal = document.getElementById("nebulaToastPortal");
  if (!portal) {
    console.log("Toast:", message);
    return;
  }
  const toast = document.createElement("div");
  toast.className = `nebula-toast ${type}`;
  toast.textContent = message;
  portal.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 800);
  }, 3500);
}

// CHAT-ONLY UPLOAD
document.getElementById('chatDedicatedFileInput')?.addEventListener('change', async (e) => {
  const files = e.target.files;
  if (!files?.length) return;
  const fileName = files.length > 1 ? `${files.length} files` : files[0].name;
  showToast(`Uploading ${fileName}...`, "info");
  try {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    const res = await fetch(API_BASE() + "/upload", { method: "POST", body: form });
    if (res.ok) {
      showToast(`File uploaded successfully!`, "success");
      loadHealth?.();
    } else {
      showToast(`Upload failed: ${res.status}`, "error");
    }
  } catch (err) {
    showToast(`❌ Failed to upload ${fileName}`, "error");
  } finally {
    e.target.value = "";
  }
});

// Safe select chat fallback
function safeSelectChat(id) {
  if (typeof selectChat === 'function') {
    selectChat(id);
  } else {
    const chat = chats?.find(c => c.id === id) || null;
    if (chat) {
      const chatLog = document.getElementById('chatLog');
      const sessionId = document.getElementById('sessionId');
      if (sessionId) sessionId.value = id;
      if (chatLog) {
        chatLog.innerHTML = (chat.messages || []).map(m => {
          const cls = m.role === 'user' ? 'bubble bubble-user' : 'bubble bubble-ai';
          return `<div class="${cls}">${m.content || ''}</div>`;
        }).join('') || `<div class="text-white/60 small">No messages yet.</div>`;
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    }
  }
}

// ============= CONSOLIDATED CHAT SIDEBAR =============
// THIS REPLACES ALL REDUNDANT SIDEBAR CODE - KEEP THIS BLOCK ONLY


(function initChatSidebar() {
  const sidebar = document.getElementById("chatSidebar");
  const toggleBtn = document.getElementById("sidebarToggle");
  const closeBtn = document.getElementById("sidebarCloseBtn");
  const chatList = document.getElementById("chatList");
  const newChatBtn = document.getElementById("newChatBtn");
  const chatLog = document.getElementById("chatLog");
  const sessionIdEl = document.getElementById("sessionId");
  const clearAllBtn = document.getElementById("clearAllChatsBtn");
  // API headers helper
  // FIXED – this sends the token correctly
function apiHeaders() {
  const token = localStorage.getItem("AUTH_TOKEN");
  if (!token) {
    console.warn("No AUTH_TOKEN found – you will get 401");
    return { "Content-Type": "application/json" };
  }
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${token}`   // ← THIS IS THE CORRECT WAY
  };
}

  // Helper: create the three-dot menu
  function createChatMenu(chatId, chatDiv) {
    const menu = document.createElement("div");
    menu.className = "chat-menu";
    menu.innerHTML = `
      <button class="rename-btn">Rename</button>
      <button class="delete-btn">Delete</button>
      <button class="export-btn">Export PDF</button>
    `;
    chatDiv.appendChild(menu);


menu.querySelector(".rename-btn").onclick = async () => {
  menu.style.display = "none";

  // Create inline rename box
  const renameBox = document.createElement("div");
  renameBox.style.position = "absolute";
  renameBox.style.top = "50%";
  renameBox.style.left = "50%";
  renameBox.style.transform = "translate(-50%, -50%)";
  renameBox.style.background = "#2c2e34";
  renameBox.style.padding = "15px";
  renameBox.style.borderRadius = "8px";
  renameBox.style.boxShadow = "0 5px 20px rgba(0,0,0,0.4)";
  renameBox.style.zIndex = "999";
  renameBox.style.display = "flex";
  renameBox.style.flexDirection = "column";
  renameBox.style.width = "260px";
  renameBox.style.textAlign = "center";

  renameBox.innerHTML = `
    <input type="text" id="renameInputBox" 
      value="${chatDiv.querySelector(".chat-title").textContent}" 
      style="padding:8px; border-radius:6px; border:none; outline:none; background:#1f2024; color:#fff; margin-bottom:10px;">
    <div style="display:flex; justify-content:center; gap:8px;">
      <button id="renameSaveBtn" style="padding:6px 14px; border:none; border-radius:6px; background:#10a37f; color:white; cursor:pointer;">Save</button>
      <button id="renameCancelBtn" style="padding:6px 14px; border:none; border-radius:6px; background:#3a3b40; color:#ccc; cursor:pointer;">Cancel</button>
    </div>
  `;

  document.body.appendChild(renameBox);
  const input = renameBox.querySelector("#renameInputBox");
  const saveBtn = renameBox.querySelector("#renameSaveBtn");
  const cancelBtn = renameBox.querySelector("#renameCancelBtn");
  input.focus();

  // Cancel handler
  cancelBtn.onclick = () => renameBox.remove();

  // Save handler
  saveBtn.onclick = async () => {
    const newTitle = input.value.trim();
    if (!newTitle) return;
    chatDiv.querySelector(".chat-title").textContent = newTitle;
    renameBox.remove();
    try {
      await fetch(API_BASE() + `/chats/${chatId}`, {
        method: "PATCH",
        headers: apiHeaders(),
        cache: "no-store" ,         // ← ADD THIS
        body: JSON.stringify({ title: newTitle })
      });
    } catch (e) {
      console.error("Rename failed:", e);
    }
  };
};




    // Delete
    menu.querySelector(".delete-btn").onclick = async () => {
      if (!confirm("Delete this chat?")) return;
      menu.style.display = "none";
      chatDiv.remove();
      try {
        await fetch(API_BASE() + `/chats/${chatId}`, {
          method: "DELETE",
          headers: apiHeaders()
        });
      } catch (e) {
        console.error("Delete failed:", e);
      }
    };

    // Export PDF
    menu.querySelector(".export-btn").onclick = async () => {
      menu.style.display = "none";
      try {
        const res = await fetch(API_BASE() + `/chats/${chatId}`, { headers: apiHeaders() });
        const data = await res.json();
        const content = data.messages
          .map(m => `${m.role.toUpperCase()}: ${m.content}`)
          .join("\n\n");

        // Create a printable PDF
        const win = window.open("", "_blank");
        win.document.write(`
          <html>
            <head><title>Chat Export - ${chatId}</title></head>
            <body>
              <pre style="white-space: pre-wrap; font-family: sans-serif;">${content}</pre>
              <script>window.print();<\/script>
            </body>
          </html>
        `);
        win.document.close();
      } catch (e) {
        console.error("Export failed:", e);
      }
    };

    return menu;
  }

  // Add chat to sidebar UI
  window.addChatToSidebar = function(id, title) {
    if (!chatList) return;
    if ([...chatList.children].some(el => el.dataset?.id === id)) return;

    const div = document.createElement("div");
    div.className = "chat-item";
    div.dataset.id = id;

    const titleSpan = document.createElement("div");
    titleSpan.className = "chat-title";
    titleSpan.textContent = title || "New Chat";
    div.appendChild(titleSpan);

   
// Three dots button
const dots = document.createElement("div");
dots.className = "chat-options";
dots.innerHTML = "⋯";
div.appendChild(dots);

const menu = createChatMenu(id, div);

// Show menu when hovering on the 3 dots
dots.addEventListener("mouseenter", (e) => {
  e.stopPropagation();
  document.querySelectorAll(".chat-menu").forEach(m => m.style.display = "none");
  menu.style.display = "flex";
});

// Hide menu when mouse leaves the dots *and* the menu
dots.addEventListener("mouseleave", (e) => {
  setTimeout(() => {
    if (!menu.matches(":hover")) menu.style.display = "none";
  }, 100);
});

// Also hide when leaving the menu itself
menu.addEventListener("mouseleave", () => {
  menu.style.display = "none";
});




  

div.addEventListener("click", (e) => {
  if (e.target.closest(".chat-options")) return;  // don't trigger when clicking dots

  // Remove active from others
  document.querySelectorAll(".chat-item").forEach(el => el.classList.remove("active"));
  div.classList.add("active");

  loadChat(id);  // ← now works perfectly
});    chatList.prepend(div);
  };


  // Clear All Chats button handler
if (clearAllBtn) {
  clearAllBtn.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear all chats?")) return;

    // Clear all chats from UI
    chatList.innerHTML = "";

    // (Optional) tell backend to clear all
    try {
      await fetch(API_BASE() + "/chats/clear-all", {
        method: "DELETE",
        headers: apiHeaders(),
      });
        showToast("All chats cleared", "error");
    } catch (e) {
      console.error("Failed to clear all chats:", e);
      showToast("Failed to clear all chats", "error");
    }
  });
}


  // Load specific chat
//   window.loadChat = async function(id) {
//     currentChatId = id;
//     if (sessionIdEl) sessionIdEl.textContent = id || "—";
//     chatLog.innerHTML = "";
//     try {
//       const res = await fetch(API_BASE() + `/chats/${id}`, { headers: apiHeaders() });
//       const data = await res.json();
//       data.messages.forEach(m => bubble(m.role, md(m.content)));
//       chatLog.scrollTop = chatLog.scrollHeight;
//     } catch (e) {
//       showToast("Failed to load chat", "error");
//     }
//   };

// Load specific chat — FIXED VERSION
// window.loadChat = async function(id) {
//   currentChatId = id;                                // ← we need this
//   if (sessionIdEl) sessionIdEl.value = id || "—";

//   chatLog.innerHTML = '<div class="text-white/40 text-center py-8">Loading chat...</div>';

//   try {
//     const res = await fetch(API_BASE() + `/chats/${id}`, { 
//       headers: apiHeaders(),
//       cache: "no-store"
//     });

//     if (!res.ok) throw new Error("Not found");

//     const data = await res.json();
//     chatLog.innerHTML = "";  // clear loading message

//     data.messages.forEach(m => {
//       bubble(m.role === "user" ? "user" : "ai", md(m.content));
//     });

//     chatLog.scrollTop = chatLog.scrollHeight;
//   } catch (e) {
//     console.error("Failed to load chat:", e);
//     chatLog.innerHTML = '<div class="text-red-400 text-center">Failed to load chat</div>';
//     showToast("Failed to load chat", "error");
//   }
// };


// window.loadChat = async function(id) {
//   currentChatId = id;
//   if (sessionIdEl) sessionIdEl.value = id || "—";

//   chatLog.innerHTML = '<div class="text-white/40 text-center py-8">Loading chat...</div>';

//   try {
//     const res = await fetch(API_BASE() + `/chats/${id}`, { 
//       headers: apiHeaders(),
//       cache: "no-store"
//     });

//     if (!res.ok) {
//       throw new Error("Not found");
//     }

//     const data = await res.json();
//     chatLog.innerHTML = "";

//     // THIS IS THE ONLY IMPORTANT PART YOU WERE MISSING
//     if (!data.messages || data.messages.length === 0) {
//       chatLog.innerHTML = '<div class="text-white/40 text-center py-8">Start typing to begin...</div>';
//     } else {
//       chatLog.innerHTML = "";
//       data.messages.forEach(m => {
//         bubble(m.role === "user" ? "user" : "ai", md(m.content));
//       });
//     }
//     chatLog.scrollTop = chatLog.scrollHeight;
//   } catch (e) {
//     console.error("Load failed:", e);
//     chatLog.innerHTML = '<div class="text-red-400 text-center">Failed to load chat</div>';
//     showToast("Failed to load chat", "error");
//   }
// };


window.loadChat = async function(id) {
  currentChatId = id;
  if (sessionIdEl) sessionIdEl.value = id || "—";

  chatLog.innerHTML = '<div class="text-white/40 text-center py-8">Loading chat...</div>';

  try {
    const res = await fetch(API_BASE() + `/chats/${id}`, { 
      headers: apiHeaders(),
      cache: "no-store"
    });

    if (!res.ok) throw new Error("HTTP " + res.status);

    const data = await res.json();
    chatLog.innerHTML = "";  // clear loading

    if (!data.messages || data.messages.length === 0) {
      chatLog.innerHTML = '<div class="text-white/40 text-center py-8">Start typing to begin...</div>';
    } else {
      data.messages.forEach(m => {
        if (m.role === "system") return;

        const role = m.role === "user" ? "user" : "ai";
        const safeHTML = (window.marked && window.DOMPurify)
          ? DOMPurify.sanitize(marked.parse(m.content || ""))
          : String(m.content || "").replace(/[<>]/g, c => c === "<" ? "&lt;" : "&gt;");

        bubble(role, safeHTML);
      });
    }
    chatLog.scrollTop = chatLog.scrollHeight;

  } catch (e) {
    console.error("Load failed:", e);
    chatLog.innerHTML = '<div class="text-red-400 text-center">Failed to load chat<br><small>Check console (F12)</small></div>';
    showToast("Failed to load chat — check browser console", "error");
  }
};





  // Load all chats from backend
  async function loadChatList() {
  try {
    const res = await fetch(API_BASE() + "/chats", { 
      headers: apiHeaders(),
      cache: "no-store"
    });
    if (!res.ok) throw new Error("Failed to fetch chats");
    const data = await res.json();
    chatList.innerHTML = "";
    data.chats.reverse().forEach(chat => {
      window.addChatToSidebar(chat.id, chat.title || "New Chat");
    });
  } catch (e) {
    console.error("Failed to load chat list:", e);
    showToast("Failed to load chats (check login/Redis)", "error");
  }
}

//new 














  // function createNewChat() {
  //   currentChatId = null;
  //   if (sessionIdEl) sessionIdEl.textContent = "—";
  //   chatLog.innerHTML = "";
  //   $("#chatInput").value = "";
  //   $("#chatInput").focus();
  //   showToast("New chat started", "success");
    
  //   // Add temporary item to sidebar
  //   const tempId = "temp_" + Date.now();
  //   window.addChatToSidebar(tempId, "New chat");
  // }


async function createNewChat() {
  try {
    const res = await fetch(API_BASE() + "/chats/new", {
      method: "POST",
      headers: apiHeaders(),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || "Failed");
    }

    const data = await res.json();
    const realId = data.chat_id;

    // CRITICAL: Set currentChatId
    currentChatId = realId;

    // Update Session ID immediately
    if (sessionIdEl) {
      sessionIdEl.value = realId;  
      }

    // Clear chat
    chatLog.innerHTML = '<div class="text-white/40 text-center py-8">Start typing to begin...</div>';
    $("#chatInput").value = "";
    $("#chatInput").focus();

    // Add to sidebar
    window.addChatToSidebar(realId, "New Chat");

    // Activate it
    document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));
    const item = document.querySelector(`.chat-item[data-id="${realId}"]`);
    if (item) item.classList.add('active');

    showToast("New chat created!", "success");

  } catch (err) {
    console.error("Create chat failed:", err);
    showToast("Login or check Redis connection", "error");
  }
}




//   function createNewChat() {
//   currentChatId = null;
//   if (sessionIdEl) sessionIdEl.textContent = "—";
//   chatLog.innerHTML = "";
//   $("#chatInput").value = "";
//   $("#chatInput").focus();
//   showToast("New chat started", "success");

//   // Remove temp item first
//   document.querySelectorAll('.chat-item[data-id^="temp_"]').forEach(el => el.remove());

//   // Reload real chats from server
//   loadChatList();  // ← THIS LINE IS KEY
// }

  if (newChatBtn) newChatBtn.addEventListener("click", createNewChat);

  // Sidebar toggle
  if (toggleBtn) {
    toggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      sidebar.classList.remove("closed");
      sidebar.classList.add("open");
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      sidebar.classList.remove("open");
      sidebar.classList.add("closed");
    });
  }

  document.addEventListener("click", (e) => {
    if (sidebar && sidebar.classList.contains("open") && 
        !sidebar.contains(e.target) && !toggleBtn?.contains(e.target)) {
      sidebar.classList.remove("open");
      sidebar.classList.add("closed");
    }

    // Hide all menus if clicked outside
    if (!e.target.closest(".chat-options") && !e.target.closest(".chat-menu")) {
      document.querySelectorAll(".chat-menu").forEach(m => m.style.display = "none");
    }
  });

  // Initialize
  loadChatList();
//   setInterval(loadChatList, 30000);
})();

// ============= SOURCE VIEWER PANEL =============

(function initSourceViewer() {
  const panel = document.getElementById('sourcePanel');
  const toggleBtn = document.getElementById('sourceToggleBtn');
  const collapseBtn = document.getElementById('sourceCollapseBtn');
  const listEl = document.getElementById('sourceList');
  const clearBtn = document.getElementById('clearSourcesBtn');
  const exportBtn = document.getElementById('exportSourcesBtn');

  const modal = document.getElementById('sourceModal');
  const modalBody = document.getElementById('modalBody');
  const modalTitle = document.getElementById('modalTitle');
  const modalClose = document.getElementById('modalCloseBtn');
  const copyCitationBtn = document.getElementById('copyCitationBtn');
  const openFileBtn = document.getElementById('modalOpenFileBtn');

  function safeText(s) { return String(s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  function openPanel() { panel?.classList.remove('closed'); panel?.classList.add('open'); panel?.setAttribute('aria-hidden', 'false'); }
  function closePanel() { panel?.classList.add('closed'); panel?.classList.remove('open'); panel?.setAttribute('aria-hidden', 'true'); }
  if (toggleBtn) toggleBtn.addEventListener('click', openPanel);
  if (collapseBtn) collapseBtn.addEventListener('click', closePanel);

  // window.showSources = function (sources) {
  //   if (!listEl) return;
  //   if (!sources || !sources.length) {
  //     listEl.innerHTML = '<div class="empty">No sources found for this response.</div>';
  //     return;
  //   }
  //   listEl.innerHTML = '';
  //   sources.forEach((src, idx) => {
  //     const it = document.createElement('div');
  //     it.className = 'source-item';
  //     it.dataset.idx = idx;

  //     const relWrap = document.createElement('div');
  //     relWrap.className = 'relevance';
  //     const relBar = document.createElement('div');
  //     relBar.className = 'rel-bar';
  //     const fill = document.createElement('div');
  //     fill.className = 'rel-fill';
  //     const pct = Math.min(100, Math.round((src.score || 0) * 100));
  //     fill.style.width = pct + '%';
  //     relBar.appendChild(fill);
  //     relWrap.appendChild(relBar);
  //     relWrap.innerHTML += `<div style="font-size:11px;color:#dcd6ff;margin-top:6px">${pct}%</div>`;

  //     const meta = document.createElement('div');
  //     meta.className = 'source-meta';
  //     const title = document.createElement('div');
  //     title.className = 'source-title';
  //     title.innerHTML = safeText(src.file || `Document ${idx + 1}`);
  //     const snippet = document.createElement('div');
  //     snippet.className = 'source-snippet';
  //     snippet.innerHTML = safeText((src.text || '').slice(0, 320)) + (src.text && src.text.length > 320 ? '…' : '');
  //     const footer = document.createElement('div');
  //     footer.className = 'source-footer';
  //     footer.innerHTML = `<div>${safeText(src.page ? 'page ' + src.page : '')}</div><div>• ${safeText(src.cursor || '')}</div>`;

  //     meta.appendChild(title);
  //     meta.appendChild(snippet);
  //     meta.appendChild(footer);

  //     it.appendChild(relWrap);
  //     it.appendChild(meta);

  //     it.addEventListener('click', () => { openModal(src); });
  //     listEl.appendChild(it);
  //   });
  //   openPanel();
  // };

  window.showSources = function (sources) {
  const listEl = document.getElementById('sourceList');
  const panel = document.getElementById('sourcePanel');
  if (!listEl || !panel) return;

  // Clear previous content
  listEl.innerHTML = '';

  if (!sources || sources.length === 0) {
    listEl.innerHTML = `
      <div class="empty text-center py-8 text-white/50">
        No sources were used for this answer.
      </div>`;
    panel.classList.add('closed');
    panel.classList.remove('open');
    return;
  }

  // Auto-open the panel when sources exist
  panel.classList.remove('closed');
  panel.classList.add('open');

  sources.forEach((src, i) => {
    const rank = i + 1;

    // Detect if it's OCR (image)
    const isImage = /\.(png|jpe?g|webp|bmp|tiff)$/i.test(src.source || '');
    const pageInfo = src.page !== null && src.page !== undefined
      ? `page ${src.page}`
      : isImage
        ? '(OCR from screenshot)'
        : '';

    const block = document.createElement('div');
    block.className = 'source-block mb-4 p-4 bg-white/5 rounded-lg border-l-4 border-purple-500 hover:bg-white/10 transition-all cursor-pointer';

    block.innerHTML = `
      <div class="flex items-center gap-3 mb-2">
        <span class="source-rank">#${rank}</span>
        <strong class="text-purple-300 truncate max-w-xs">${escapeHtml(src.source || 'Unknown file')}</strong>
        ${pageInfo ? `<span class="text-sm text-gray-400">→ ${pageInfo}</span>` : ''}
      </div>
      <div class="source-text text-sm leading-relaxed font-mono bg-black/30 p-3 rounded border border-white/10 max-h-48 overflow-y-auto text-gray-200">
        ${escapeHtml(src.text || '').replace(/\n/g, '<br>')}
      </div>
    `;

    // Click to open modal (optional — keep your existing modal if you want)
    block.addEventListener('click', () => openModal(src));

    listEl.appendChild(block);
  });
};

// Helper: safely escape HTML + preserve newlines
function escapeHtml(text) {
  if (typeof text !== 'string') return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}


  function openModal(src) {
    if (!modal) return;
    modalTitle.textContent = `${src.file || 'Source'} ${src.page ? ' — page ' + src.page : ''}`;
    modalBody.innerHTML = `<pre style="white-space:pre-wrap;">${safeText(src.text || '')}</pre>`;
    modal.classList.remove('hidden');
    modal._current = src;
  }

  function closeModal() {
    if (!modal) return;
    modal.classList.add('hidden');
    modal._current = null;
  }

  if (modalClose) modalClose.addEventListener('click', closeModal);
  if (modal) modal.addEventListener('click', (e) => { if (e.target === modal.querySelector('.modal-backdrop')) closeModal(); });

  if (copyCitationBtn) copyCitationBtn.addEventListener('click', () => {
    const src = modal._current;
    if (!src) return;
    const citation = `${src.file || 'Document'}${src.page ? (' — page ' + src.page) : ''}\n\n${(src.text || '').slice(0, 300)}`;
    navigator.clipboard && navigator.clipboard.writeText(citation).then(() => {
      copyCitationBtn.textContent = 'Copied ✓';
      setTimeout(() => copyCitationBtn.textContent = 'Copy citation', 1400);
    }).catch(() => alert('Copy failed — please copy manually.'));
  });

  if (openFileBtn) openFileBtn.addEventListener('click', () => {
    const src = modal._current;
    if (!src) return;
    if (src.url) window.open(src.url, '_blank');
    else alert('No file URL provided for this source.');
  });

  if (clearBtn) clearBtn.addEventListener('click', () => {
    if (listEl) listEl.innerHTML = '<div class="empty">No sources yet — ask a question to see retrieval results.</div>';
    closePanel();
  });

  


  if (exportBtn) exportBtn.addEventListener('click', () => {
    const items = Array.from(listEl?.querySelectorAll('.source-item') || []).map(it => {
      const idx = it.dataset.idx;
      return { idx, text: it.querySelector('.source-snippet')?.textContent || '' };
    });
    const blob = new Blob([JSON.stringify(items, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'nebula_sources.json';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeModal(); closePanel(); }
  });

})();

const $ = (s) => document.querySelector(s),
  esc = breakout.escape;
const labels = {
  championship: "Championship",
  "primeira-liga": "Primeira Liga",
  "belgian-pro-league": "Belgian Pro League",
  eredivisie: "Eredivisie",
};
const leagueName = (name) =>
  labels[name] ||
  String(name)
    .replaceAll("-", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
let players = [],
  saved = new Set(),
  dataset = null,
  view = "rankings",
  page = 0,
  filtered = [],
  toastTimer;
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").hidden = true), 3500);
}
function setView(next) {
  view = next;
  page = 0;
  const titles = {
    rankings: "Player rankings",
    shortlist: "My shortlist",
    model: "Model insights",
  };
  const descriptions = {
    rankings: "A closer look at the players with a bigger next chapter.",
    shortlist: "The prospects you want to keep in view.",
    model: "Understand the evidence behind the predictions.",
  };
  $("#view-title").textContent = titles[view];
  $("#breadcrumb").textContent = titles[view];
  $("#view-description").textContent = descriptions[view];
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("#rankings-view").hidden = view === "model";
  $("#model-view").hidden = view !== "model";
  render();
}
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => setView(b.dataset.view)));
function render() {
  $("#saved-count").textContent = saved.size;
  $("#shortlist-count").textContent = saved.size;
  const term = $("#search").value.toLowerCase().trim();
  filtered = players.filter(
    (p) =>
      (view !== "shortlist" || saved.has(p.id)) &&
      (!term || `${p.name} ${p.team}`.toLowerCase().includes(term)) &&
      ($("#cohort").value === "all" || p.source === $("#cohort").value) &&
      ($("#league").value === "all" || p.league === $("#league").value) &&
      ($("#season").value === "all" || p.season === $("#season").value) &&
      ($("#age").value === "all" ||
        (p.age !== null && p.age <= Number($("#age").value))),
  );
  filtered.sort((a, b) =>
    $("#sort").value === "name"
      ? a.name.localeCompare(b.name)
      : $("#sort").value === "age"
        ? (a.age ?? 999) - (b.age ?? 999)
        : b.probability - a.probability,
  );
  page = Math.min(page, Math.max(0, Math.ceil(filtered.length / 25) - 1));
  const shown = filtered.slice(page * 25, page * 25 + 25);
  $("#matches").textContent =
    `${filtered.length.toLocaleString()} ${filtered.length === 1 ? "record" : "records"}${view === "shortlist" ? " in your shortlist" : ""}`;
  $("#export").disabled = !filtered.length;
  $("#table-wrap").innerHTML = shown.length
    ? `<table class="data-table"><thead><tr><th>#</th><th>PLAYER / CLUB</th><th>LEAGUE</th><th>AGE</th><th>BREAKOUT SCORE</th><th>OUTCOME</th><th><span class="sr-only">Shortlist</span></th></tr></thead><tbody>${shown
        .map(
          (p, i) =>
            `<tr><td class="rank">${String(page * 25 + i + 1).padStart(2, "0")}</td><td><div class="player-cell"><span class="initials" aria-hidden="true">${esc(
              p.name
                .split(" ")
                .map((x) => x[0])
                .slice(0, 2)
                .join(""),
            )}</span><div><button class="player-name" data-player="${p.id}">${esc(p.name)}</button><span class="player-club">${esc(p.team)} / ${esc(p.season)}</span></div></div></td><td class="league-name">${esc(leagueName(p.league))}</td><td>${p.age ?? "--"}</td><td><div class="prob"><span>${(p.probability * 100).toFixed(1)}%</span><span class="prob-track"><span style="width:${p.probability * 100}%"></span></span></div></td><td><span class="outcome ${p.label === null ? "unknown" : ""}">${p.label === 1 ? "Confirmed breakout" : p.label === 0 ? "Below label threshold" : "Outcome not recorded"}</span></td><td><button class="save-button ${saved.has(p.id) ? "saved" : ""}" data-save="${p.id}" title="${saved.has(p.id) ? "Remove from" : "Add to"} shortlist" aria-label="${saved.has(p.id) ? "Remove" : "Save"} ${esc(p.name)}" aria-pressed="${saved.has(p.id)}"><i data-lucide="bookmark"></i></button></td></tr>`,
        )
        .join("")}</tbody></table>`
    : `<div class="empty-state"><i data-lucide="${view === "shortlist" ? "bookmark" : "search"}"></i><h3>${view === "shortlist" && !saved.size ? "Your next prospect starts here." : "No players match these filters."}</h3><p>${view === "shortlist" && !saved.size ? "Save players from the rankings to build your private shortlist." : "Try another name, league, or age range."}</p><button class="button" id="empty-reset">${view === "shortlist" && !saved.size ? "Explore rankings" : "Reset filters"}</button></div>`;
  $("#page-label").textContent = filtered.length
    ? `Showing ${page * 25 + 1}-${Math.min(page * 25 + 25, filtered.length)} of ${filtered.length}`
    : "No results";
  $("#prev").disabled = page === 0;
  $("#next").disabled = (page + 1) * 25 >= filtered.length;
  document
    .querySelectorAll("[data-player]")
    .forEach((b) => (b.onclick = () => showPlayer(b.dataset.player)));
  document
    .querySelectorAll("[data-save]")
    .forEach((b) => (b.onclick = () => toggleSave(b.dataset.save, b)));
  if ($("#empty-reset"))
    $("#empty-reset").onclick = () => {
      resetFilters();
      if (view === "shortlist" && !saved.size) setView("rankings");
    };
  breakout.icons();
}
async function toggleSave(id, button) {
  button.disabled = true;
  try {
    const wasSaved = saved.has(id);
    await breakout.api(`/api/shortlist/${id}`, {
      method: wasSaved ? "DELETE" : "POST",
    });
    wasSaved ? saved.delete(id) : saved.add(id);
    toast(
      wasSaved ? "Removed from your shortlist." : "Saved to your shortlist.",
    );
    render();
    if ($("#player-dialog").open) showPlayer(id, false);
  } catch (e) {
    toast(e.message);
    button.disabled = false;
  }
}
function showPlayer(id, open = true) {
  const p = players.find((p) => p.id === id);
  if (!p) return;
  $("#player-detail").innerHTML =
    `<article class="player-report"><h2>${esc(p.name)}</h2><p class="muted">${esc(p.team)} / ${esc(leagueName(p.league))}</p><div class="report-score"><strong>${(p.probability * 100).toFixed(1)}%</strong><span>Calibrated breakout probability at the time of prediction</span></div><dl class="report-grid"><div><dt>AGE AT PREDICTION</dt><dd>${p.age ?? "Not available"}</dd></div><div><dt>SEASON</dt><dd>${esc(p.season)}</dd></div><div><dt>OBSERVED DESTINATION</dt><dd>${esc(p.destination || "Not recorded")}</dd></div><div><dt>DATA SOURCE</dt><dd>${esc(p.source)}</dd></div>${p.lgbm != null ? `<div><dt>LIGHTGBM SCORE</dt><dd>${(p.lgbm * 100).toFixed(1)}%</dd></div>` : ""}${p.xgb != null ? `<div><dt>XGBOOST SCORE</dt><dd>${(p.xgb * 100).toFixed(1)}%</dd></div>` : ""}</dl><div class="report-warning">${dataset.snapshot ? "This is a selected historical case published in the project README, not a current player valuation. Individual SHAP values and detailed player statistics are not included in this snapshot." : "Saved prediction output. Individual SHAP explanations remain available in the original Streamlit dashboard when their artifacts are present."}</div><button class="button dark" id="detail-save"><i data-lucide="bookmark"></i>${saved.has(id) ? "Remove from shortlist" : "Save to shortlist"}</button></article>`;
  $("#detail-save").onclick = (e) => toggleSave(id, e.currentTarget);
  breakout.icons();
  if (open) $("#player-dialog").showModal();
}
function resetFilters() {
  $("#search").value = "";
  for (const id of ["cohort", "league", "season", "age"])
    $(`#${id}`).value = "all";
  $("#sort").value = "probability";
  page = 0;
  render();
}
$("#close-player").onclick = () => $("#player-dialog").close();
$("#reset").onclick = resetFilters;
for (const id of ["search", "cohort", "league", "season", "age", "sort"])
  $(`#${id}`).addEventListener(id === "search" ? "input" : "change", () => {
    page = 0;
    render();
  });
$("#prev").onclick = () => {
  page--;
  render();
};
$("#next").onclick = () => {
  page++;
  render();
};
$("#logout").onclick = async () => {
  try {
    await breakout.api("/api/logout", { method: "POST" });
    location.href = "/login";
  } catch (e) {
    toast(e.message);
  }
};
$("#logout-mobile").onclick = $("#logout").onclick;
$("#export").onclick = () => {
  const quote = (v) =>
    '"' +
    String(v ?? "")
      .replace(/^[=+@\-\t\r]/, "'$&")
      .replaceAll('"', '""') +
    '"';
  const rows = [
    ["player", "club", "league", "season", "age", "probability", "source"],
    ...filtered.map((p) => [
      p.name,
      p.team,
      p.league,
      p.season,
      p.age,
      p.probability,
      p.source,
    ]),
  ];
  const url = URL.createObjectURL(
    new Blob([rows.map((r) => r.map(quote).join(",")).join("\r\n")], {
      type: "text/csv;charset=utf-8",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "breakout-scouting.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function load() {
  try {
    dataset = await breakout.api("/api/players");
    players = dataset.players;
    saved = new Set(dataset.saved);
    $("#source-notice span").textContent = dataset.notice;
    $("#player-count").textContent = players.length.toLocaleString();
    $("#league-count").textContent = new Set(players.map((p) => p.league)).size;
    $("#highest").textContent = players.length
      ? `${(Math.max(...players.map((p) => p.probability)) * 100).toFixed(1)}%`
      : "--";
    $("#data-source").textContent = dataset.source;
    if (players.some((p) => p.source === "Scouting predictions")) {
      $("#cohort").value = "Scouting predictions";
    }
    const report = dataset.model_report;
    if (report?.metrics) {
      $("#metric-source").textContent =
        "HISTORICAL EVALUATION / EVALUATION_RESULTS.JSON";
      $("#metric-auc").textContent = report.metrics.roc_auc.toFixed(3);
      $("#metric-p20").textContent =
        `${(report.metrics.precision_at_20 * 100).toFixed(1)}%`;
      $("#metric-ap").textContent = report.metrics.average_precision.toFixed(3);
      $("#metric-brier").textContent = report.metrics.brier_score.toFixed(3);
    }
    if (report?.features) {
      $("#feature-source").textContent =
        "Global SHAP importance from feature_importance.csv. Not an explanation for any individual player.";
      const highest = Math.max(
        ...report.features.map((f) => f.importance),
        0.001,
      );
      $("#feature-bars").innerHTML = report.features
        .map(
          (f) =>
            `<div><span>${esc(f.name.replaceAll("_", " "))}</span><strong>${f.importance.toFixed(3)}</strong><i style="--width:${(f.importance / highest) * 100}%"></i></div>`,
        )
        .join("");
    }
    if (!dataset.snapshot) {
      $("#artifact-title").textContent = "Connected to your pipeline outputs";
      $("#artifact-description").textContent =
        `${players.length.toLocaleString()} player-season records loaded from saved prediction CSVs. Scouting predictions and historical evaluation are separate cohorts. These are not live or newly trained forecasts.`;
    }
    for (const key of ["league", "season"]) {
      const options = [...new Set(players.map((p) => p[key]))].sort();
      for (const value of options) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = key === "league" ? leagueName(value) : value;
        $(`#${key}`).append(option);
      }
    }
    render();
  } catch (e) {
    $("#app-error").textContent = e.message;
    $("#app-error").hidden = false;
    $("#source-notice span").textContent = "Data could not be loaded.";
    $("#table-wrap").innerHTML =
      '<div class="empty-state"><h3>Unable to load rankings.</h3><button class="button" id="retry">Retry</button></div>';
    $("#retry").onclick = load;
  }
}
load();

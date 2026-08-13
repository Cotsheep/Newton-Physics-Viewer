"use strict";

const app = document.querySelector("#app");
const STATUS_LABELS = {
  created: "已创建",
  running: "运行中",
  succeeded: "已完成",
  partially_succeeded: "部分完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  unknown: "状态未知",
};
const TEMPLATE_LABELS = {
  drop: "摔落试验",
  slope_friction: "坡度试验",
};

const DEVELOPMENT_OUTCOME_LABELS = {
  moved: "观察：发生沿坡移动",
  stayed_near_start: "观察：保持在起点附近",
  inconclusive: "观察：结果不明确",
};

let resultIndex = null;
let refreshTimer = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatTime(value) {
  if (!value) return "时间未记录";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function shortHash(value) {
  return typeof value === "string" && value.length > 12 ? value.slice(0, 12) : value || "未记录";
}

function statusBadge(status) {
  const badge = element("span", "status", STATUS_LABELS[status] || status || "状态未知");
  badge.dataset.status = status || "unknown";
  return badge;
}

function routeHref(parameters = {}) {
  const query = new URLSearchParams(parameters);
  return `#${query.toString()}`;
}

function currentRoute() {
  const raw = window.location.hash.startsWith("#") ? window.location.hash.slice(1) : "";
  return new URLSearchParams(raw);
}

function pauseOtherVideos(activeVideo) {
  document.querySelectorAll("video").forEach((video) => {
    if (video !== activeVideo && !video.paused) video.pause();
  });
}

function attachPlaybackPolicy(scope) {
  scope.querySelectorAll("video").forEach((video) => {
    video.addEventListener("play", () => pauseOtherVideos(video));
  });
}

function coverNode(asset) {
  if (asset.cover_url) {
    const image = element("img");
    image.src = asset.cover_url;
    image.alt = `${asset.display_name} 的资产封面`;
    image.loading = "lazy";
    return image;
  }
  return element("div", "asset-cover-placeholder", "N");
}

function renderHome() {
  const fragment = document.createDocumentFragment();
  const heading = element("section", "page-heading");
  const headingCopy = element("div");
  headingCopy.append(
    element("p", "eyebrow", "Asset library"),
    element("h1", "", "资产试验结果"),
    element(
      "p",
      "page-description",
      "每张封面对应一个资产。点击封面后，可按试验种类查看该资产的全部工况录像。",
    ),
  );
  heading.append(headingCopy, element("p", "refresh-note", `索引更新：${formatTime(resultIndex.generated_at)}`));
  fragment.append(heading);

  if (!resultIndex.assets.length) {
    const empty = element("section", "empty-state");
    empty.append(
      element("h1", "", "还没有可浏览的试验录像"),
      element(
        "p",
        "",
        "完成第一个工况并重新生成结果索引后，资产封面会出现在这里。",
      ),
    );
    fragment.append(empty);
    app.replaceChildren(fragment);
    return;
  }

  const grid = element("section", "asset-grid");
  resultIndex.assets.forEach((asset) => {
    const card = element("article", "asset-card");
    const coverLink = element("a", "asset-cover-link");
    coverLink.href = routeHref({ asset: asset.identity });
    coverLink.append(coverNode(asset));
    const overlay = element("span", "cover-overlay");
    overlay.append(
      element("span", "", `${asset.runs.length} 次运行`),
      statusBadge(asset.latest_status),
    );
    coverLink.append(overlay);

    const copy = element("div", "asset-copy");
    copy.append(
      element("h2", "asset-title", asset.display_name),
      element("p", "asset-identity", asset.identity),
    );
    const tags = element("div", "tag-row");
    asset.templates.forEach((template) => {
      tags.append(element("span", "tag", TEMPLATE_LABELS[template] || template));
    });
    copy.append(tags);
    card.append(coverLink, copy);
    grid.append(card);
  });
  fragment.append(grid);
  app.replaceChildren(fragment);
}

function describeCondition(condition) {
  if (!condition || typeof condition !== "object") return "工况参数未记录";
  const pairs = Object.entries(condition).map(([key, value]) => `${key}: ${value}`);
  return pairs.length ? pairs.join(" · ") : "工况参数未记录";
}

function caseCard(asset, run, testCase) {
  const card = element("article", "case-card");
  const media = element("div", "case-media");
  if (testCase.video_url) {
    const video = element("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = testCase.video_url;
    if (testCase.poster_url) video.poster = testCase.poster_url;
    video.setAttribute("playsinline", "");
    media.append(video);
  } else if (testCase.poster_url) {
    const image = element("img");
    image.src = testCase.poster_url;
    image.alt = testCase.label || testCase.case_id;
    image.loading = "lazy";
    media.append(image);
  } else {
    media.append(element("div", "case-media-placeholder", "该工况没有可播放录像"));
  }

  const copy = element("div", "case-copy");
  copy.append(
    element("h3", "", testCase.label || testCase.case_id),
    element("p", "", describeCondition(testCase.condition)),
  );
  const tags = element("div", "tag-row");
  tags.append(statusBadge(testCase.status));
  if (testCase.authoritative === false) {
    tags.append(element("span", "tag", "非正式开发冒烟"));
  }
  if (DEVELOPMENT_OUTCOME_LABELS[testCase.development_outcome]) {
    tags.append(
      element("span", "tag", DEVELOPMENT_OUTCOME_LABELS[testCase.development_outcome]),
    );
  }
  if (typeof testCase.duration_seconds === "number") {
    tags.append(element("span", "tag", `${testCase.duration_seconds.toFixed(1)} 秒`));
  }
  copy.append(tags);

  if (testCase.video_url) {
    const actions = element("div", "case-actions");
    const playerLink = element("a", "", "进入独立播放页");
    playerLink.href = routeHref({
      play: run.run_id,
      case: testCase.case_id,
      asset: asset.identity,
    });
    const downloadLink = element("a", "", "下载 MP4");
    downloadLink.href = testCase.video_url;
    downloadLink.download = "";
    actions.append(playerLink, downloadLink);
    copy.append(actions);
  }
  card.append(media, copy);
  return card;
}

function renderAsset(identity) {
  const asset = resultIndex.assets.find((item) => item.identity === identity);
  if (!asset) {
    renderNotFound("找不到这个资产的结果。它可能已进入回收站，或结果索引刚刚更新。");
    return;
  }

  const fragment = document.createDocumentFragment();
  const back = element("a", "back-link", "← 返回全部资产");
  back.href = "#";
  fragment.append(back);

  const heading = element("section", "page-heading");
  const headingCopy = element("div");
  headingCopy.append(
    element("p", "eyebrow", "Asset experiments"),
    element("h1", "", asset.display_name),
    element("p", "page-description", asset.identity),
  );
  heading.append(headingCopy, statusBadge(asset.latest_status));
  fragment.append(heading);

  asset.templates.forEach((template) => {
    const runs = asset.runs.filter((run) => run.template === template);
    const section = element("section", "experiment-section");
    const title = element("h2", "section-title");
    title.append(
      document.createTextNode(TEMPLATE_LABELS[template] || template),
      element("span", "tag", `${runs.length} 次运行`),
    );
    section.append(title);

    runs.forEach((run, index) => {
      const details = element("details", "run-block");
      if (index === 0) details.open = true;
      const summary = element("summary");
      const summaryMain = element("div", "run-summary-main");
      summaryMain.append(
        element("strong", "", index === 0 ? "最新一次运行" : formatTime(run.created_at)),
        element(
          "span",
          "run-meta",
          `${formatTime(run.created_at)} · 配置 ${run.profile_name || "未记录"} · 资产版本 ${shortHash(run.asset_version)}`,
        ),
      );
      summary.append(summaryMain, statusBadge(run.status));
      details.append(summary);

      const caseGrid = element("div", "case-grid");
      if (run.cases.length) {
        run.cases.forEach((testCase) => caseGrid.append(caseCard(asset, run, testCase)));
      } else {
        caseGrid.append(element("p", "page-description", run.progress || "这次运行尚未产生工况录像。"));
      }
      details.append(caseGrid);
      section.append(details);
    });
    fragment.append(section);
  });

  app.replaceChildren(fragment);
  attachPlaybackPolicy(app);
}

function findCase(runId, caseId) {
  for (const asset of resultIndex.assets) {
    for (const run of asset.runs) {
      if (run.run_id !== runId) continue;
      const testCase = run.cases.find((item) => item.case_id === caseId);
      if (testCase) return { asset, run, testCase };
    }
  }
  return null;
}

function renderPlayer(runId, caseId) {
  const found = findCase(runId, caseId);
  if (!found || !found.testCase.video_url) {
    renderNotFound("找不到这段工况录像。");
    return;
  }
  const { asset, run, testCase } = found;
  const fragment = document.createDocumentFragment();
  const back = element("a", "back-link", `← 返回 ${asset.display_name}`);
  back.href = routeHref({ asset: asset.identity });
  fragment.append(back);

  const player = element("section", "player-shell");
  const video = element("video");
  video.controls = true;
  video.autoplay = false;
  video.preload = "metadata";
  video.src = testCase.video_url;
  if (testCase.poster_url) video.poster = testCase.poster_url;
  video.setAttribute("playsinline", "");
  const details = element("div", "player-details");
  details.append(
    element("p", "eyebrow", TEMPLATE_LABELS[run.template] || run.template),
    element("h1", "", testCase.label || testCase.case_id),
    element("p", "page-description", describeCondition(testCase.condition)),
  );
  const facts = element("dl");
  [
    ["资产", asset.identity],
    ["运行", run.run_id],
    ["资产版本", run.asset_version],
    ["运行配置", run.profile_name || "未记录"],
    ["完成时间", formatTime(testCase.finished_at)],
  ].forEach(([label, value]) => {
    facts.append(element("dt", "", label), element("dd", "", value));
  });
  details.append(facts);
  player.append(video, details);
  fragment.append(player);
  app.replaceChildren(fragment);
  attachPlaybackPolicy(app);
}

function renderNotFound(message) {
  const empty = element("section", "empty-state");
  empty.append(element("h1", "", "没有找到结果"), element("p", "", message));
  const back = element("a", "back-link", "← 返回全部资产");
  back.href = "#";
  empty.append(back);
  app.replaceChildren(empty);
}

function renderError(error) {
  const empty = element("section", "empty-state");
  const box = element("div", "error-box");
  box.append(
    element("h1", "", "结果页暂时无法读取"),
    element(
      "p",
      "",
      `${error.message || error}。请确认结果服务仍在运行，并重新生成 index.json。`,
    ),
  );
  empty.append(box);
  app.replaceChildren(empty);
}

function renderRoute() {
  if (!resultIndex) return;
  const route = currentRoute();
  const runId = route.get("play");
  const caseId = route.get("case");
  if (runId && caseId) {
    renderPlayer(runId, caseId);
  } else if (route.has("asset")) {
    renderAsset(route.get("asset"));
  } else {
    renderHome();
  }
}

async function loadIndex({ rerender = true } = {}) {
  const response = await fetch(`index.json?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const nextIndex = await response.json();
  if (!nextIndex || !Array.isArray(nextIndex.assets)) {
    throw new Error("index.json 格式不正确");
  }
  resultIndex = nextIndex;
  if (rerender) renderRoute();
}

function scheduleRefresh() {
  window.clearInterval(refreshTimer);
  refreshTimer = window.setInterval(async () => {
    const videoIsPlaying = Array.from(document.querySelectorAll("video")).some(
      (video) => !video.paused && !video.ended,
    );
    if (videoIsPlaying) return;
    try {
      await loadIndex({ rerender: true });
    } catch (error) {
      console.warn("Result index refresh failed", error);
    }
  }, 10000);
}

window.addEventListener("hashchange", renderRoute);
loadIndex()
  .then(scheduleRefresh)
  .catch(renderError);

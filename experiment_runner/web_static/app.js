"use strict";

const app = document.querySelector("#app");
const STATUS_LABELS = {
  created: "等待开始",
  running: "测试中",
  succeeded: "已完成",
  partially_succeeded: "部分完成",
  failed: "运行失败",
  cancelled: "已取消",
  interrupted: "已中断",
  unknown: "状态未知",
};
const TEMPLATE_LABELS = {
  drop: "摔落测试",
  slope_friction: "斜坡测试",
};

const DEVELOPMENT_OUTCOME_LABELS = {
  moved: "沿斜坡发生了移动",
  stayed_near_start: "保持在起点附近",
  inconclusive: "暂时无法判断是否移动",
};
const NON_AUTHORITATIVE_TITLE = "仅用于验证测试流程";
const NON_AUTHORITATIVE_BOUNDARY = "本次不评价资产的物理参数是否准确。";

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

function formatFiniteNumber(value, fractionDigits = 3) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "未记录";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: fractionDigits }).format(value);
}

function formatMetric(value, unit, fractionDigits = 3) {
  const formatted = formatFiniteNumber(value, fractionDigits);
  return formatted === "未记录" ? formatted : `${formatted} ${unit}`;
}

function formatVector(value, unit) {
  if (
    !Array.isArray(value)
    || value.length !== 3
    || value.some((item) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    return "未记录";
  }
  return `[${value.map((item) => item.toFixed(3)).join(", ")}] ${unit}`;
}

function formatFiniteState(value) {
  if (value === true) return "未发现无效数值";
  if (value === false) return "计算中出现无效数值";
  return "未记录";
}

function formatConditionValue(value) {
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "未记录";
  if (Array.isArray(value)) {
    const formatted = value.map(formatConditionValue);
    return formatted.includes("未记录") ? "未记录" : formatted.join(", ");
  }
  if (value === null || value === undefined || value === "") return "未记录";
  return String(value);
}

function nonAuthoritativeNotice() {
  const notice = element("aside", "smoke-notice");
  notice.append(
    element("strong", "", NON_AUTHORITATIVE_TITLE),
    element("span", "", NON_AUTHORITATIVE_BOUNDARY),
  );
  return notice;
}

function observationValue(testCase, key) {
  if (testCase.finite === false) return "数值异常，不能解释";
  if (testCase.status === "created") return "尚未运行";
  if (testCase.status !== "succeeded") return "观察未完成";
  const measurement = testCase.observations?.[key];
  const labels = {
    invalid: "测量记录无效", unverified: "有效性未记录",
    incomplete: "观察未完成", not_started: "尚未运行", not_measured: "未测量",
    unsupported: "当前场景暂不支持测量", not_observed: "观察期内未触地",
    no_liftoff: "触地后未观察到整体离地",
  };
  if (measurement) {
    if (key === "first_rebound_ratio" && measurement.status === "measured"
        && measurement.unit === "1" && Number.isFinite(measurement.value)) {
      const percent = measurement.value * 100;
      return percent !== 0 && Math.abs(percent) < 0.0001
        ? `${percent.toExponential(2)} %` : formatMetric(percent, "%", 4);
    }
    if (["measured", "legacy_record"].includes(measurement.status)
        && measurement.unit === "m" && Number.isFinite(measurement.value)) {
      if (measurement.value !== 0 && Math.abs(measurement.value) < 0.01) {
        return Math.abs(measurement.value) < 1e-7
          ? `${measurement.value.toExponential(2)} 米`
          : formatMetric(measurement.value * 1000, "毫米", 4);
      }
      return lengthLabel(measurement.value);
    }
    return labels[measurement.status] || "未测量";
  }
  // Old servers have no observation protocol; preserve the recorded number,
  // explicitly labelled as legacy below, without inventing measurement metadata.
  if (key === "along_slope_displacement" && testCase.finite === true
      && Number.isFinite(testCase.displacement_along_slope)) {
    return lengthLabel(testCase.displacement_along_slope);
  }
  return "未测量";
}

function observationPanel(run, testCase) {
  const section = element("section", "inspection-panel");
  const slope = run.template === "slope_friction";
  section.append(element("h3", "", "量化观察"));
  const rows = slope
    ? [["沿坡位移", observationValue(testCase, "along_slope_displacement")]]
    : [["首次回弹高度", observationValue(testCase, "first_rebound_height")],
      ["占实际释放净空比例", observationValue(testCase, "first_rebound_ratio")],
      ["最大地面穿透", observationValue(testCase, "maximum_ground_penetration")]];
  const measurement = testCase.observations?.along_slope_displacement;
  if (slope && measurement?.status === "measured") {
    rows.push(["测量时窗", `${formatMetric(measurement.start_time_seconds, "秒")} → ${formatMetric(measurement.end_time_seconds, "秒")}`]);
  }
  const dropDepth = testCase.observations?.maximum_ground_penetration;
  const dropHeight = testCase.observations?.first_rebound_height;
  const hasDropSamples = !slope && Number.isFinite(dropDepth?.sample_interval_seconds);
  if (hasDropSamples) {
    rows.push(["采样间隔", formatMetric(dropDepth.sample_interval_seconds * 1000, "毫秒")],
      ["测量时窗", `${formatMetric(dropDepth.start_time_seconds, "秒")} → ${formatMetric(dropDepth.end_time_seconds, "秒")}`]);
    if (dropDepth.status === "measured" && dropDepth.value > 0) rows.push(["最大穿透发生于", formatMetric(dropDepth.peak_time_seconds, "秒")]);
    if (dropHeight?.status === "measured") rows.push(["首次回弹峰值发生于", formatMetric(dropHeight.peak_time_seconds, "秒")]);
  }
  const list = element("dl", "metric-list");
  rows.forEach(([label, value]) => list.append(element("dt", "", label), element("dd", "", value)));
  section.append(list);
  section.append(element("p", "measurement-note", slope
    ? (measurement?.status === "measured"
      ? "取单刚体坐标原点起末位置差，投影到下坡方向；正值向下坡，负值向上坡。它是净位移，不是累计路程，也不能区分滑动、滚动或翻倒。"
      : "旧记录若有位移会保留显示；未记录的测量点、方向和采样时窗不补推。")
    : (hasDropSamples
      ? "回弹取首次整体离地至再次触地期间的最大净空；穿透取整个观察时窗内的最大几何穿入深度。数值为采样点极值，不包含步间运动；不大于 0.01 毫米的净空按触地处理，不是质量合格阈值。"
      : "这次记录未提供可用的回弹和穿透测量。未测量不等于零，请结合录像观察。")));
  section.append(element("h3", "", "看录像时关注"), element("p", "", slope
    ? "是否开始运动，以滑动、滚动还是翻倒为主；怎样减速，是否在观察期内停止。单次位移不能证明摩擦参数合理。"
    : "首次回弹、碰撞后翻滚，是否穿地或弹飞；随后是否持续抖动、反复弹跳或下沉。录像结束不等于已经稳定。"));
  return section;
}

function testConditions(run, testCase) {
  const section = element("section", "inspection-panel");
  section.append(element("h3", "", "试验条件"));
  const condition = testCase.condition || {};
  const rows = [["资产版本", run.asset_version ? run.asset_version.slice(0, 12) : "未记录"]];
  if (run.template === "drop") {
    const levels = {low: "低档", medium: "中档", high: "高档"};
    if (condition.case_scope === "three_height_drop_validation_v1") {
      rows.push(["释放档位", `${levels[condition.height_level] || "未记录"} · ${formatMetric(condition.clearance_scale, "倍场景缩放基准")}`]);
      rows.push(["参考面", condition.reference_policy === "asset_priority_over_ground_v1"
        ? "中性水平面，资产接触参数优先" : "未记录"]);
      if (condition.scale_reference?.kind === "visual_only_meter_ruler_and_square_v1") {
        rows.push(["画面尺度参照", `标尺数字单位为米；地面方框边长 ${lengthLabel(condition.scale_reference.square_side_m)}`]);
      }
    }
    rows.push(["设置的释放净空", lengthLabel(condition.clearance_m)]);
    const actual = testCase.observations?.maximum_ground_penetration?.initial_clearance_m;
    rows.push(["按碰撞几何测得的释放净空", lengthLabel(actual)]);
  } else {
    rows.push(["坡度", formatMetric(condition.slope_angle_degrees ?? testCase.slope_angle_degrees, "°", 1)]);
  }
  rows.push(["资产特征尺寸", lengthLabel(condition.characteristic_length_m)],
    ["场景缩放基准", lengthLabel(condition.effective_length_m)],
    ["物理观察时长（配置）", formatMetric(testCase.duration_seconds, "秒")],
    ["录像时长（含初始展示）", formatMetric(testCase.video_duration_seconds, "秒")],
    ["初始线速度", formatVector(condition.initial_velocity_mps, "m/s")],
    ["初始角速度", formatVector(condition.initial_angular_velocity_rps, "rad/s")]);
  const list = element("dl", "metric-list");
  rows.forEach(([label, value]) => list.append(element("dt", "", label), element("dd", "", value)));
  section.append(list);
  if (run.template === "drop" && isFunctionTest(run, testCase)
      && !Number.isFinite(testCase.observations?.maximum_ground_penetration?.initial_clearance_m)) {
    section.append(element("p", "measurement-note", "释放距离为配置值，实际离地距离尚未校验。"));
  }
  return section;
}

const PARAMETER_LABELS = {
  "mjc:solref": "接触恢复与阻尼",
  "mjc:solimp": "接触约束随穿透的变化",
  "physics:dynamicFriction": "滑动摩擦",
  "mjc:rollingfriction": "滚动摩擦",
};

function parameterPanel(run, testCase = null) {
  const section = element("section", "inspection-panel parameter-panel");
  section.append(element("h3", "", "待检查参数与依据"));
  const fields = run.template === "drop" ? ["mjc:solref", "mjc:solimp"]
    : ["physics:dynamicFriction", "mjc:rollingfriction"];
  const snapshot = run.physics_parameters;
  const records = snapshot?.schema_version === 1 && Array.isArray(snapshot.records)
    ? snapshot.records.filter((record) => record && typeof record === "object") : [];
  const list = element("dl", "metric-list");
  fields.forEach((field) => {
    const entries = records.filter((record) => record.field === field);
    const values = [...new Set(entries.filter((record) => record.status === "recorded")
      .map((record) => formatConditionValue(record.value)))];
    const missing = entries.filter((record) => record.status !== "recorded").length;
    const value = entries.length ? (values.length ? values.join("；") : "未取得常量标注值") : "未记录";
    list.append(element("dt", "", `${PARAMETER_LABELS[field]} · ${field}`),
      element("dd", "", `${value}${missing ? `（${missing} 个碰撞形状的值缺失或不可用）` : ""}`));
  });
  if (!records.length) {
    section.append(list, element("p", "measurement-note", "这次运行没有参数快照和完整参考面条件，无法据此核对参数是否生效；历史记录不会用当前参数补填。"));
    return section;
  }
  list.append(element("dt", "", "参数来源"), element("dd", "", "该版本 USD 标注；原始标注或测试补充的来源未细分"));
  list.append(element("dt", "", "实际接触参数"), element("dd", "", "未记录求解器最终接触值，不能把 USD 标注当作已验证生效值"));
  const policy = testCase?.contact_policy;
  list.append(element("dt", "", "参考面条件"), element("dd", "", policy?.asset_controls_contact_parameters === true
    && Number.isInteger(policy.reference_surface_priority)
    ? `已记录参考面优先级 ${policy.reference_surface_priority}；配置意图由资产控制接触，实际接触值仍待核对`
    : "本次记录未提供完整参考面接触条件"));
  section.append(list);
  if (records.length) {
    const details = element("details", "parameter-locations");
    details.append(element("summary", "", "查看各碰撞形状的标注位置"));
    const locations = element("ul");
    records.filter((record) => fields.includes(record.field)).forEach((record) => {
      locations.append(element("li", "", `${record.collision_prim} · ${record.field}：${record.status === "recorded" ? formatConditionValue(record.value) : "未取得常量标注值"} · 标注位置 ${record.value_prim || "未绑定物理材质"}`));
    });
    details.append(locations);
    section.append(details);
  }
  return section;
}

function coveragePanel(asset) {
  const section = element("section", "coverage-panel");
  section.append(element("h2", "", "检查覆盖范围"));
  section.append(element("p", "", "最终需要：低、中、高三档摔落；缓、中、陡坡与水平滑行；同条件下与其他资产及标准参照物比较。"));
  const functionOnly = asset.runs.every((run) => isFunctionTest(run));
  section.append(element("p", "", functionOnly
    ? "当前只有功能测试记录，正式多工况尚未开放；不能据此视为完整物理检查。"
    : "请逐项核对版本与工况；现有记录尚未提供完整基线覆盖清单，不能确认最终检查已完成。"));
  const list = element("ul", "coverage-list");
  ["drop", "slope_friction"].forEach((template) => {
    const runs = asset.runs.filter((run) => run.template === template);
    const count = runs.reduce((total, run) => total + run.cases.length, 0);
    list.append(element("li", "", `${TEMPLATE_LABELS[template]}：${count ? `${count} 项工况记录（包含历史或未完成项）` : "暂无运行记录"}`));
  });
  list.append(element("li", "", "水平滑行与标准参照物比较：尚无可核查的覆盖记录"));
  section.append(list);
  dropValidationGroups(asset).forEach((group) => {
    section.append(element("h3", "", `三档摔落开发验证 · 版本 ${group.version.slice(0, 12)}`));
    const items = element("ul", "coverage-list");
    for (const [level, label] of [["low", "低档"], ["medium", "中档"], ["high", "高档"]]) {
      const record = group.cases[level];
      const item = element("li", "", `${label}：${record ? (record.complete ? "已完成录像，待人工检查" : STATUS_LABELS[record.testCase.status] === "已完成" ? "记录不完整" : STATUS_LABELS[record.testCase.status] || "状态未知") : "尚未运行"}`);
      if (record?.testCase.video_url) {
        const link = element("a", "", " 查看录像");
        link.href = routeHref({asset: asset.identity, play: record.run.run_id, case: record.testCase.case_id});
        item.append(link);
      }
      items.append(item);
    }
    section.append(items, element("p", "measurement-note", `同一资产版本、源码 ${group.commit.slice(0, 12)} 与运行配置分组；每档观察 10 秒。开发覆盖不等于正式验收。`));
  });
  return section;
}

function dropValidationGroups(asset) {
  const groups = new Map();
  const scales = {low: .5, medium: 1, high: 2};
  for (const run of asset.runs) {
    if (run.template !== "drop" || run.profile_name !== "mujoco-warp-cuda-drop-validation-10s-v1"
        || !run.asset_version || !run.git_commit) continue;
    for (const testCase of run.cases) {
      const condition = testCase.condition || {};
      const level = condition.height_level;
      if (condition.case_scope !== "three_height_drop_validation_v1"
          || !Object.hasOwn(scales, level) || condition.clearance_scale !== scales[level]) continue;
      const key = JSON.stringify([run.asset_version, run.git_commit, run.profile_name, condition.reference_policy]);
      if (!groups.has(key)) groups.set(key, {version: run.asset_version, commit: run.git_commit, cases: {}});
      const group = groups.get(key);
      if (group.cases[level]) continue; // Index runs are newest first; retain latest attempt.
      group.cases[level] = {run, testCase, complete: testCase.status === "succeeded"
        && testCase.finite === true && testCase.physics_steps === 10000
        && testCase.duration_seconds === 10 && testCase.video_duration_seconds === 10.5
        && !!testCase.video_url && condition.reference_policy === "asset_priority_over_ground_v1"};
    }
  }
  return [...groups.values()];
}

function assetName(asset) {
  const name = asset.display_name || asset.identity.split("/").pop();
  // Translate known names only; never invent a category for an unknown asset.
  const translations = { scissors: "剪刀", "blue-box": "蓝色方块" };
  const base = name.replace(/-[0-9a-f]{8,64}$/i, "");
  return translations[base] || name;
}

function assetSource(asset) {
  const parts = asset.identity.split("/");
  const source = parts.length > 1 ? (parts[0] === "fixtures" ? "内置测试样本" : parts[0]) : "来源未标注";
  const suffix = parts[parts.length - 1].match(/-([0-9a-f]{8,64})$/i);
  return suffix ? `${source} · 样本 ${suffix[1].slice(0, 8)}` : source;
}

function isFunctionTest(run, testCase = {}) {
  return run.authoritative === false || testCase.authoritative === false;
}

function caseTitle(run, testCase) {
  if (testCase.condition?.case_scope === "three_height_drop_validation_v1") {
    const level = {low: "低档", medium: "中档", high: "高档"}[testCase.condition.height_level];
    return `${level || "未标注档位"}摔落 · 开发验证`;
  }
  const titles = isFunctionTest(run, testCase)
    ? { drop: "摔落功能测试", slope_friction: "斜坡功能测试" }
    : TEMPLATE_LABELS;
  return titles[run.template] || "测试记录";
}

function lengthLabel(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "未记录";
  return Math.abs(value) < 1 && value !== 0
    ? formatMetric(value * 100, "厘米", 2)
    : formatMetric(value, "米");
}

function caseSummary(testCase) {
  if (testCase.finite === false) {
    return "计算中出现无效数值，请查看技术详情；本次结果不能用于判断运动是否正常。";
  }
  const summaries = {
    created: "测试尚未开始。",
    running: "测试正在进行，结果尚未完整。",
    failed: "本次运行未完成，请展开技术详情查看记录。",
    interrupted: "本次运行已中断，现有录像可能不完整。",
    cancelled: "本次运行已取消。",
    partially_succeeded: "部分测试已完成，请分别查看每项结果。",
  };
  if (testCase.status === "succeeded") {
    return testCase.video_url ? "测试已完成，可以播放录像。" : "测试已完成，本次未提供录像。";
  }
  return summaries[testCase.status] || "完成状态尚未记录。";
}

function technicalDetails(asset, run, testCase = null) {
  const details = element("details", "technical-details");
  details.append(element("summary", "", "详情"));
  const facts = element("dl", "metric-list");
  const rows = [
    ["资产标识", asset.identity],
    ["运行编号", run.run_id],
    ["资产版本", run.asset_version],
    ["运行配置", run.profile_name || "未记录"],
    ["源码版本", run.git_commit || "未记录"],
  ];
  if (testCase) {
    rows.push(
      ["工况编号", testCase.case_id],
      ["原始工况名称", testCase.label || "未记录"],
      ["计算步数", formatFiniteNumber(testCase.physics_steps, 0)],
      ["数值检查", formatFiniteState(testCase.finite)],
    );
    if (run.template === "slope_friction") {
      rows.push(["最终速度（X、Y、Z）", formatVector(testCase.final_linear_velocity, "m/s")]);
      rows.push(["旧版移动标签（非质量结论）", DEVELOPMENT_OUTCOME_LABELS[testCase.development_outcome] || "未记录"]);
    }
  }
  rows.forEach(([label, value]) => facts.append(element("dt", "", label), element("dd", "", value || "未记录")));
  details.append(facts);
  if (testCase && testCase.condition && Object.keys(testCase.condition).length) {
    details.append(element("h3", "", "原始测试参数"));
    const parameters = element("dl", "metric-list");
    Object.entries(testCase.condition).forEach(([key, value]) => {
      parameters.append(element("dt", "", key), element("dd", "", formatConditionValue(value)));
    });
    details.append(parameters);
  }
  if (run.failure_summary || run.progress) {
    details.append(element("h3", "", "运行记录"), element("p", "raw-record", run.failure_summary || run.progress));
  }
  if (run.checksums_url) {
    const checksums = element("a", "", "查看文件校验清单");
    checksums.href = run.checksums_url;
    details.append(checksums);
  }
  return details;
}

function statusBadge(status) {
  const badge = element("span", "status", STATUS_LABELS[status] || "状态未知");
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
  const latest = asset.runs[0];
  const cover = asset.cover_url || latest?.cases.find((item) => item.poster_url)?.poster_url || latest?.preview_url;
  if (cover) {
    const image = element("img");
    image.src = cover;
    image.alt = `${assetName(asset)} 的测试画面`;
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
    element("p", "eyebrow", "测试记录"),
    element("h1", "", "查看资产测试"),
    element(
      "p",
      "page-description",
      "选择一个资产，查看测试结果和录像。",
    ),
  );
  const updatedAt = resultIndex.generated_at ? `记录更新于 ${formatTime(resultIndex.generated_at)}` : "更新时间未记录";
  heading.append(headingCopy, element("p", "refresh-note", updatedAt));
  fragment.append(heading);

  if (!resultIndex.assets.length) {
    const empty = element("section", "empty-state");
    empty.append(
      element("h1", "", "还没有测试记录"),
      element(
        "p",
        "",
        "测试结果发布后，会显示在这里。",
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
    coverLink.setAttribute("aria-label", `查看${assetName(asset)}的测试 · ${assetSource(asset)} · ${STATUS_LABELS[asset.latest_status] || "状态未知"}`);
    coverLink.append(coverNode(asset));
    const overlay = element("span", "cover-overlay");
    overlay.append(
      element("span", "", `${asset.runs.length} 次测试`),
      statusBadge(asset.latest_status),
    );
    coverLink.append(overlay);

    const copy = element("div", "asset-copy");
    copy.append(
      element("h2", "asset-title", assetName(asset)),
      element("p", "asset-source", assetSource(asset)),
    );
    const tags = element("div", "tag-row");
    asset.templates.forEach((template) => {
      tags.append(element("span", "tag", TEMPLATE_LABELS[template] || "其他测试"));
    });
    copy.append(tags);
    card.append(coverLink, copy);
    grid.append(card);
  });
  fragment.append(grid);
  app.replaceChildren(fragment);
}

function describeCondition(condition, duration) {
  const values = condition && typeof condition === "object" ? condition : {};
  const parts = [];
  if (Number.isFinite(values.clearance_m)) parts.push(`起始时距地面 ${lengthLabel(values.clearance_m)}`);
  if (Number.isFinite(values.slope_angle_degrees)) parts.push(`斜坡角度 ${formatMetric(values.slope_angle_degrees, "°", 1)}`);
  if (Number.isFinite(duration)) parts.push(`模拟时长 ${formatMetric(duration, "秒")}`);
  return parts.length ? parts.join(" · ") : "测试条件未记录，可展开技术详情查看已有信息。";
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
    image.alt = `${assetName(asset)}的${caseTitle(run, testCase)}画面`;
    image.loading = "lazy";
    media.append(image);
  } else {
    media.append(element("div", "case-media-placeholder", "暂无录像"));
  }

  const copy = element("div", "case-copy");
  copy.append(
    element("h3", "", caseTitle(run, testCase)),
    element("p", "case-outcome", caseSummary(testCase)),
    element("p", "", describeCondition(testCase.condition, testCase.duration_seconds)),
  );
  const nonAuthoritative = isFunctionTest(run, testCase);
  if (nonAuthoritative) copy.append(nonAuthoritativeNotice());
  copy.append(testConditions(run, testCase), observationPanel(run, testCase));
  const tags = element("div", "tag-row");
  tags.append(statusBadge(testCase.status));
  copy.append(tags);

  if (testCase.video_url) {
    const actions = element("div", "case-actions");
    const playerLink = element("a", "primary-action", "播放录像");
    playerLink.href = routeHref({
      play: run.run_id,
      case: testCase.case_id,
      asset: asset.identity,
    });
    const downloadLink = element("a", "", "下载录像");
    downloadLink.href = testCase.video_url;
    downloadLink.download = "";
    actions.append(playerLink, downloadLink);
    copy.append(actions);
  }
  copy.append(technicalDetails(asset, run, testCase));
  card.append(media, copy);
  return card;
}

function renderAsset(identity) {
  const asset = resultIndex.assets.find((item) => item.identity === identity);
  if (!asset) {
    renderNotFound("这项资产的测试记录暂时不可用，请返回资产列表查看。");
    return;
  }

  const fragment = document.createDocumentFragment();
  const back = element("a", "back-link", "← 返回全部资产");
  back.href = "#";
  fragment.append(back);

  const heading = element("section", "page-heading");
  const headingCopy = element("div");
  headingCopy.append(
    element("p", "eyebrow", "资产测试"),
    element("h1", "", assetName(asset)),
    element("p", "page-description", assetSource(asset)),
  );
  heading.append(headingCopy, statusBadge(asset.latest_status));
  fragment.append(heading);

  fragment.append(coveragePanel(asset));
  asset.templates.forEach((template) => {
    const runs = asset.runs.filter((run) => run.template === template);
    const section = element("section", "experiment-section");
    const title = element("h2", "section-title");
    title.append(
      document.createTextNode(TEMPLATE_LABELS[template] || "其他测试"),
      element("span", "tag", `${runs.length} 次测试`),
    );
    section.append(title);

    runs.forEach((run, index) => {
      const details = element("details", "run-block");
      if (index === 0) details.open = true;
      const summary = element("summary");
      const summaryMain = element("div", "run-summary-main");
      summaryMain.append(
        element("strong", "", index === 0 ? "最近一次测试" : "历史测试"),
        element(
          "span",
          "run-meta",
          `${formatTime(run.created_at)} · ${run.cases.length} 项测试记录`,
        ),
      );
      summary.append(summaryMain, statusBadge(run.status));
      details.append(summary);

      const caseGrid = element("div", "case-grid");
      if (run.cases.length) {
        run.cases.forEach((testCase) => caseGrid.append(caseCard(asset, run, testCase)));
      } else {
        caseGrid.append(element("p", "page-description", caseSummary(run)), technicalDetails(asset, run));
      }
      details.append(parameterPanel(run, run.cases[0]), caseGrid);
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
    renderNotFound("这段录像暂时不可用，请返回资产列表查看其他记录。");
    return;
  }
  const { asset, run, testCase } = found;
  const fragment = document.createDocumentFragment();
  const back = element("a", "back-link", `← 返回${assetName(asset)}的测试记录`);
  back.href = routeHref({ asset: asset.identity });
  fragment.append(back);

  const heading = element("section", "page-heading");
  const headingCopy = element("div");
  headingCopy.append(
    element("p", "eyebrow", assetSource(asset)),
    element("h1", "", `${assetName(asset)} · ${caseTitle(run, testCase)}`),
    element("p", "page-description", caseSummary(testCase)),
  );
  heading.append(headingCopy, statusBadge(testCase.status));
  fragment.append(heading);

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
    element("h2", "", "本次测试"),
    element("p", "page-description", describeCondition(testCase.condition, testCase.duration_seconds)),
    element("p", "completion-time", `完成时间：${formatTime(testCase.finished_at)}`),
  );
  if (isFunctionTest(run, testCase)) {
    details.append(nonAuthoritativeNotice());
  }
  details.append(testConditions(run, testCase), observationPanel(run, testCase), parameterPanel(run, testCase));
  const download = element("a", "download-action", "下载录像");
  download.href = testCase.video_url;
  download.download = "";
  details.append(download, technicalDetails(asset, run, testCase));
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
      "与结果服务的连接暂时不可用，或返回的数据不完整。请确认服务器上的结果服务和本地连接窗口仍在运行，然后重试。",
    ),
  );
  const retry = element("button", "primary-action", "重新加载");
  retry.type = "button";
  retry.addEventListener("click", () => loadIndex().then(scheduleRefresh).catch(renderError));
  const technical = element("details", "technical-details");
  technical.append(element("summary", "", "技术详情"), element("p", "raw-record", String(error.message || error)));
  box.append(retry, technical);
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

async function loadIndex({ rerender = true, onlyIfChanged = false } = {}) {
  const response = await fetch(`index.json?t=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const nextIndex = await response.json();
  if (!nextIndex || !Array.isArray(nextIndex.assets)) {
    throw new Error("index.json 格式不正确");
  }
  const changed = JSON.stringify(resultIndex) !== JSON.stringify(nextIndex);
  resultIndex = nextIndex;
  if (rerender && (!onlyIfChanged || changed)) renderRoute();
}

function scheduleRefresh() {
  window.clearInterval(refreshTimer);
  refreshTimer = window.setInterval(async () => {
    const videoIsPlaying = Array.from(document.querySelectorAll("video")).some(
      (video) => !video.paused && !video.ended,
    );
    if (videoIsPlaying) return;
    try {
      await loadIndex({ rerender: true, onlyIfChanged: true });
    } catch (error) {
      console.warn("Result index refresh failed", error);
    }
  }, 10000);
}

window.addEventListener("hashchange", () => {
  renderRoute();
  window.scrollTo({ top: 0, left: 0, behavior: "auto" });
});
loadIndex()
  .then(scheduleRefresh)
  .catch(renderError);

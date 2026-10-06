// Pure presentation tests. Browser layout is checked separately using local fixtures.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({
  document: { querySelector: () => ({}), createElement: tag => ({
    tag, children: [], textContent: "", append(...children) { this.children.push(...children); },
  }) },
  window: { addEventListener: () => {} },
  fetch: () => new Promise(() => {}),
  URLSearchParams, Intl, console,
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../experiment_runner/web_static/app.js"), "utf8"), context);
const metric = (overrides = {}) => ({status: "succeeded", finite: true, observations: {
  along_slope_displacement: {status: "measured", value: 0, unit: "m"},
}, ...overrides});
const display = (data) => context.observationValue(data, "along_slope_displacement");

test("zero, unmeasured and invalid remain distinguishable", () => {
  assert.equal(display(metric()), "0 米");
  assert.equal(display(metric({observations: {}})), "未测量");
  assert.equal(display(metric({finite: false})), "数值异常，不能解释");
});
test("unfinished run never displays a leftover successful measurement", () => {
  for (const status of ["running", "failed", "interrupted", "cancelled"]) {
    assert.equal(display(metric({status})), "观察未完成");
  }
  assert.equal(display(metric({status: "created"})), "尚未运行");
});
test("legacy values are preserved only when validity is recorded", () => {
  const old = {status: "succeeded", finite: true, displacement_along_slope: -0.25};
  assert.equal(display(old), "-25 厘米");
  delete old.finite;
  assert.equal(display(old), "未测量");
});
test("invalid new record never falls back to legacy value", () => {
  const data = metric({displacement_along_slope: 100, observations: {
    along_slope_displacement: {status: "invalid", value: null},
  }});
  assert.equal(display(data), "测量记录无效");
});
test("non-finite values and wrong units are not displayed as measurements", () => {
  for (const changes of [{value: Infinity}, {value: NaN}, {value: null}, {value: "0"}, {unit: "cm"}]) {
    const data = metric();
    Object.assign(data.observations.along_slope_displacement, changes);
    assert.equal(display(data), "未测量");
  }
});
test("drop ratios use percent and fine penetration does not round to zero", () => {
  const data = {status: "succeeded", finite: true, observations: {
    first_rebound_ratio: {status: "measured", unit: "1", value: .4},
    maximum_ground_penetration: {status: "measured", unit: "m", value: .000001},
  }};
  assert.equal(context.observationValue(data, "first_rebound_ratio"), "40 %");
  assert.equal(context.observationValue(data, "maximum_ground_penetration"), "0.001 毫米");
});
test("no impact, no liftoff and unfinished rebound never look like a zero rebound", () => {
  for (const [status, label] of [["not_observed", "观察期内未触地"],
    ["no_liftoff", "触地后未观察到整体离地"], ["incomplete", "观察未完成"],
    ["unsupported", "当前场景暂不支持测量"]]) {
    const data = {status: "succeeded", finite: true, observations: {
      first_rebound_height: {status, value: null, unit: "m"},
    }};
    assert.equal(context.observationValue(data, "first_rebound_height"), label);
  }
});

function validationRun(level, overrides = {}) {
  return {template:"drop", profile_name:"mujoco-warp-cuda-drop-validation-10s-v1",
    asset_version:"a".repeat(64), git_commit:"b".repeat(40), run_id:`run-${level}`,
    cases:[{case_id:`drop-${level}-validation`, status:"succeeded", finite:true,
      physics_steps:10000, duration_seconds:10, video_duration_seconds:10.5, video_url:"/fixture.mp4",
      condition:{height_level:level, clearance_scale:{low:.5,medium:1,high:2}[level],
        case_scope:"three_height_drop_validation_v1", reference_policy:"asset_priority_over_ground_v1"}}],
    ...overrides};
}
function fixedValidationRun(overrides = {}) {
  const run = validationRun("fixed-1m", overrides);
  Object.assign(run.cases[0].condition, {case_scope:"fixed_1m_drop_validation_v1",
    clearance_scale:null, fixed_clearance_m:1, clearance_m:1, actual_initial_clearance_m:1});
  return run;
}
test("fixed one-metre drop is separate from a one-times relative drop", () => {
  const fixed = fixedValidationRun();
  const [group] = context.dropValidationGroups({runs:[fixed, validationRun("medium")]});
  assert.equal(group.cases["fixed-1m"].complete, true);
  assert.equal(group.cases.medium.complete, true);
  assert.equal(group.cases.low, undefined);
  assert.match(context.caseTitle(fixed, fixed.cases[0]), /固定 1 米摔落/);
});
test("condition and coverage panels show fixed metres and missing fixed coverage", () => {
  const textOf = node => [node.textContent, ...node.children.map(textOf)].join(" ");
  const fixed = fixedValidationRun();
  const conditions = textOf(context.testConditions(fixed, fixed.cases[0]));
  assert.match(conditions, /固定高度 · 1 米/);
  assert.doesNotMatch(conditions, /倍场景缩放基准/);
  const asset = {identity:"fixtures/box", runs:[validationRun("low")]};
  assert.match(textOf(context.coveragePanel(asset)), /固定 1 米：尚未运行/);
  asset.runs.unshift(fixed);
  assert.match(textOf(context.coveragePanel(asset)), /固定 1 米：已完成录像，待人工检查/);
});
test("fixed drop needs verified one-metre clearance and cannot fill relative coverage", () => {
  for (const actual of [undefined, null, 0.1, NaN]) {
    const fixed = fixedValidationRun();
    fixed.cases[0].condition.actual_initial_clearance_m = actual;
    const [group] = context.dropValidationGroups({runs:[fixed]});
    assert.equal(group.cases["fixed-1m"].complete, false);
    assert.equal(group.cases.medium, undefined);
  }
  const wrongScale = fixedValidationRun();
  wrongScale.cases[0].condition.clearance_scale = 1;
  assert.equal(context.dropValidationGroups({runs:[wrongScale]}).length, 0);
});
test("fixed drop keeps latest failure and separates source versions", () => {
  const failed = fixedValidationRun();
  failed.cases[0].status = "failed";
  const groups = context.dropValidationGroups({runs:[failed, fixedValidationRun(),
    fixedValidationRun({git_commit:"c".repeat(40)})]});
  assert.equal(groups.length, 2);
  assert.equal(groups[0].cases["fixed-1m"].complete, false);
  assert.equal(groups[1].cases["fixed-1m"].complete, true);
});
test("drop coverage groups matching versions and code, retaining missing levels", () => {
  const runs = [validationRun("low"), validationRun("medium"), validationRun("high", {asset_version:"c".repeat(64)})];
  const groups = context.dropValidationGroups({runs});
  assert.equal(groups.length, 2);
  assert.equal(groups[0].cases.low.complete, true);
  assert.equal(groups[0].cases.high, undefined);
  assert.equal(groups[1].cases.high.complete, true);
  assert.match(context.caseTitle(runs[0], runs[0].cases[0]), /低档/);
});
test("old smoke, failed latest attempts and incomplete records cannot fill coverage", () => {
  const failed = validationRun("low");
  failed.cases[0].status = "failed";
  const short = validationRun("medium");
  short.cases[0].physics_steps = 1000;
  const old = validationRun("high", {profile_name:"mujoco-warp-cuda-dt1ms-video-smoke-v1"});
  const [group] = context.dropValidationGroups({runs:[failed, validationRun("low"), short, old]});
  assert.equal(group.cases.low.complete, false);
  assert.equal(group.cases.medium.complete, false);
  assert.equal(group.cases.high, undefined);
});
test("code or reference-condition changes remain separate comparison groups", () => {
  const changed = validationRun("high");
  changed.cases[0].condition.reference_policy = "unknown";
  const groups = context.dropValidationGroups({runs:[validationRun("low"),
    validationRun("medium", {git_commit:"c".repeat(40)}), changed]});
  assert.equal(groups.length, 3);
  assert.equal(groups[2].cases.high.complete, false);
});

test("external input verification requires all three recorded checks", () => {
  const verified = {status:"verified", stage:"after_run",
    checks:["before_load", "after_load", "after_run"].map(stage => ({stage}))};
  assert.match(context.assetInputVerificationLabel({asset_input_verification:verified}), /均与所选版本一致/);
  for (const partial of [undefined, {...verified, checks:[]}, {...verified, stage:"after_load"}]) {
    assert.equal(context.assetInputVerificationLabel({asset_input_verification:partial}), "未完成全部复核");
  }
  assert.match(context.assetInputVerificationLabel({asset_input_verification:{status:"failed"}}), /不可归属于所选版本/);
});

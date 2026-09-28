// Pure presentation tests. Browser layout is checked separately using local fixtures.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({
  document: { querySelector: () => ({}) },
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

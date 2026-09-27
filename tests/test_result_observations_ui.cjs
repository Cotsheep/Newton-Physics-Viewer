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

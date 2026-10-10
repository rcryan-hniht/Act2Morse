const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

test('closed fist requires 600ms, triggers once, and resets after release', () => {
  const source = fs.readFileSync(`${__dirname}/../src/camera.ts`, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  });
  const sandbox = {
    exports: {}, require: () => ({}), performance,
    document: { createElement: () => ({ getContext: () => null }) },
  };
  vm.runInNewContext(outputText, sandbox);
  const camera = sandbox.exports.cameraController;
  const events = [];
  camera.onGestureAction(event => events.push(event.action));
  const ctx = new Proxy({}, { get: () => () => {} });
  const landmarks = Array.from({ length: 21 }, (_, i) => ({ x: i / 25, y: 0.5, z: 0 }));
  const frame = fist => ({
    landmarks: [landmarks],
    gestures: [[{ categoryName: fist ? 'Closed_Fist' : 'None', score: 1 }]],
    handedness: [[{ categoryName: 'Left', score: 1 }]],
  });
  const render = (time, fist = true) => camera.renderHandGestureResult(ctx, frame(fist), 640, 480, 1, time);
  render(1000);
  render(1260);
  render(1599);
  assert.deepEqual(events, []);
  render(1600);
  render(2000);
  assert.deepEqual(events, ['space']);
  render(2100, false);
  render(2200);
  render(2799);
  assert.deepEqual(events, ['space']);
  render(2800);
  assert.deepEqual(events, ['space', 'space']);
});

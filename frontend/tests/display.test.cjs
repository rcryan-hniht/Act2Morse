const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

test('completed characters appear for 1s without restarting on repeated frames', () => {
  const source = fs.readFileSync(`${__dirname}/../src/main.ts`, 'utf8');
  const start = source.indexOf('const cameraCharacter =');
  const end = source.indexOf('function pulseFloatingIcon()', start);
  const { outputText } = ts.transpileModule(source.slice(start, end), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const character = { hidden: true, textContent: '' };
  const line = { textContent: '', scrollWidth: 100 };
  const timers = new Map();
  let nextId = 0;
  const sandbox = {
    decodedText: '', decodedTextDisplay: line,
    document: { getElementById: () => character },
    window: {
      setTimeout: (callback, ms) => {
        assert.equal(ms, 1000);
        timers.set(++nextId, callback);
        return nextId;
      },
      clearTimeout: id => timers.delete(id),
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(outputText, sandbox);
  const update = text => {
    sandbox.decodedText = text;
    vm.runInContext('updateDisplay()', sandbox);
  };
  update('A');
  assert.equal(line.textContent, 'A');
  assert.equal(character.textContent, 'A');
  assert.equal(character.hidden, false);
  update('A');
  assert.equal(nextId, 1);
  update('AB');
  assert.equal(character.textContent, 'B');
  assert.equal(timers.has(1), false);
  timers.get(2)();
  assert.equal(character.hidden, true);
  update('AB ');
  assert.equal(nextId, 2);
  update('AB C');
  assert.equal(character.hidden, false);
  update('AB ');
  assert.equal(character.hidden, true);
  update('');
  assert.equal(line.textContent, 'Waiting for Morse...');
});

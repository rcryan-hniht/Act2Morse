const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

test('disconnect cancels queued retries and allows a fresh connection', () => {
  const source = fs.readFileSync(`${__dirname}/../src/ws.ts`, 'utf8').replace('import.meta.env.VITE_WS_URL', 'undefined');
  const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } });
  const sockets = [], timers = new Map();
  class Socket {
    static OPEN = 1;
    readyState = 0;
    constructor() { sockets.push(this); }
    close() { this.readyState = 3; }
  }
  const sandbox = { exports: {}, WebSocket: Socket,
    window: { location: { protocol: 'http:', hostname: 'localhost' } },
    setTimeout: cb => { timers.set(1, cb); return 1; },
    clearTimeout: id => timers.delete(id),
  };
  vm.runInNewContext(outputText, sandbox);
  const bridge = sandbox.exports.wsBridge;
  assert.equal(sockets.length, 0);
  bridge.connect();
  bridge.connect();
  assert.equal(sockets.length, 1);
  sockets[0].readyState = 3;
  sockets[0].onclose();
  const retry = timers.get(1);
  bridge.disconnect();
  assert.equal(timers.size, 0);
  retry();
  assert.equal(sockets.length, 1);
  bridge.connect();
  sockets[1].readyState = 1;
  sockets[1].onopen();
  assert.equal(bridge.isConnected(), true);
  bridge.disconnect();
  assert.equal(bridge.isConnected(), false);
  assert.equal(sockets[1].onopen, null);
  assert.equal(sockets[1].onclose, null);
  bridge.connect();
  assert.equal(sockets.length, 3);
});

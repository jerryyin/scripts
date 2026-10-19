// Exercise a real native clangd with the flags from the shared CoC settings.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'devenv-clangd-'));
const pending = new Map();
let child, incoming = Buffer.alloc(0), sequence = 0, stderr = '';
function send(message) {
  const body = Buffer.from(JSON.stringify({ jsonrpc: '2.0', ...message }));
  child.stdin.write(`Content-Length: ${body.length}\r\n\r\n`);
  child.stdin.write(body);
}
function request(method, params) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => reject(Error(`${method} timed out\n${stderr}`)), 20000);
    pending.set(id, { resolve, reject, timer });
    send({ id, method, params });
  });
}
try {
  const source = 'struct Point { int x; };\nint twice(Point p) { return p.x * 2; }\n';
  const file = path.join(fixture, 'sample.cpp');
  fs.writeFileSync(file, source);
  child = spawn(job.clangd, job.arguments, { cwd: fixture, windowsHide: true });
  const closed = new Promise(resolve => child.once('close', resolve));
  child.on('error', error => {
    for (const entry of pending.values()) { clearTimeout(entry.timer); entry.reject(error); }
  });
  child.stderr.on('data', bytes => { stderr = (stderr + bytes).slice(-12000); });
  child.stdout.on('data', bytes => {
    incoming = Buffer.concat([incoming, bytes]);
    while (true) {
      const end = incoming.indexOf('\r\n\r\n');
      if (end < 0) break;
      const length = Number(/Content-Length:\s*(\d+)/i.exec(incoming.subarray(0, end).toString())[1]);
      if (incoming.length < end + 4 + length) break;
      const message = JSON.parse(incoming.subarray(end + 4, end + 4 + length));
      incoming = incoming.subarray(end + 4 + length);
      if (message.method && message.id !== undefined) {
        send({ id: message.id, result: message.method === 'workspace/configuration' ? message.params.items.map(() => null) : null });
      } else if (pending.has(message.id)) {
        const entry = pending.get(message.id);
        pending.delete(message.id);
        clearTimeout(entry.timer);
        if (message.error) entry.reject(Error(JSON.stringify(message.error)));
        else entry.resolve(message.result);
      }
    }
  });
  const initialized = await request('initialize', {
    processId: process.pid,
    rootUri: pathToFileURL(fixture).href,
    capabilities: { textDocument: { semanticTokens: {
      requests: { full: true }, formats: ['relative'],
      tokenTypes: ['namespace', 'type', 'class', 'struct', 'parameter', 'variable', 'property', 'function', 'method'],
      tokenModifiers: ['declaration', 'definition', 'readonly', 'static'],
    } } },
  });
  assert.ok(initialized.capabilities.semanticTokensProvider, 'clangd advertises semantic token support');
  send({ method: 'initialized', params: {} });
  send({ method: 'textDocument/didOpen', params: { textDocument: { uri: pathToFileURL(file).href, languageId: 'cpp', version: 1, text: source } } });
  const tokens = await request('textDocument/semanticTokens/full', { textDocument: { uri: pathToFileURL(file).href } });
  assert.ok(tokens.data.length > 0 && tokens.data.length % 5 === 0, 'clangd returns C++ semantic tokens');
  await request('shutdown', null);
  send({ method: 'exit' });
  const timeout = setTimeout(() => child.kill(), 3000);
  await closed;
  clearTimeout(timeout);
  assert.equal(child.exitCode, 0, stderr);
  console.log(`PASS: Native clangd returned ${tokens.data.length / 5} C++ semantic tokens with the shared CoC flags`);
} finally {
  for (const entry of pending.values()) clearTimeout(entry.timer);
  if (child && child.exitCode === null) {
    const closed = new Promise(resolve => child.once('close', resolve));
    child.kill();
    await closed;
  }
  const resolved = path.resolve(fixture);
  assert.equal(path.dirname(resolved), path.resolve(os.tmpdir()));
  assert.ok(path.basename(resolved).startsWith('devenv-clangd-'));
  fs.rmSync(resolved, { recursive: true, force: true });
}

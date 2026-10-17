// Initialize the installed adapter over real ACP stdio; no login or model call.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
const packagePath = process.argv[2];
const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
const target = typeof pkg.bin === 'string' ? pkg.bin : pkg.bin['codex-acp'];
assert.ok(target, 'installed package declares codex-acp');
const child = spawn(process.execPath, [path.resolve(path.dirname(packagePath), target)], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
let incoming = '', diagnostic = '';
const closed = new Promise(resolve => child.once('close', resolve));
try {
  const response = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error(`ACP initialization timed out: ${diagnostic}`)), 45000);
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', code => { clearTimeout(timer); reject(Error(`ACP exited ${code}: ${diagnostic}`)); });
    child.stderr.on('data', bytes => { diagnostic = (diagnostic + bytes).slice(-4000); });
    child.stdout.on('data', bytes => {
      incoming += bytes;
      let end;
      while ((end = incoming.indexOf('\n')) >= 0) {
        const line = incoming.slice(0, end).trim(); incoming = incoming.slice(end + 1);
        if (!line) continue;
        try {
          const message = JSON.parse(line);
          if (message.id === 1) { clearTimeout(timer); resolve(message); }
        } catch (error) { clearTimeout(timer); reject(error); }
      }
    });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: 'windows-deployment-test', version: '1.0.0' } } }) + '\n');
  });
  assert.equal(response.error, undefined, JSON.stringify(response.error));
  assert.equal(response.result.protocolVersion, 1);
  assert.ok(response.result.agentCapabilities);
  console.log('PASS: installed Codex ACP adapter responds to a real initialize request');
} finally {
  child.stdin.end();
  let timer;
  const exited = await Promise.race([closed.then(() => true), new Promise(resolve => { timer = setTimeout(() => resolve(false), 5000); })]);
  clearTimeout(timer);
  if (!exited && child.pid) { execFileSync('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' }); }
}

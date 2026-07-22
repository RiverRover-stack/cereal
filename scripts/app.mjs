/**
 * Starts the whole application: FastAPI backend + Astro frontend.
 *
 * Why this exists instead of `concurrently "npm:app:api" "npm:app:web"`:
 * every `npm run` on Windows goes through the `npm.cmd` batch shim, so Ctrl+C
 * hits cmd.exe's "Terminate batch job (Y/N)?" prompt once per child and can
 * return the shell before the servers are actually gone. Spawning the real
 * binaries directly (node + the venv python) means no shim, no prompt.
 *
 * Shutdown also has to be a *tree* kill: `uvicorn --reload` runs the app in a
 * multiprocessing child that inherits the listening socket, so killing only the
 * parent leaves port 8000 held by an orphan that still serves requests.
 */

import { spawn } from 'node:child_process';
import { connect } from 'node:net';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const isWindows = process.platform === 'win32';

const API_PORT = process.env.API_PORT ?? '8000';
const WEB_PORT = process.env.WEB_PORT ?? '4321';

// The venv layout differs per platform; Scripts/ on Windows, bin/ elsewhere.
const python = isWindows
  ? join(root, 'backend', '.venv', 'Scripts', 'python.exe')
  : join(root, 'backend', '.venv', 'bin', 'python');

if (!existsSync(python)) {
  console.error(
    `\nNo virtualenv found at ${python}\n` +
      `Create it first (see backend/README.md):\n` +
      `  cd backend && python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt\n`,
  );
  process.exit(1);
}

// Astro's own bin, run through node — avoids the astro.cmd shim.
const astroBin = join(root, 'node_modules', 'astro', 'bin', 'astro.mjs');

const services = [
  {
    name: 'api',
    color: '[35m', // magenta
    command: python,
    args: ['-m', 'uvicorn', 'app.main:app', '--reload', '--port', API_PORT],
    cwd: join(root, 'backend'),
  },
  {
    name: 'web',
    color: '[36m', // cyan
    command: process.execPath,
    args: [astroBin, 'dev', '--port', WEB_PORT],
    cwd: root,
  },
];

const RESET = '[0m';
const children = new Map();
let shuttingDown = false;
let webDetached = false;

function write(service, chunk) {
  const text = chunk.toString();
  // Keep blank trailing lines from doubling up the prefix.
  for (const line of text.split(/\r?\n/)) {
    if (line.length === 0) continue;
    process.stdout.write(`${service.color}[${service.name}]${RESET} ${line}\n`);
  }
}

function start(service) {
  const child = spawn(service.command, service.args, {
    cwd: service.cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    // Do not let the console deliver Ctrl+C straight to the children; this
    // process owns shutdown so both die in a defined order.
    detached: !isWindows,
  });

  child.stdout.on('data', (c) => write(service, c));
  child.stderr.on('data', (c) => write(service, c));

  child.on('error', (err) => {
    write(service, `failed to start: ${err.message}`);
    shutdown(1);
  });

  child.on('exit', (code, signal) => {
    children.delete(service.name);
    if (shuttingDown) return;

    // `astro dev` puts itself in the background (exiting 0) when it detects an
    // AI-agent environment, so a clean exit here is not a crash — the server is
    // still up, just detached. Adopt it rather than tearing the API down.
    if (service.name === 'web' && code === 0) {
      webDetached = true;
      write(service, 'dev server detached into the background (agent environment detected)');
      write(service, 'it will be stopped along with the API on Ctrl+C');
      return;
    }

    write(service, `exited (${signal ?? `code ${code}`}) — stopping the other service`);
    shutdown(code ?? 1);
  });

  children.set(service.name, child);
  return child;
}

function killTree(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  if (isWindows) {
    // /T takes the whole tree, which is what catches uvicorn's reload worker.
    spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], {
      stdio: 'ignore',
      windowsHide: true,
    });
  } else {
    try {
      process.kill(-child.pid, 'SIGTERM');
    } catch {
      child.kill('SIGTERM');
    }
  }
}

function shutdown(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children.values()) killTree(child);

  // A detached dev server survives killing our children, so ask Astro to stop
  // it explicitly. Harmless when nothing is running.
  if (webDetached) {
    spawn(process.execPath, [astroBin, 'dev', 'stop'], {
      cwd: root,
      stdio: 'ignore',
      windowsHide: true,
    });
  }

  // Give the tree kills a moment to land before reporting the ports back.
  setTimeout(() => process.exit(code), 1500);
}

for (const signal of ['SIGINT', 'SIGTERM', 'SIGBREAK']) {
  process.on(signal, () => {
    process.stdout.write('\nStopping both services...\n');
    shutdown(0);
  });
}

function listening(host, port) {
  return new Promise((resolve) => {
    const socket = connect({ port: Number(port), host });
    const settle = (result) => {
      socket.destroy();
      resolve(result);
    };
    socket.setTimeout(1000);
    socket.on('connect', () => settle(true));
    socket.on('timeout', () => settle(false));
    socket.on('error', () => settle(false));
  });
}

/**
 * Resolves true if anything is accepting connections on the port. Both stacks
 * are probed: uvicorn binds 127.0.0.1 while Astro binds localhost, which
 * resolves to ::1 first on Windows — checking only IPv4 misses the dev server.
 */
async function portInUse(port) {
  const results = await Promise.all([listening('127.0.0.1', port), listening('::1', port)]);
  return results.some(Boolean);
}

/**
 * A crashed `uvicorn --reload` can leave its multiprocessing worker holding the
 * port after the parent is gone, and the resulting bind error is opaque. Check
 * up front and name the culprit instead. Nothing is killed automatically — the
 * port could belong to something unrelated.
 */
async function preflight() {
  const busy = [];
  for (const [label, port] of [
    ['api', API_PORT],
    ['web', WEB_PORT],
  ]) {
    if (await portInUse(port)) busy.push({ label, port });
  }
  if (busy.length === 0) return;

  console.error('\nCannot start — these ports are already in use:\n');
  for (const { label, port } of busy) {
    console.error(`  ${label}  port ${port}`);
  }
  console.error(
    `\nUsually a leftover server from a previous run. Find and stop it with:\n` +
      (isWindows
        ? `  netstat -ano | findstr :${busy[0].port}\n` +
          `  taskkill /PID <pid> /T /F\n`
        : `  lsof -i :${busy[0].port}\n  kill <pid>\n`) +
      `\nA detached Astro dev server stops with:  npx astro dev stop\n`,
  );
  process.exit(1);
}

console.log(
  `Starting Trends Arc\n` +
    `  api  http://localhost:${API_PORT}  (health: /health, docs: /docs)\n` +
    `  web  http://localhost:${WEB_PORT}\n` +
    `Press Ctrl+C to stop both.\n`,
);

await preflight();
for (const service of services) start(service);

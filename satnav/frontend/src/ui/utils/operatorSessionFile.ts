import { apiBaseUrl } from "@/api/config";
import { ApiError } from "@/api/errors";
import { getJson, postJson } from "@/api/http";

interface SessionLogLine {
  timestamp: string;
  tag: string;
  message: string;
}

interface OperatorLogStatusResponse {
  active: boolean;
  log_path: string | null;
  log_path_relative: string | null;
  started_at: string | null;
  timestamp: string;
}

/** Mirrors backend: one log file per FastAPI process (not per browser tab or inference). */
let sessionOpen = false;
let sessionLogPath: string | null = null;
let syncPromise: Promise<boolean> | null = null;
const pendingLines: string[] = [];
let flushTimer: number | null = null;
let flushChain: Promise<void> = Promise.resolve();

export function formatLogLineForFile(line: SessionLogLine): string {
  return `[${line.timestamp}] [${line.tag}] ${line.message}`;
}

export function isSessionLogFileOpen(): boolean {
  return sessionOpen;
}

export function getSessionLogPath(): string | null {
  return sessionLogPath;
}

/** Read log path from backend (file is opened when FastAPI starts). */
export async function syncOperatorLogFromBackend(): Promise<boolean> {
  if (syncPromise) {
    return syncPromise;
  }
  syncPromise = (async () => {
    try {
      const status = await getJson<OperatorLogStatusResponse>("/api/satnav/operator/logs/status");
      if (status.active) {
        sessionOpen = true;
        sessionLogPath = status.log_path_relative ?? status.log_path;
        return true;
      }
    } catch (error) {
      console.warn("SatNav operator log status failed", error);
    }
    sessionOpen = false;
    return false;
  })().finally(() => {
    syncPromise = null;
  });
  return syncPromise;
}

function scheduleFlush(): void {
  if (flushTimer !== null) {
    return;
  }
  flushTimer = window.setTimeout(() => {
    flushTimer = null;
    void flushPendingLines();
  }, 100);
}

async function flushPendingLines(): Promise<void> {
  if (!sessionOpen || pendingLines.length === 0) {
    return;
  }
  const lines = pendingLines.splice(0, pendingLines.length);
  flushChain = flushChain
    .then(async () => {
      await postJson("/api/satnav/operator/logs/append", { lines });
    })
    .catch((error) => {
      if (error instanceof ApiError && error.status === 409) {
        sessionOpen = false;
      }
      pendingLines.unshift(...lines);
      console.warn("SatNav operator log append failed", error);
    });
  await flushChain;
}

/** Append one console line (batched). Does not open/close files on the frontend. */
export function appendSessionLogLine(line: SessionLogLine): void {
  const push = () => {
    pendingLines.push(formatLogLineForFile(line));
    scheduleFlush();
  };
  if (sessionOpen) {
    push();
    return;
  }
  void syncOperatorLogFromBackend().then((ready) => {
    if (ready) {
      push();
    }
  });
}

/** Flush pending lines before tab unload; does not close the API log file. */
export async function flushSessionLogLines(): Promise<void> {
  await syncOperatorLogFromBackend();
  if (flushTimer !== null) {
    window.clearTimeout(flushTimer);
    flushTimer = null;
  }
  await flushPendingLines();
}

function requestAppendKeepalive(lines: string[]): void {
  if (lines.length === 0) {
    return;
  }
  try {
    void fetch(`${apiBaseUrl}/api/satnav/operator/logs/append`, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ lines }),
      keepalive: true,
    });
  } catch {
    // page may already be unloading
  }
}

/** pagehide: best-effort flush; file stays open until FastAPI exits. */
export function flushSessionLogLinesOnPageHide(): void {
  if (flushTimer !== null) {
    window.clearTimeout(flushTimer);
    flushTimer = null;
  }
  if (!sessionOpen || pendingLines.length === 0) {
    return;
  }
  const lines = pendingLines.splice(0, pendingLines.length);
  requestAppendKeepalive(lines);
}

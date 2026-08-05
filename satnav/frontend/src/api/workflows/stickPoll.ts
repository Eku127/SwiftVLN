import { defaultPollIntervals } from "../config";
import { flightClient } from "../clients/flight";
import type { StickTaskResponse } from "../types";

const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]);

export interface PollStickTaskOptions {
  intervalMs?: number;
  timeoutMs?: number;
  onUpdate?: (response: StickTaskResponse) => void;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export async function pollStickTask(
  taskId: string,
  options: PollStickTaskOptions = {},
): Promise<StickTaskResponse> {
  const intervalMs = options.intervalMs ?? defaultPollIntervals.stickTaskMs;
  const timeoutMs = options.timeoutMs ?? 120_000;
  const deadline = Date.now() + timeoutMs;
  let latest: StickTaskResponse | null = null;

  while (Date.now() < deadline) {
    latest = await flightClient.getStickTask(taskId);
    options.onUpdate?.(latest);
    if (TERMINAL.has((latest.status ?? "").toUpperCase())) {
      return latest;
    }
    await sleep(intervalMs);
  }

  throw new Error(
    `stick-task ${taskId} polling timed out after ${timeoutMs}ms (last=${latest?.status ?? "unknown"})`,
  );
}

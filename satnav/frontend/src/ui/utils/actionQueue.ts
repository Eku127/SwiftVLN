/**
 * Pure action-queue logic bridging model inference and flight stick-tasks.
 * UI always renders 4 slots; effective queue length (queueLength) may be 1–4.
 */
import type { InferenceResponse, NavAction } from "@/api";

/** Fixed number of UI slots (matches max 4 actions per inference). */
export const QUEUE_SLOT_COUNT = 4;

/** Stick-task status from GET .../flight/stick-task/{task_id}. */
export type StickTaskStatus = "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";

/** Runtime state for a single action slot. */
export interface ActionQueueSlot {
  /** Predicted action: 0=STOP, 1=forward, 2=turn left, 3=turn right; null if empty. */
  action: NavAction | null;
  /** Whether forward/turn has been submitted for this slot. */
  flightTriggered: boolean;
  /** Stick-task ID after flight submit; null before trigger. */
  taskId: string | null;
  /** Flight execution status for this slot; null before trigger. */
  stickStatus: StickTaskStatus | null;
}

/** Overall action queue state. */
export interface ActionQueueState {
  /** True while a multi-step cycle is active (starts on performed_inference=true). */
  cycleActive: boolean;
  /** Effective slot count this cycle (1–4), from latest performed_inference actions.length. */
  queueLength: number;
  /** Fixed 4 slots; only indices [0, queueLength) are valid. */
  slots: ActionQueueSlot[];
  /** Index of the slot to execute (0..queueLength-1); queueLength when cycle ends. */
  currentIndex: number | null;
}

/** Create 4 empty slots (no action, no task, not triggered). */
export function createEmptySlots(): ActionQueueSlot[] {
  return Array.from({ length: QUEUE_SLOT_COUNT }, () => ({
    action: null,
    flightTriggered: false,
    taskId: null,
    stickStatus: null,
  }));
}

/** Initial queue: no active cycle, no valid slots. */
export function createInitialQueueState(): ActionQueueState {
  return {
    cycleActive: false,
    queueLength: 0,
    slots: createEmptySlots(),
    currentIndex: null,
  };
}

/** Normalize API stick-task status string; returns null if unrecognized. */
export function normalizeStickStatus(
  status: string | null | undefined,
): StickTaskStatus | null {
  const normalized = (status ?? "").toUpperCase();
  if (
    normalized === "PENDING" ||
    normalized === "RUNNING" ||
    normalized === "COMPLETED" ||
    normalized === "FAILED"
  ) {
    return normalized;
  }
  return null;
}

/**
 * Derive current slot index from remaining_actions.
 * Formula: currentIndex = queueLength - remaining_actions.length
 */
export function computeCurrentIndex(
  queueLength: number,
  remainingActions: NavAction[],
): number {
  return queueLength - remainingActions.length;
}

/**
 * Merge an inference response into the queue.
 *
 * - performed_inference=true with non-empty actions: start new cycle (length 1–4), index 0
 * - feedback frame (performed_inference=false): keep slot actions, advance index via remaining_actions
 */
export function applyInferenceToQueue(
  state: ActionQueueState,
  result: Pick<
    InferenceResponse,
    "actions" | "remaining_actions" | "performed_inference"
  >,
): ActionQueueState {
  if (result.performed_inference && result.actions.length > 0) {
    const queueLength = result.actions.length;
    const slots = createEmptySlots();
    for (let index = 0; index < queueLength; index += 1) {
      slots[index] = {
        action: result.actions[index] ?? null,
        flightTriggered: false,
        taskId: null,
        stickStatus: null,
      };
    }
    return {
      cycleActive: true,
      queueLength,
      slots,
      currentIndex: 0,
    };
  }

  if (!state.cycleActive || state.queueLength === 0) {
    return state;
  }

  const currentIndex = computeCurrentIndex(
    state.queueLength,
    result.remaining_actions,
  );
  /** Empty remaining means all actions in this cycle have been consumed. */
  const cycleComplete = result.remaining_actions.length === 0;

  return {
    ...state,
    currentIndex: cycleComplete ? state.queueLength : currentIndex,
    cycleActive: !cycleComplete,
  };
}

/** Snapshot of the current executable slot; null if cycle inactive or index out of range. */
export function getCurrentSlot(
  state: ActionQueueState,
): ActionQueueSlot | null {
  if (
    !state.cycleActive ||
    state.currentIndex === null ||
    state.currentIndex < 0 ||
    state.currentIndex >= state.queueLength
  ) {
    return null;
  }
  return state.slots[state.currentIndex] ?? null;
}

/** True when a cycle was started and has finished (cycleActive cleared). */
export function isQueueCycleComplete(state: ActionQueueState): boolean {
  return state.queueLength > 0 && !state.cycleActive;
}

/**
 * Whether the Inference button should be enabled.
 * Allowed: no cycle, cycle complete, or current slot stick-task COMPLETED.
 * Blocked: PENDING/RUNNING/FAILED, or flight not yet triggered for current slot.
 */
export function canRunInference(params: {
  /** Inference API request in flight. */
  inferenceBusy: boolean;
  /** Flight submit or stick-task polling in progress. */
  flightBusy: boolean;
  queue: ActionQueueState;
}): boolean {
  if (params.inferenceBusy || params.flightBusy) {
    return false;
  }
  if (!params.queue.cycleActive) {
    return true;
  }
  if (isQueueCycleComplete(params.queue)) {
    return true;
  }
  const slot = getCurrentSlot(params.queue);
  if (!slot) {
    return true;
  }
  if (!slot.flightTriggered) {
    return false;
  }
  if (slot.stickStatus === "PENDING" || slot.stickStatus === "RUNNING") {
    return false;
  }
  if (slot.stickStatus === "FAILED") {
    return false;
  }
  return slot.stickStatus === "COMPLETED";
}

/**
 * Whether the Execute One Step button should be enabled.
 * Allowed: after inference, current slot not triggered, or FAILED retry.
 * Blocked: STOP, DRC invalidated, flight in progress.
 */
export function canExecuteStep(params: {
  flightBusy: boolean;
  flightReady: boolean;
  /** DRC invalidated after STOP; user must re-acquire flight control. */
  drcInvalidated: boolean;
  /** STOP engaged (model or manual). */
  stopActive: boolean;
  /** Latest inference next_action. */
  nextAction: NavAction | null;
  queue: ActionQueueState;
}): boolean {
  if (
    params.flightBusy ||
    params.stopActive ||
    params.drcInvalidated ||
    !params.flightReady
  ) {
    return false;
  }
  if (!params.queue.cycleActive || params.nextAction === null) {
    return false;
  }
  if (params.nextAction === 0) {
    return false;
  }
  const slot = getCurrentSlot(params.queue);
  if (!slot) {
    return false;
  }
  if (!slot.flightTriggered) {
    return true;
  }
  return slot.stickStatus === "FAILED";
}

/** Immutably patch fields on the slot at the given index. */
export function updateSlotAtIndex(
  state: ActionQueueState,
  index: number,
  patch: Partial<ActionQueueSlot>,
): ActionQueueState {
  if (index < 0 || index >= QUEUE_SLOT_COUNT) {
    return state;
  }
  const slots = state.slots.map((slot, slotIndex) =>
    slotIndex === index ? { ...slot, ...patch } : slot,
  );
  return { ...state, slots };
}

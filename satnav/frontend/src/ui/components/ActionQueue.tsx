import type { NavAction } from "@/api";
import { NAV_ACTION_LABEL } from "@/api";

import { ActionIcon } from "./ActionIcon";
import type { StickTaskStatus } from "../utils/actionQueue";
import { QUEUE_SLOT_COUNT } from "../utils/actionQueue";

/** @deprecated Legacy export: single action or empty slot. */
export type ActionSlot = NavAction | null;

/** Per-slot view model passed to ActionQueue (mapped from ActionQueueState.slots). */
export interface ActionQueueSlotView {
  /** Action code for this slot. */
  action: NavAction | null;
  /** Stick-task status for this slot. */
  stickStatus: StickTaskStatus | null;
  /** Whether flight control has been triggered for this slot. */
  flightTriggered: boolean;
}

interface ActionQueueProps {
  /** Up to 4 slots of display data. */
  slots: ActionQueueSlotView[];
  /** Effective slot count this cycle (1–4). */
  queueLength: number;
  /** Current executable slot index; null if no active cycle. */
  currentIndex: number | null;
  /** Whether to show STOP styling (model or manual). */
  stopActive: boolean;
  variant?: "default" | "sidebar";
}

/** Card visual state for border, icon, and status pill styling. */
type CardVisualState = "idle" | "active" | "pending" | "done" | "failed" | "stop";

/**
 * Resolve UI card state from queue position, flight status, and STOP flag.
 */
function resolveCardState(
  /** Slot index 0–3. */
  index: number,
  currentIndex: number | null,
  slot: ActionQueueSlotView,
  queueLength: number,
  stopActive: boolean,
): CardVisualState {
  if (index >= queueLength || slot.action === null) {
    return "idle";
  }
  if (stopActive && currentIndex === index) {
    return "stop";
  }
  if (currentIndex !== null && index < currentIndex) {
    return slot.stickStatus === "FAILED" ? "failed" : "done";
  }
  if (currentIndex !== null && index > currentIndex) {
    return "pending";
  }
  if (!slot.flightTriggered) {
    return "pending";
  }
  switch (slot.stickStatus) {
    case "PENDING":
    case "RUNNING":
      return "active";
    case "COMPLETED":
      return "done";
    case "FAILED":
      return "failed";
    default:
      return "pending";
  }
}

/**
 * Status pill label (red box area). User-facing strings stay Chinese.
 */
function statusLabel(
  state: CardVisualState,
  isEmpty: boolean,
  stickStatus: StickTaskStatus | null,
): string {
  if (isEmpty) {
    return "—";
  }
  switch (state) {
    case "stop":
      return "STOP";
    case "active":
      return stickStatus === "PENDING" ? "准备中" : "执行中";
    case "done":
      return "完成";
    case "failed":
      return "失败";
    case "pending":
      return "等待";
    case "idle":
    default:
      return "—";
  }
}

/**
 * Icon action to render; STOP state forces action=0.
 */
function displayAction(
  slot: ActionQueueSlotView,
  state: CardVisualState,
): NavAction | null {
  if (slot.action === null) {
    return null;
  }
  if (state === "stop") {
    return 0;
  }
  return slot.action;
}

/**
 * Four-slot action queue: icon, label, and stick-task status pill.
 */
export function ActionQueue({
  slots,
  queueLength,
  currentIndex,
  stopActive,
  variant = "default",
}: ActionQueueProps) {
  const iconSize = variant === "sidebar" ? 30 : 34;
  /** Pad to 4 entries so map never sees undefined. */
  const items = Array.from({ length: QUEUE_SLOT_COUNT }, (_, index) => slots[index] ?? {
    action: null,
    stickStatus: null,
    flightTriggered: false,
  });

  return (
    <section
      className={`action-queue action-queue--${variant}${
        variant === "default" ? " panel" : ""
      }`}
    >
      <div className="action-queue__title">
        <span className="action-queue__title-icon" aria-hidden="true">
          ☰
        </span>
        Action 队列
      </div>
      <div className={`action-queue__grid action-queue__grid--${variant}`}>
        {items.map((slot, index) => {
          const isEmpty = index >= queueLength || slot.action === null;
          const state = resolveCardState(
            index,
            currentIndex,
            slot,
            queueLength,
            stopActive,
          );
          const action = displayAction(slot, state);
          return (
            <div
              key={`slot-${index}`}
              className={`aq-card aq-card--${state}${isEmpty ? " aq-card--empty" : ""}`}
            >
              <div className="aq-card__index">{index + 1}</div>
              <div className="aq-card__body">
                <div className={`aq-card__icon aq-card__icon--${state}`}>
                  {action !== null ? <ActionIcon action={action} size={iconSize} /> : null}
                </div>
                <div className="aq-card__label">
                  {action === null ? "—" : NAV_ACTION_LABEL[action]}
                </div>
              </div>
              <div className="aq-card__status">
                {statusLabel(state, isEmpty, slot.stickStatus)}
              </div>
            </div>
          );
        })}
      </div>

      <style>{`
        .action-queue--default {
          padding: 0.9rem 1rem;
        }
        .action-queue--sidebar {
          padding-top: 0.15rem;
        }
        .action-queue__title {
          display: flex;
          align-items: center;
          gap: 0.4rem;
          font-weight: 600;
          margin-bottom: 0.7rem;
          color: #e8f1f8;
        }
        .action-queue__title-icon {
          color: #7fa0b8;
          font-size: 0.9rem;
        }
        .action-queue__grid {
          display: grid;
          gap: 0.55rem;
        }
        .action-queue__grid--default,
        .action-queue__grid--sidebar {
          grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .aq-card {
          position: relative;
          display: flex;
          flex-direction: column;
          min-height: 7.4rem;
          border-radius: 10px;
          border: 1px solid rgba(111, 142, 160, 0.22);
          padding: 0.45rem 0.35rem 0.4rem;
          background: linear-gradient(180deg, #0f1c28 0%, #0a141d 100%);
          text-align: center;
        }
        .aq-card--empty {
          border-color: rgba(95, 125, 146, 0.2);
          background: linear-gradient(180deg, #0e1a25 0%, #0a131c 100%);
        }
        .aq-card--empty .aq-card__label,
        .aq-card--empty .aq-card__status {
          color: #5f7d92;
        }
        .aq-card--empty .aq-card__status {
          background: rgba(95, 125, 146, 0.08);
          border-color: rgba(95, 125, 146, 0.12);
        }
        .aq-card--pending {
          border-color: rgba(240, 180, 41, 0.28);
          background: linear-gradient(180deg, rgba(42, 34, 18, 0.35) 0%, #0a141d 100%);
        }
        .aq-card--done {
          border-color: rgba(95, 125, 146, 0.28);
          background: linear-gradient(180deg, #0e1a25 0%, #0a131c 100%);
        }
        .aq-card--active {
          border-color: #2ee89a;
          background: linear-gradient(180deg, rgba(20, 58, 46, 0.55) 0%, rgba(10, 24, 20, 0.95) 100%);
          box-shadow:
            0 0 0 1px rgba(46, 232, 154, 0.22),
            0 0 16px rgba(46, 232, 154, 0.18);
        }
        .aq-card--failed,
        .aq-card--stop {
          border-color: rgba(255, 107, 118, 0.45);
          background: linear-gradient(180deg, rgba(58, 20, 24, 0.55) 0%, rgba(24, 10, 12, 0.95) 100%);
          box-shadow:
            0 0 0 1px rgba(255, 107, 118, 0.2),
            0 0 14px rgba(255, 107, 118, 0.14);
        }
        .aq-card__index {
          position: absolute;
          top: 0.32rem;
          left: 0.32rem;
          width: 1rem;
          height: 1rem;
          border-radius: 999px;
          display: grid;
          place-items: center;
          font-size: 0.58rem;
          font-weight: 700;
          color: #6f8ea0;
          background: rgba(111, 142, 160, 0.1);
          border: 1px solid rgba(111, 142, 160, 0.2);
        }
        .aq-card--active .aq-card__index {
          color: #b8ffe0;
          border-color: rgba(46, 232, 154, 0.45);
          background: rgba(46, 232, 154, 0.14);
        }
        .aq-card--pending .aq-card__index {
          color: #f0b429;
          border-color: rgba(240, 180, 41, 0.4);
          background: rgba(240, 180, 41, 0.12);
        }
        .aq-card--failed .aq-card__index,
        .aq-card--stop .aq-card__index {
          color: #ff9aa2;
          border-color: rgba(255, 107, 118, 0.45);
          background: rgba(255, 107, 118, 0.14);
        }
        .aq-card__body {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 0.85rem 0.15rem 0.35rem;
        }
        .aq-card__icon {
          display: grid;
          place-items: center;
          min-height: 2.15rem;
          margin-bottom: 0.2rem;
        }
        .aq-card--pending .aq-card__icon {
          color: #f0b429;
        }
        .aq-card__icon--active {
          color: #2ee89a;
        }
        .aq-card__icon--done {
          color: #5a7080;
        }
        .aq-card__icon--failed,
        .aq-card__icon--stop {
          color: #ff8a93;
        }
        .action-icon__svg {
          display: block;
        }
        .aq-card__label {
          font-weight: 600;
          font-size: 0.74rem;
          line-height: 1.2;
          color: #e8f1f8;
        }
        .aq-card--done .aq-card__label {
          color: #6f8ea0;
        }
        .aq-card--failed .aq-card__label,
        .aq-card--stop .aq-card__label {
          color: #ffb3ba;
        }
        .aq-card__status {
          margin-top: auto;
          border-radius: 6px;
          padding: 0.22rem 0.2rem;
          font-size: 0.64rem;
          line-height: 1.2;
          border: 1px solid transparent;
        }
        .aq-card--active .aq-card__status {
          color: #d8fff0;
          background: rgba(46, 232, 154, 0.28);
          border-color: rgba(46, 232, 154, 0.45);
        }
        .aq-card--pending .aq-card__status {
          color: #f0b429;
          background: rgba(240, 180, 41, 0.18);
          border-color: rgba(240, 180, 41, 0.32);
        }
        .aq-card--done .aq-card__status {
          color: #5f7d92;
          background: rgba(95, 125, 146, 0.1);
          border-color: rgba(95, 125, 146, 0.16);
        }
        .aq-card--failed .aq-card__status,
        .aq-card--stop .aq-card__status {
          color: #ffc4c9;
          background: rgba(255, 107, 118, 0.22);
          border-color: rgba(255, 107, 118, 0.38);
        }
      `}</style>
    </section>
  );
}

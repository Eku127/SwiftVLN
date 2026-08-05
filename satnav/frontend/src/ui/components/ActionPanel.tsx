import type { NavAction, StickTaskResponse } from "@/api";

import { ActionQueue, type ActionQueueSlotView } from "./ActionQueue";
import { NAV_ACTION_LABEL } from "@/api";
import { RefreshIcon } from "./RefreshIcon";
import { StopNoticeModal } from "./StopNoticeModal";
import {
  buildStickProgressView,
  extractStickBackendData,
  formatStickBackendDetail,
} from "../utils/stickProgress";

interface ActionPanelProps {
  /** Flight task in progress (submit or stick-task polling). */
  flightBusy: boolean;
  /** Whether Execute One Step is enabled (from canExecuteStep). */
  executeEnabled: boolean;
  /** Whether Skip is enabled (current slot FAILED, flight idle). */
  skipEnabled: boolean;
  flightProgress: number;
  latestStickTask: StickTaskResponse | null;
  /** Latest inference next_action for the stats row. */
  inferenceNextAction: NavAction | null;
  /** Latest inference remaining_actions. */
  inferenceRemainingActions: NavAction[];
  /** Latest inference performed_inference flag. */
  inferencePerformed: boolean | null;
  /** Per-slot queue display data. */
  queueSlots: ActionQueueSlotView[];
  /** Effective slot count this cycle. */
  queueLength: number;
  /** Current executable slot index. */
  currentIndex: number | null;
  /** Whether STOP state is active. */
  stopEngaged: boolean;
  /** Whether the STOP notice modal is open. */
  stopModalOpen: boolean;
  onDismissStopModal: () => void;
  /** Current stick-task ID for manual refresh. */
  currentStickTaskId: string | null;
  stickRefreshBusy: boolean;
  /** Calls forward/turn only; does not run inference. */
  onRunOneStep: () => void;
  /** Treat current FAILED slot as COMPLETED and unlock inference. */
  onSkipCurrentStep: () => void;
  onRefreshStickTask: () => void;
  /** Emergency STOP: defer if flight running, else immediate STOP. */
  onEmergencyStop: () => void;
  className?: string;
}

export function ActionPanel(props: ActionPanelProps) {
  const stickView = buildStickProgressView(props.latestStickTask);
  const backendData = extractStickBackendData(props.latestStickTask);
  const backendDetail = formatStickBackendDetail(backendData, props.latestStickTask);
  const progressValue =
    stickView?.progress !== null && stickView?.progress !== undefined
      ? Math.round(stickView.progress)
      : props.flightProgress;

  return (
    <section
      className={["panel", "action-panel", props.className].filter(Boolean).join(" ")}
    >
      <StopNoticeModal open={props.stopModalOpen} onClose={props.onDismissStopModal} />

      <ActionQueue
        slots={props.queueSlots}
        queueLength={props.queueLength}
        currentIndex={props.currentIndex}
        stopActive={props.stopEngaged && props.inferenceNextAction === 0}
        variant="sidebar"
      />

      <div className="action-panel__controls">
        <button
          type="button"
          className="btn btn-accent"
          disabled={!props.executeEnabled || props.flightBusy}
          onClick={props.onRunOneStep}
        >
          执行一步
        </button>
        <button
          type="button"
          className="btn btn-skip"
          disabled={!props.skipEnabled || props.flightBusy}
          onClick={props.onSkipCurrentStep}
        >
          跳过
        </button>
        <button
          type="button"
          className="btn btn-danger"
          onClick={props.onEmergencyStop}
        >
          ■ 应急 STOP
        </button>
      </div>

      <div className="action-panel__stats">
        <div className="action-panel__stat">
          <span className="action-panel__stat-label">当前动作</span>
          <span>
            {props.inferenceNextAction === null
              ? "—"
              : `${props.inferenceNextAction} · ${NAV_ACTION_LABEL[props.inferenceNextAction]}`}
          </span>
        </div>
        <div className="action-panel__stat">
          <span className="action-panel__stat-label">队列剩余</span>
          <span className="mono">
            {JSON.stringify(props.inferenceRemainingActions)}
          </span>
        </div>
        <div className="action-panel__stat">
          <span className="action-panel__stat-label">inference</span>
          <span>
            {props.inferencePerformed === null
              ? "—"
              : String(props.inferencePerformed)}
          </span>
        </div>
      </div>

      <div className="progress-block">
        <div className="progress-block__header">
          <div className="progress-block__label">飞控执行进度</div>
          <div className="progress-block__tools">
            <button
              type="button"
              className="progress-block__refresh"
              aria-label="刷新 stick-task 状态"
              title="刷新 stick-task 状态"
              disabled={!props.currentStickTaskId || props.stickRefreshBusy}
              onClick={props.onRefreshStickTask}
            >
              <RefreshIcon />
            </button>
            <div className="progress-block__percent">
              {progressValue > 0 ? `${progressValue}%` : "—"}
            </div>
          </div>
        </div>
        <div className="progress-block__bar">
          <div
            className="progress-block__fill"
            style={{ width: `${progressValue}%` }}
          />
        </div>
        <div className="progress-block__hint">
          {stickView
            ? stickView.headline
            : progressValue > 0
              ? "飞控任务执行中..."
              : "等待飞控完成..."}
        </div>
      </div>

      <div className="backend-data-panel">
        <div className="backend-data-panel__title">飞控执行详情</div>
        <pre className="backend-data-panel__body mono">
          {backendDetail ?? "等待 stick-task 返回执行详情 …"}
        </pre>
      </div>

      <style>{`
        .action-panel {
          padding: 0.85rem 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.6rem;
          height: 100%;
          min-height: 0;
        }
        .btn {
          border: none;
          border-radius: 10px;
          padding: 0.75rem 0.9rem;
          font-weight: 600;
        }
        .btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
        .btn-accent {
          background: #1a4a68;
          color: #d9f4ff;
          border: 1px solid #2d6f97;
        }
        .btn-danger {
          background: linear-gradient(180deg, #ff6b76, #e52f41);
          color: white;
        }
        .btn-skip {
          background: #1e3344;
          color: #c8dce8;
          border: 1px solid #3d5f78;
        }
        .action-panel__controls {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 0.5rem;
        }
        .action-panel__stats {
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          gap: 0.25rem 1rem;
          font-size: 0.78rem;
          line-height: 1.35;
          color: #cfe0ec;
        }
        .action-panel__stat {
          display: inline-flex;
          align-items: baseline;
          gap: 0.3rem;
          min-width: 0;
        }
        .action-panel__stat-label {
          color: var(--text-muted);
          flex-shrink: 0;
        }
        .action-panel__stat-label::after {
          content: "：";
        }
        .progress-block__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.35rem;
        }
        .progress-block__label {
          font-size: 0.85rem;
        }
        .progress-block__tools {
          display: flex;
          align-items: center;
          gap: 0.45rem;
        }
        .progress-block__refresh {
          width: 1.65rem;
          height: 1.65rem;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: #102030;
          color: #9eb6c8;
          display: grid;
          place-items: center;
          padding: 0;
        }
        .progress-block__refresh:hover:not(:disabled) {
          color: var(--text);
          border-color: #2d5068;
          background: #152a3d;
        }
        .progress-block__refresh:disabled {
          opacity: 0.45;
          cursor: not-allowed;
        }
        .progress-block__percent {
          font-size: 0.82rem;
          color: #9ff3c8;
          min-width: 2.5rem;
          text-align: right;
        }
        .progress-block__bar {
          height: 8px;
          background: #0a1824;
          border-radius: 999px;
          overflow: hidden;
          border: 1px solid var(--border);
        }
        .progress-block__fill {
          height: 100%;
          background: linear-gradient(90deg, #35c2ff, #1fd47a);
          transition: width 0.25s ease;
        }
        .progress-block__hint {
          color: var(--text-muted);
          font-size: 0.75rem;
          margin-top: 0.25rem;
        }
        .backend-data-panel {
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #08131d;
          overflow: hidden;
          flex: 1 1 auto;
          min-height: 0;
          display: flex;
          flex-direction: column;
        }
        .backend-data-panel__title {
          padding: 0.45rem 0.75rem;
          font-size: 0.76rem;
          color: #8fa6b8;
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
          background: #0a1824;
          flex: 0 0 auto;
        }
        .backend-data-panel__body {
          margin: 0;
          padding: 0.55rem 0.75rem 0.65rem;
          flex: 1 1 auto;
          min-height: 7.5rem;
          overflow: auto;
          font-size: 0.76rem;
          line-height: 1.5;
          color: #9eb6c8;
          white-space: pre-wrap;
          word-break: break-word;
        }
      `}</style>
    </section>
  );
}

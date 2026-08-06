import { useEffect, useRef } from "react";

import type { StickTaskResponse } from "@/api";

import {
  getCurrentSlot,
  type ActionQueueState,
} from "../utils/actionQueue";

export const AUTO_FLIGHT_FAILURE_TIMEOUT_MS = 100_000;

export interface UseAutoFlightOrchestratorParams {
  autoFlightEnabled: boolean;
  autoFlightArmed: boolean;
  autoFlightPaused: boolean;
  autoFlightFailureDeadlineAt: number | null;
  stopEngaged: boolean;
  inferenceBusy: boolean;
  flightBusy: boolean;
  executeEnabled: boolean;
  inferenceEnabled: boolean;
  actionQueue: ActionQueueState;
  latestStickTask: StickTaskResponse | null;
  onRunOneStep: () => Promise<void>;
  onInference: (options?: { fromAuto?: boolean }) => Promise<boolean>;
  onSkipCurrentStep: () => void;
  onAutoFlightFailurePause: () => void;
  onAutoFlightFailureTimeout: () => void;
  setAutoFlightCountdownSec: (value: number | null) => void;
}

/**
 * Schedules automatic execute/inference and handles post-flight failure routing.
 * Does not duplicate stick/inference logic — only calls existing handlers.
 *
 * 全自动飞控编排层：在用户手动完成首帧推理后，按与手工操作相同的门控
 * （executeEnabled / inferenceEnabled）自动串联「执行一步」与「推理」。
 * 飞控失败时：TIMEOUT 自动跳过；其他失败交由 controller 暂停并弹框。
 * 不改变 stick 轮询、队列推进或推理 API 的既有实现，仅调用已有 handler。
 */
export function useAutoFlightOrchestrator({
  autoFlightEnabled,
  autoFlightArmed,
  autoFlightPaused,
  autoFlightFailureDeadlineAt,
  stopEngaged,
  inferenceBusy,
  flightBusy,
  executeEnabled,
  inferenceEnabled,
  actionQueue,
  latestStickTask,
  onRunOneStep,
  onInference,
  onSkipCurrentStep,
  onAutoFlightFailurePause,
  onAutoFlightFailureTimeout,
  setAutoFlightCountdownSec,
}: UseAutoFlightOrchestratorParams): void {
  const schedulingRef = useRef(false);
  const failureHandledKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (!autoFlightFailureDeadlineAt) {
      setAutoFlightCountdownSec(null);
      return;
    }

    const tick = () => {
      const remainingMs = autoFlightFailureDeadlineAt - Date.now();
      const remainingSec = Math.max(0, Math.ceil(remainingMs / 1000));
      setAutoFlightCountdownSec(remainingSec);
      if (remainingSec <= 0) {
        onAutoFlightFailureTimeout();
      }
    };

    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [
    autoFlightFailureDeadlineAt,
    onAutoFlightFailureTimeout,
    setAutoFlightCountdownSec,
  ]);

  useEffect(() => {
    if (
      !autoFlightEnabled ||
      !autoFlightArmed ||
      autoFlightPaused ||
      stopEngaged ||
      inferenceBusy ||
      flightBusy
    ) {
      return;
    }

    const slot = getCurrentSlot(actionQueue);
    if (slot?.flightTriggered && slot.stickStatus === "FAILED") {
      const rawStatus = (latestStickTask?.status ?? "").toUpperCase();
      const failureKey = `${slot.taskId ?? "none"}:${rawStatus}`;
      if (failureHandledKeyRef.current === failureKey) {
        return;
      }
      failureHandledKeyRef.current = failureKey;

      if (rawStatus === "TIMEOUT") {
        onSkipCurrentStep();
        failureHandledKeyRef.current = null;
      } else {
        onAutoFlightFailurePause();
      }
      return;
    }

    if (schedulingRef.current) {
      return;
    }

    const schedule = async () => {
      if (executeEnabled) {
        schedulingRef.current = true;
        try {
          await onRunOneStep();
        } finally {
          schedulingRef.current = false;
        }
        return;
      }
      if (inferenceEnabled) {
        schedulingRef.current = true;
        try {
          await onInference({ fromAuto: true });
        } finally {
          schedulingRef.current = false;
        }
      }
    };

    void schedule();
  }, [
    actionQueue,
    autoFlightArmed,
    autoFlightEnabled,
    autoFlightPaused,
    executeEnabled,
    flightBusy,
    inferenceBusy,
    inferenceEnabled,
    latestStickTask?.status,
    onAutoFlightFailurePause,
    onInference,
    onRunOneStep,
    onSkipCurrentStep,
    stopEngaged,
  ]);

  useEffect(() => {
    if (!autoFlightPaused) {
      failureHandledKeyRef.current = null;
    }
  }, [autoFlightPaused]);

  useEffect(() => {
    if (!autoFlightEnabled) {
      failureHandledKeyRef.current = null;
    }
  }, [autoFlightEnabled]);
}

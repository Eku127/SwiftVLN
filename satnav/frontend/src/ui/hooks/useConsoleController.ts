import type { NavAction, StickTaskResponse } from "@/api";
import {
  defaultPollIntervals,
  executeFlightAction,
  flightClient,
  healthClient,
  mediaClient,
  modelClient,
  pollStickTask,
  runInferenceStep,
  rtmpClient,
} from "@/api";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  applyInferenceToQueue,
  canExecuteStep,
  canRunInference,
  createInitialQueueState,
  normalizeStickStatus,
  updateSlotAtIndex,
  type ActionQueueState,
} from "../utils/actionQueue";
import {
  computeStickProgressPercent,
  extractStickBackendData,
} from "../utils/stickProgress";

export interface AircraftPose {
  latitude: number | null;
  longitude: number | null;
  heading: number | null;
  height: number | null;
  receivedAtMs: number | null;
  rcSn: string | null;
  deviceSn: string | null;
}

export interface LogLine {
  id: string;
  timestamp: string;
  tag: string;
  message: string;
  level?: "info" | "warn" | "error";
}

type Phase =
  | "idle"
  | "starting"
  | "inferring"
  | "flying"
  | "waiting_fc"
  | "stopped"
  | "error";

const DEFAULT_LOGIN = {
  username: "",
  password: "",
  flag: 1,
};

/** Placeholder only — replace with your RC / aircraft SNs in the UI (do not commit real SNs). */
const DEFAULT_DEVICE = {
  rc_sn: "",
  device_sn: "",
};

function nowStamp(): string {
  const d = new Date();
  return d.toLocaleTimeString("zh-CN", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

const EMPTY_OSD_FIELDS = {
  latitude: null,
  longitude: null,
  heading: null,
  height: null,
  receivedAtMs: null,
} as const;

function readDeviceBinding(device: {
  rc_sn?: string | null;
  device_sn?: string | null;
}): Pick<AircraftPose, "rcSn" | "deviceSn"> {
  return {
    rcSn: device.rc_sn ?? null,
    deviceSn: device.device_sn ?? null,
  };
}

function readOsdSnapshot(osd: {
  attitude_head?: number | null;
  latitude?: number | null;
  longitude?: number | null;
  height?: number | null;
  received_at_ms?: number | null;
}): Pick<AircraftPose, "latitude" | "longitude" | "heading" | "height" | "receivedAtMs"> {
  return {
    latitude: osd.latitude ?? null,
    longitude: osd.longitude ?? null,
    heading: osd.attitude_head ?? null,
    height: osd.height ?? null,
    receivedAtMs: osd.received_at_ms ?? null,
  };
}

export function useConsoleController() {
  const [instruction, setInstruction] = useState(
    "Fly to the lake and fly around the lake clockwise, then stop near the dock.",
  );
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [flightBusy, setFlightBusy] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [apiHealthy, setApiHealthy] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [rtmpActive, setRtmpActive] = useState(false);
  const [flightBackendHealthy, setFlightBackendHealthy] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);
  const [flightReady, setFlightReady] = useState(false);
  const [aircraftPose, setAircraftPose] = useState<AircraftPose>({
    ...EMPTY_OSD_FIELDS,
    rcSn: null,
    deviceSn: null,
  });
  const [sessionId, setSessionId] = useState<string | null>(null);
  // --- Action queue: inference <-> flight control loop ---
  /** Four-slot queue runtime state (1–4 valid slots; UI always shows 4). */
  const [actionQueue, setActionQueue] = useState<ActionQueueState>(createInitialQueueState);
  /** Latest inference next_action; drives Execute One Step and stats row. */
  const [inferenceNextAction, setInferenceNextAction] = useState<NavAction | null>(null);
  /** Latest inference remaining_actions. */
  const [inferenceRemainingActions, setInferenceRemainingActions] = useState<NavAction[]>([]);
  /** Latest inference performed_inference flag. */
  const [inferencePerformed, setInferencePerformed] = useState<boolean | null>(null);
  /** True after STOP (model next_action=0 or emergency STOP took effect). */
  const [stopEngaged, setStopEngaged] = useState(false);
  /** After STOP, DRC is invalid until user re-acquires flight control. */
  const [drcInvalidated, setDrcInvalidated] = useState(false);
  /** STOP notice modal visibility. */
  const [stopModalOpen, setStopModalOpen] = useState(false);
  /**
   * Emergency STOP pending flag: set while flight is RUNNING;
   * activateStop runs after the current stick-task finishes.
   */
  const pendingManualStopRef = useRef(false);
  const [flightProgress, setFlightProgress] = useState(0);
  const [latestStickTask, setLatestStickTask] = useState<StickTaskResponse | null>(null);
  const [currentStickTaskId, setCurrentStickTaskId] = useState<string | null>(null);
  const [stickRefreshBusy, setStickRefreshBusy] = useState(false);
  const [rawMeta, setRawMeta] = useState("—");
  const [modelInputMeta, setModelInputMeta] = useState("—");
  const [rawImageTick, setRawImageTick] = useState(0);
  const [modelInputTick, setModelInputTick] = useState(0);
  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const [autoScrollLogs, setAutoScrollLogs] = useState(true);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const logSeq = useRef(0);
  const modelLogCursor = useRef(0);

  const appendLog = useCallback((tag: string, message: string, level: LogLine["level"] = "info") => {
    logSeq.current += 1;
    setLogLines((prev) => [
      ...prev.slice(-400),
      {
        id: String(logSeq.current),
        timestamp: nowStamp(),
        tag,
        message,
        level,
      },
    ]);
  }, []);

  /**
   * Enter STOP state: next_action=0, invalidate DRC, show modal.
   * @param source - "model" from inference; "manual" from emergency STOP
   */
  const activateStop = useCallback(
    (source: "model" | "manual") => {
      setInferenceNextAction(0);
      setStopEngaged(true);
      setDrcInvalidated(true);
      setFlightReady(false);
      setStopModalOpen(true);
      pendingManualStopRef.current = false;
      appendLog(
        "UI",
        source === "manual" ? "应急 STOP 已生效" : "模型返回 STOP",
        "warn",
      );
    },
    [appendLog],
  );

  const refreshDeviceInfo = useCallback(async () => {
    try {
      const device = await flightClient.getDeviceInfo();
      setLoggedIn(Boolean(device.logged_in));
      setFlightReady(Boolean(device.drc_ready));
      setAircraftPose((prev) => ({
        ...prev,
        ...readDeviceBinding(device),
      }));
    } catch {
      setLoggedIn(false);
      setFlightReady(false);
    }
  }, []);

  const refreshOsd = useCallback(async () => {
    try {
      const osd = await flightClient.getOsdLatest();
      setAircraftPose((prev) => ({
        ...prev,
        ...readOsdSnapshot(osd),
      }));
    } catch {
      setAircraftPose((prev) => ({
        ...prev,
        ...EMPTY_OSD_FIELDS,
      }));
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    const [healthResult, modelResult, rtmpResult, flightResult] =
      await Promise.allSettled([
        healthClient.getHealth(),
        modelClient.getStatus(),
        rtmpClient.getStatus(),
        flightClient.getBackendStatus(),
      ]);

    if (healthResult.status === "fulfilled") {
      setApiHealthy(healthResult.value.status === "ok");
    } else {
      setApiHealthy(false);
    }

    if (modelResult.status === "fulfilled") {
      setModelReady(modelResult.value.status === "ready");
    } else {
      setModelReady(false);
    }

    if (rtmpResult.status === "fulfilled") {
      const rtmpStatus = rtmpResult.value;
      setRtmpActive(rtmpStatus.connected && rtmpStatus.stream_active);
      if (rtmpStatus.video?.width && rtmpStatus.video?.height) {
        setRawMeta(`${rtmpStatus.video.width}×${rtmpStatus.video.height}`);
      }
    } else {
      setRtmpActive(false);
    }

    if (flightResult.status === "fulfilled") {
      setFlightBackendHealthy(flightResult.value.healthy);
    } else {
      setFlightBackendHealthy(false);
    }

    const failures = [
      healthResult.status === "rejected" ? `health: ${healthResult.reason}` : null,
      modelResult.status === "rejected" ? `model: ${modelResult.reason}` : null,
      rtmpResult.status === "rejected" ? `rtmp: ${rtmpResult.reason}` : null,
      flightResult.status === "rejected" ? `flight: ${flightResult.reason}` : null,
    ].filter(Boolean);
    if (failures.length > 0) {
      appendLog("SYS", `状态刷新部分失败: ${failures.join("; ")}`, "warn");
    }

    await refreshDeviceInfo();
  }, [appendLog, refreshDeviceInfo]);

  const pullModelLogs = useCallback(async () => {
    try {
      const payload = await modelClient.getLogs(modelLogCursor.current, 100);
      if (payload.logs.length > 0) {
        for (const entry of payload.logs) {
          appendLog(
            entry.source.toUpperCase(),
            entry.message,
            entry.level === "error" ? "error" : "info",
          );
        }
        modelLogCursor.current = payload.latest_sequence;
      }
    } catch {
      // backend may still be starting
    }
  }, [appendLog]);

  useEffect(() => {
    void refreshStatus();
    const timer = window.setInterval(() => {
      void refreshStatus();
    }, defaultPollIntervals.statusMs);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  useEffect(() => {
    if (!rtmpActive) {
      return;
    }
    const timer = window.setInterval(() => {
      setRawImageTick((value) => value + 1);
    }, defaultPollIntervals.rawImageMs);
    return () => window.clearInterval(timer);
  }, [rtmpActive]);

  useEffect(() => {
    if (rtmpActive) {
      setRawImageTick((value) => value + 1);
    }
  }, [rtmpActive]);

  useEffect(() => {
    if (!flightReady) {
      setAircraftPose((prev) => ({
        ...prev,
        ...EMPTY_OSD_FIELDS,
      }));
      return;
    }
    void refreshOsd();
    const timer = window.setInterval(() => {
      void refreshOsd();
    }, defaultPollIntervals.osdMs);
    return () => window.clearInterval(timer);
  }, [flightReady, refreshOsd]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void pullModelLogs();
    }, defaultPollIntervals.modelLogsMs);
    return () => window.clearInterval(timer);
  }, [pullModelLogs]);

  const rawImageUrl = useMemo(
    () => (rtmpActive ? mediaClient.rawImageUrl(rawImageTick) : ""),
    [rawImageTick, rtmpActive],
  );

  const modelInputUrl = useMemo(
    () => (sessionId ? mediaClient.modelInputImageUrl(modelInputTick) : ""),
    [modelInputTick, sessionId],
  );

  const phaseLabel = useMemo(() => {
    switch (phase) {
      case "idle":
        return "等待开始";
      case "starting":
        return "准备飞控";
      case "inferring":
        return "推理中";
      case "flying":
        return "下发飞控";
      case "waiting_fc":
        return "等待飞控";
      case "stopped":
        return "已停止";
      case "error":
        return "异常";
      default:
        return "—";
    }
  }, [phase]);

  /**
   * Apply inference API response to queue and display snapshot; does not call flight APIs.
   * New cycle on performed_inference=true; triggers STOP when next_action=0.
   */
  const applyInference = useCallback(
    (result: Awaited<ReturnType<typeof runInferenceStep>>) => {
      if (result.performed_inference && result.actions.length > 0) {
        setStopEngaged(false);
      }
      setActionQueue((prev) => applyInferenceToQueue(prev, result));
      setSessionId(result.session_id);
      setInferenceNextAction(result.next_action);
      setInferenceRemainingActions(result.remaining_actions);
      setInferencePerformed(result.performed_inference);
      setModelInputMeta(result.image_path.split("/").pop() ?? result.image_path);
      setModelInputTick((v) => v + 1);
      appendLog(
        "INFER",
        `actions=${JSON.stringify(result.actions)}, next_action=${result.next_action}, remaining_actions=${JSON.stringify(result.remaining_actions)}, performed_inference=${result.performed_inference}`,
      );
      if (result.next_action === 0) {
        activateStop("model");
      }
    },
    [activateStop, appendLog],
  );

  /** Whether Inference button is enabled (see actionQueue.canRunInference). */
  const inferenceEnabled = useMemo(
    () =>
      canRunInference({
        inferenceBusy,
        flightBusy,
        queue: actionQueue,
      }),
    [actionQueue, flightBusy, inferenceBusy],
  );

  /** Whether Execute One Step is enabled (see actionQueue.canExecuteStep). */
  const executeEnabled = useMemo(
    () =>
      canExecuteStep({
        flightBusy,
        flightReady,
        drcInvalidated,
        stopActive: stopEngaged || inferenceNextAction === 0,
        nextAction: inferenceNextAction,
        queue: actionQueue,
      }),
    [
      actionQueue,
      drcInvalidated,
      flightBusy,
      flightReady,
      inferenceNextAction,
      stopEngaged,
    ],
  );

  /** POST /api/satnav/model/inference only; does not trigger flight control. */
  const onInference = useCallback(async () => {
    setInferenceBusy(true);
    setPhase("inferring");
    try {
      appendLog("UI", "推理");
      const result = await runInferenceStep(instruction);
      applyInference(result);
      setPhase("idle");
    } catch (error) {
      setPhase("error");
      appendLog("INFER", String(error), "error");
    } finally {
      setInferenceBusy(false);
    }
  }, [appendLog, applyInference, instruction]);

  const onLoginFlightSystem = useCallback(
    async (form: { username: string; password: string; flag: number }) => {
      setFlightBusy(true);
      setPhase("starting");
      try {
        appendLog("UI", "登陆飞控系统");
        const body =
          form.username && form.password
            ? {
                username: form.username,
                password: form.password,
                flag: form.flag,
              }
            : { flag: form.flag };
        const login = await flightClient.login(body);
        setLoggedIn(Boolean(login.logged_in));
        await refreshDeviceInfo();
        appendLog(
          "FC",
          `${login.auth_method === "refresh" ? "token 刷新" : "login"} 完成，workspace_id=${login.workspace_id ?? "—"}`,
        );
        setPhase("idle");
      } catch (error) {
        setPhase("error");
        appendLog("FC", String(error), "error");
        throw error;
      } finally {
        setFlightBusy(false);
      }
    },
    [appendLog, refreshDeviceInfo],
  );

  const onRegisterDevice = useCallback(
    async (rcSn: string, deviceSn: string) => {
      setFlightBusy(true);
      try {
        appendLog("UI", `摇控/无人机绑定 rc_sn=${rcSn}, device_sn=${deviceSn}`);
        const device = await flightClient.registerDevice({
          rc_sn: rcSn,
          device_sn: deviceSn,
        });
        setAircraftPose({
          ...EMPTY_OSD_FIELDS,
          ...readDeviceBinding(device),
        });
        setLoggedIn(Boolean(device.logged_in));
        setFlightReady(Boolean(device.drc_ready));
        appendLog(
          "FC",
          `绑定完成 rc_sn=${device.rc_sn ?? "—"}, device_sn=${device.device_sn ?? "—"}`,
        );
      } catch (error) {
        setPhase("error");
        appendLog("FC", String(error), "error");
        throw error;
      } finally {
        setFlightBusy(false);
      }
    },
    [appendLog],
  );

  /** After DRC acquire, clear STOP/DRC-invalid flags so forward/turn can run again. */
  const onAcquireFlightControl = useCallback(async () => {
    setFlightBusy(true);
    setPhase("starting");
    try {
      appendLog("UI", "获取飞行控制");
      const drc = await flightClient.acquireDrc({ expire_sec: 3600 });
      setFlightReady(Boolean(drc.drc_ready));
      setDrcInvalidated(false);
      setStopEngaged(false);
      await refreshDeviceInfo();
      await refreshOsd();
      appendLog("FC", `DRC 就绪，client_id=${drc.client_id}`);
      setPhase("idle");
    } catch (error) {
      setPhase("error");
      appendLog("FC", String(error), "error");
    } finally {
      setFlightBusy(false);
    }
  }, [appendLog, refreshDeviceInfo, refreshOsd]);

  /** Write stick-task poll result into the current slot's stickStatus / taskId. */
  const updateCurrentSlotStatus = useCallback(
    (status: ReturnType<typeof normalizeStickStatus>, taskId?: string) => {
      setActionQueue((prev) => {
        const index = prev.currentIndex;
        if (index === null) {
          return prev;
        }
        return updateSlotAtIndex(prev, index, {
          stickStatus: status,
          ...(taskId ? { taskId } : {}),
        });
      });
    },
    [],
  );

  /** Handle stick-task poll/refresh: sync slot state, progress bar, and detail panel. */
  const applyStickTaskUpdate = useCallback(
    (stick: StickTaskResponse) => {
      setLatestStickTask(stick);
      const stickStatus = normalizeStickStatus(stick.status);
      if (stickStatus) {
        updateCurrentSlotStatus(stickStatus, stick.task_id);
      }
      const data = extractStickBackendData(stick);
      const pct = computeStickProgressPercent(data, stick);
      if (pct !== null) {
        setFlightProgress(Math.round(pct));
      } else {
        const status = (stick.status ?? "").toUpperCase();
        if (status === "COMPLETED") {
          setFlightProgress(100);
        } else if (status === "RUNNING") {
          setFlightProgress((value) => (value > 0 ? value : 8));
        }
      }
    },
    [updateCurrentSlotStatus],
  );

  const onRefreshStickTask = useCallback(async () => {
    if (!currentStickTaskId) {
      appendLog("UI", "无可刷新的 stick-task", "warn");
      return;
    }
    setStickRefreshBusy(true);
    try {
      appendLog("UI", `刷新 stick-task ${currentStickTaskId}`);
      const stick = await flightClient.getStickTask(currentStickTaskId);
      applyStickTaskUpdate(stick);
      appendLog("FC", `stick-task 刷新: status=${stick.status}`);
    } catch (error) {
      appendLog("FC", String(error), "error");
    } finally {
      setStickRefreshBusy(false);
    }
  }, [appendLog, applyStickTaskUpdate, currentStickTaskId]);

  /**
   * Execute One Step: call forward/turn from inferenceNextAction, poll stick-task to terminal state.
   * Does not advance currentIndex here (next inference feedback frame does that).
   */
  const onRunOneStep = useCallback(async () => {
    /** Action to submit from the latest inference. */
    const nextAction = inferenceNextAction;
    /** Current queue slot index. */
    const currentIndex = actionQueue.currentIndex;

    if (!executeEnabled || nextAction === null || nextAction === 0 || currentIndex === null) {
      appendLog("UI", "当前不可执行飞控，请先完成推理并确认上一步飞控状态", "warn");
      return;
    }

    setFlightBusy(true);
    setPhase("flying");
    setFlightProgress(0);
    setLatestStickTask(null);
    try {
      appendLog("UI", `执行一步：action=${nextAction}, slot=${currentIndex + 1}`);
      const flight = await executeFlightAction(nextAction);
      if (!flight?.task_id) {
        setPhase("idle");
        return;
      }

      /** Initial status from flight submit response, usually PENDING. */
      const initialStatus = normalizeStickStatus(flight.status) ?? "PENDING";
      setActionQueue((prev) =>
        updateSlotAtIndex(prev, currentIndex, {
          flightTriggered: true,
          taskId: flight.task_id,
          stickStatus: initialStatus,
        }),
      );

      setPhase("waiting_fc");
      setCurrentStickTaskId(flight.task_id);
      appendLog("FC", `下发 action=${flight.action}, task_id=${flight.task_id}`);

      const stickTask = await pollStickTask(flight.task_id, {
        onUpdate: (stick) => {
          applyStickTaskUpdate(stick);
          appendLog("FC", `stick-task ${stick.task_id} status=${stick.status}`);
        },
      });

      setLatestStickTask(stickTask);
      /** Terminal status after polling; gates the next inference. */
      const finalStatus = normalizeStickStatus(stickTask.status) ?? "FAILED";
      setActionQueue((prev) =>
        updateSlotAtIndex(prev, currentIndex, {
          stickStatus: finalStatus,
        }),
      );

      appendLog("FC", `stick-task 结束: ${stickTask.status}`);

      /** If emergency STOP was queued during flight, activate STOP after this task ends. */
      if (pendingManualStopRef.current) {
        activateStop("manual");
        setPhase("stopped");
        return;
      }

      if (finalStatus === "COMPLETED") {
        setFlightProgress(100);
        setPhase("idle");
      } else {
        setPhase("error");
      }
    } catch (error) {
      setPhase("error");
      appendLog("STEP", String(error), "error");
    } finally {
      setFlightBusy(false);
    }
  }, [
    actionQueue.currentIndex,
    activateStop,
    appendLog,
    applyStickTaskUpdate,
    executeEnabled,
    inferenceNextAction,
  ]);

  /**
   * Emergency STOP (always enabled):
   * - If flight is running: queue pendingManualStop, activateStop after task completes
   * - Otherwise: activateStop immediately
   */
  const onEmergencyStop = useCallback(() => {
    if (flightBusy) {
      pendingManualStopRef.current = true;
      appendLog("UI", "应急 STOP 已登记，当前飞控任务完成后生效", "warn");
      return;
    }
    activateStop("manual");
    setPhase("stopped");
  }, [activateStop, appendLog, flightBusy]);

  /** Dismiss STOP modal only; does not restore DRC (user must re-acquire flight control). */
  const dismissStopModal = useCallback(() => {
    setStopModalOpen(false);
  }, []);

  const onRefreshRtmp = useCallback(async () => {
    setSettingsBusy(true);
    try {
      appendLog("UI", "刷新 RTMP 服务");
      const status = await rtmpClient.refresh();
      setRtmpActive(status.connected && status.stream_active);
      if (status.video?.width && status.video?.height) {
        setRawMeta(`${status.video.width}×${status.video.height}`);
      }
      appendLog(
        "RTMP",
        `refresh 完成，stream_active=${status.stream_active}`,
        status.stream_active ? "info" : "warn",
      );
    } catch (error) {
      appendLog("RTMP", String(error), "error");
    } finally {
      setSettingsBusy(false);
    }
  }, [appendLog]);

  const clearLogs = useCallback(() => {
    setLogLines([]);
  }, []);

  return {
    instruction,
    setInstruction,
    inferenceBusy,
    flightBusy,
    phaseLabel,
    apiHealthy,
    modelReady,
    rtmpActive,
    flightBackendHealthy,
    loggedIn,
    flightReady,
    aircraftPose,
    defaultRcSn: DEFAULT_DEVICE.rc_sn,
    defaultDeviceSn: DEFAULT_DEVICE.device_sn,
    defaultLoginUsername: DEFAULT_LOGIN.username,
    defaultLoginPassword: DEFAULT_LOGIN.password,
    defaultLoginFlag: DEFAULT_LOGIN.flag,
    sessionId,
    actionQueue,
    inferenceNextAction,
    inferenceRemainingActions,
    inferencePerformed,
    inferenceEnabled,
    executeEnabled,
    stopEngaged,
    stopModalOpen,
    dismissStopModal,
    flightProgress,
    latestStickTask,
    currentStickTaskId,
    stickRefreshBusy,
    rawImageUrl,
    modelInputUrl,
    rawMeta,
    modelInputMeta,
    logLines,
    autoScrollLogs,
    setAutoScrollLogs,
    clearLogs,
    settingsBusy,
    onRefreshRtmp,
    onInference,
    onLoginFlightSystem,
    onRegisterDevice,
    onAcquireFlightControl,
    onRunOneStep,
    onRefreshStickTask,
    onEmergencyStop,
  };
}

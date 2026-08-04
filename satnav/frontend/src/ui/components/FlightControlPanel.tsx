import { useState } from "react";

import type { AircraftPose } from "../hooks/useConsoleController";
import { LoginFlightModal, type LoginFlightForm } from "./LoginFlightModal";
import { RegisterDeviceModal } from "./RegisterDeviceModal";

interface FlightControlPanelProps {
  busy: boolean;
  loggedIn: boolean;
  drcReady: boolean;
  aircraftPose: AircraftPose;
  defaultRcSn: string;
  defaultDeviceSn: string;
  defaultLoginUsername: string;
  defaultLoginPassword: string;
  defaultLoginFlag: number;
  onRegisterDevice: (rcSn: string, deviceSn: string) => Promise<void>;
  onLogin: (form: LoginFlightForm) => Promise<void>;
  onAcquireControl: () => void;
  className?: string;
}

function formatCoord(value: number | null): string {
  return value === null ? "—" : value.toFixed(6);
}

function formatHeading(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)}°`;
}

function formatHeight(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)} m`;
}

function formatReceivedAtMs(value: number | null): string {
  if (value === null) {
    return "—";
  }
  return new Date(value).toLocaleString("zh-CN", {
    hour12: false,
    timeZone: "Asia/Shanghai",
  });
}

function formatSn(value: string | null): string {
  return value?.trim() ? value : "—";
}

function resolveHint(loggedIn: boolean, drcReady: boolean, deviceBound: boolean): string {
  if (drcReady) {
    return "飞行控制已就绪";
  }
  if (loggedIn) {
    return "已登录，待获取飞行控制";
  }
  if (!deviceBound) {
    return "请先完成摇控/无人机绑定";
  }
  return "请先登录飞控系统";
}

function isDeviceBound(aircraftPose: AircraftPose): boolean {
  return Boolean(aircraftPose.rcSn?.trim() && aircraftPose.deviceSn?.trim());
}

export function FlightControlPanel({
  busy,
  loggedIn,
  drcReady,
  aircraftPose,
  defaultRcSn,
  defaultDeviceSn,
  defaultLoginUsername,
  defaultLoginPassword,
  defaultLoginFlag,
  onRegisterDevice,
  onLogin,
  onAcquireControl,
  className,
}: FlightControlPanelProps) {
  const [registerOpen, setRegisterOpen] = useState(false);
  const [registerBusy, setRegisterBusy] = useState(false);
  const [loginOpen, setLoginOpen] = useState(false);
  const deviceBound = isDeviceBound(aircraftPose);

  const handleRegisterSubmit = async (rcSn: string, deviceSn: string) => {
    setRegisterBusy(true);
    try {
      await onRegisterDevice(rcSn, deviceSn);
      setRegisterOpen(false);
    } finally {
      setRegisterBusy(false);
    }
  };

  return (
    <>
      <section
        className={["panel", "flight-control-panel", className].filter(Boolean).join(" ")}
      >
        <div className="flight-control-panel__head">
          <div className="flight-control-panel__title">飞控连接</div>
          <div className="flight-control-panel__hint">
            {resolveHint(loggedIn, drcReady, deviceBound)}
          </div>
        </div>

        <div className="flight-control-panel__actions">
          <button
            type="button"
            className="btn btn-secondary"
            disabled={busy}
            onClick={() => setRegisterOpen(true)}
          >
            摇控/无人机绑定
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={busy || !deviceBound}
            title={deviceBound ? undefined : "请先完成摇控/无人机绑定"}
            onClick={() => setLoginOpen(true)}
          >
            登陆飞控系统
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={busy || !loggedIn}
            onClick={onAcquireControl}
          >
            获取飞行控制
          </button>
        </div>

        <div className="flight-control-panel__status">
          <div className="flight-control-panel__status-head">
            <div className="flight-control-panel__status-title">飞行器当前状态</div>
            <div className="flight-control-panel__status-time mono">
              {formatReceivedAtMs(aircraftPose.receivedAtMs)}
            </div>
          </div>
          <div className="flight-control-panel__status-grid mono">
            <div className="flight-control-panel__status-item">
              <span className="label">纬度</span>
              <span className="value">{formatCoord(aircraftPose.latitude)}</span>
            </div>
            <div className="flight-control-panel__status-item">
              <span className="label">经度</span>
              <span className="value">{formatCoord(aircraftPose.longitude)}</span>
            </div>
            <div className="flight-control-panel__status-item">
              <span className="label">航向</span>
              <span className="value">{formatHeading(aircraftPose.heading)}</span>
            </div>
            <div className="flight-control-panel__status-item">
              <span className="label">对地高</span>
              <span className="value">{formatHeight(aircraftPose.height)}</span>
            </div>
            <div className="flight-control-panel__status-item">
              <span className="label">摇控SN</span>
              <span className="value" title={aircraftPose.rcSn ?? undefined}>
                {formatSn(aircraftPose.rcSn)}
              </span>
            </div>
            <div className="flight-control-panel__status-item">
              <span className="label">无人机SN</span>
              <span className="value" title={aircraftPose.deviceSn ?? undefined}>
                {formatSn(aircraftPose.deviceSn)}
              </span>
            </div>
          </div>
        </div>

        <style>{`
          .flight-control-panel {
            padding: 0.85rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            height: 100%;
            min-height: 0;
          }
          .flight-control-panel__head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.75rem;
          }
          .flight-control-panel__title {
            font-weight: 600;
            font-size: 0.95rem;
            flex-shrink: 0;
          }
          .flight-control-panel__hint {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-align: right;
            line-height: 1.35;
          }
          .flight-control-panel__actions {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.4rem;
          }
          .btn {
            border: none;
            border-radius: 10px;
            padding: 0.58rem 0.35rem;
            font-weight: 600;
            font-size: 0.72rem;
            line-height: 1.25;
          }
          .btn:disabled {
            opacity: 0.55;
            cursor: not-allowed;
          }
          .btn-secondary {
            background: #173247;
            color: var(--text);
            border: 1px solid var(--border);
          }
          .flight-control-panel__status {
            border-radius: 10px;
            border: 1px solid var(--border);
            background: #08131d;
            padding: 0.6rem 0.75rem;
          }
          .flight-control-panel__status-title {
            font-size: 0.78rem;
            color: var(--text-muted);
          }
          .flight-control-panel__status-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.75rem;
            margin-bottom: 0.4rem;
          }
          .flight-control-panel__status-time {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-align: right;
            flex-shrink: 0;
          }
          .flight-control-panel__status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.35rem 0.65rem;
            font-size: 0.78rem;
            line-height: 1.35;
          }
          .flight-control-panel__status-item {
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            min-width: 0;
          }
          .flight-control-panel__status-grid .label {
            color: var(--text-muted);
            flex-shrink: 0;
          }
          .flight-control-panel__status-grid .value {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
        `}</style>
      </section>

      <RegisterDeviceModal
        open={registerOpen}
        busy={registerBusy || busy}
        initialRcSn={aircraftPose.rcSn ?? defaultRcSn}
        initialDeviceSn={aircraftPose.deviceSn ?? defaultDeviceSn}
        onClose={() => setRegisterOpen(false)}
        onSubmit={handleRegisterSubmit}
      />

      <LoginFlightModal
        open={loginOpen}
        busy={busy}
        loggedIn={loggedIn}
        initialUsername={defaultLoginUsername}
        initialPassword={defaultLoginPassword}
        initialFlag={defaultLoginFlag}
        onClose={() => setLoginOpen(false)}
        onSubmit={onLogin}
      />
    </>
  );
}

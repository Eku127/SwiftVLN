import { useEffect, useRef, useState } from "react";

import { SettingsIcon } from "./SettingsIcon";

interface HeaderProps {
  apiHealthy: boolean;
  modelReady: boolean;
  rtmpActive: boolean;
  flightBackendHealthy: boolean;
  sessionId: string | null;
  phaseLabel: string;
  settingsBusy?: boolean;
  onRefreshRtmp: () => void;
  className?: string;
}

export function Header({
  apiHealthy,
  modelReady,
  rtmpActive,
  flightBackendHealthy,
  sessionId,
  phaseLabel,
  settingsBusy = false,
  onRefreshRtmp,
  className,
}: HeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) {
      return;
    }

    const onPointerDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };

    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, [menuOpen]);

  const handleRefreshRtmp = () => {
    setMenuOpen(false);
    onRefreshRtmp();
  };

  return (
    <header className={["panel", "header", className].filter(Boolean).join(" ")}>
      <div className="header__brand">
        <div className="header__logo">◎</div>
        <div>
          <div className="header__title">SatNav</div>
          <div className="header__subtitle">SwiftVLN 实机导航控制台</div>
        </div>
      </div>

      <div className="header__badges">
        <span className={`badge ${apiHealthy ? "ok" : "warn"}`}>
          系统 {apiHealthy ? "正常" : "异常"}
        </span>
        <span className={`badge ${modelReady ? "ok" : "warn"}`}>
          推理服务 {modelReady ? "已就绪" : "启动中"}
        </span>
        <span className={`badge ${rtmpActive ? "ok" : "warn"}`}>
          RTMP {rtmpActive ? "已连接" : "未连接"}
        </span>
        <span className={`badge ${flightBackendHealthy ? "ok" : "warn"}`}>
          飞控 {flightBackendHealthy ? "已连接" : "未连接"}
        </span>
      </div>

      <div className="header__tools">
        <div className="header__settings" ref={menuRef}>
          <button
            type="button"
            className="header__gear-btn"
            aria-label="设置"
            aria-expanded={menuOpen}
            disabled={settingsBusy}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <SettingsIcon />
          </button>

          {menuOpen ? (
            <div className="header__dropdown" role="menu">
              <button
                type="button"
                className="header__dropdown-item"
                role="menuitem"
                disabled={settingsBusy}
                onClick={handleRefreshRtmp}
              >
                刷新 RTMP 服务
              </button>
            </div>
          ) : null}
        </div>

        <div className="header__meta mono">
          <div className="header__session" title={sessionId ?? undefined}>
            Session ID: {sessionId ?? "—"}
          </div>
          <div className="header__phase">{phaseLabel}</div>
        </div>
      </div>

      <style>{`
        .header {
          display: flex;
          align-items: center;
          gap: 1.25rem;
          padding: 0.9rem 1.1rem;
          flex-wrap: wrap;
        }
        .header__brand {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          min-width: 220px;
        }
        .header__logo {
          width: 2.2rem;
          height: 2.2rem;
          border-radius: 999px;
          display: grid;
          place-items: center;
          background: rgba(31, 212, 122, 0.15);
          color: var(--accent);
          font-size: 1.1rem;
        }
        .header__title {
          font-size: 1.35rem;
          font-weight: 700;
        }
        .header__subtitle {
          color: var(--text-muted);
          font-size: 0.85rem;
        }
        .header__badges {
          display: flex;
          gap: 0.5rem;
          flex-wrap: wrap;
          flex: 1;
        }
        .header__tools {
          display: flex;
          align-items: flex-start;
          gap: 0.55rem;
          margin-left: auto;
        }
        .header__settings {
          position: relative;
          flex-shrink: 0;
        }
        .header__gear-btn {
          width: 2rem;
          height: 2rem;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: #102030;
          color: #9eb6c8;
          display: grid;
          place-items: center;
          padding: 0;
        }
        .header__gear-btn:hover:not(:disabled) {
          color: var(--text);
          border-color: #2d5068;
          background: #152a3d;
        }
        .header__gear-btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
        .header__dropdown {
          position: absolute;
          top: calc(100% + 0.35rem);
          right: 0;
          min-width: 10.5rem;
          padding: 0.35rem;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #0d1a24;
          box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
          z-index: 20;
        }
        .header__dropdown-item {
          width: 100%;
          border: none;
          border-radius: 8px;
          background: transparent;
          color: var(--text);
          text-align: left;
          padding: 0.55rem 0.65rem;
          font-size: 0.84rem;
        }
        .header__dropdown-item:hover:not(:disabled) {
          background: #152a3d;
        }
        .header__dropdown-item:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
        .header__meta {
          text-align: right;
          color: var(--text-muted);
          font-size: 0.82rem;
          min-width: 0;
        }
        .header__session {
          max-width: min(22rem, 42vw);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .header__phase {
          color: var(--info);
          margin-top: 0.15rem;
        }
      `}</style>
    </header>
  );
}

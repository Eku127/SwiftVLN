import { useEffect, useRef } from "react";
import type { LogLine } from "../hooks/useConsoleController";

interface LogPanelProps {
  lines: LogLine[];
  autoScroll: boolean;
  onAutoScrollChange: (value: boolean) => void;
  onClear: () => void;
  className?: string;
}

export function LogPanel({
  lines,
  autoScroll,
  onAutoScrollChange,
  onClear,
  className,
}: LogPanelProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [lines, autoScroll]);

  return (
    <section className={["panel", "log-panel", className].filter(Boolean).join(" ")}>
      <div className="log-panel__toolbar">
        <div className="log-panel__title">实时过程日志</div>
        <div className="log-panel__actions">
          <label className="log-panel__toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => onAutoScrollChange(e.target.checked)}
            />
            自动滚动
          </label>
          <button type="button" className="log-panel__btn" onClick={onClear}>
            清空
          </button>
          <span className="badge ok">LIVE</span>
        </div>
      </div>

      <div className="log-panel__body mono">
        {lines.length === 0 ? (
          <div className="log-panel__empty">等待日志...</div>
        ) : (
          lines.map((line) => (
            <div
              key={line.id}
              className={`log-line log-line--${line.level ?? "info"}`}
            >
              <span className="log-line__time">[{line.timestamp}]</span>
              <span className="log-line__tag">[{line.tag}]</span>
              <span>{line.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      <style>{`
        .log-panel {
          padding: 0.85rem 1rem 1rem;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          min-height: 0;
          height: 100%;
        }
        .log-panel__toolbar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 0.55rem;
          flex-wrap: wrap;
        }
        .log-panel__title {
          font-weight: 600;
        }
        .log-panel__actions {
          display: flex;
          align-items: center;
          gap: 0.65rem;
        }
        .log-panel__toggle {
          display: flex;
          gap: 0.35rem;
          align-items: center;
          color: var(--text-muted);
          font-size: 0.82rem;
        }
        .log-panel__btn {
          border: 1px solid var(--border);
          background: #102030;
          color: var(--text);
          border-radius: 8px;
          padding: 0.25rem 0.55rem;
          font-size: 0.8rem;
        }
        .log-panel__body {
          flex: 1 1 auto;
          min-height: 0;
          overflow: auto;
          background: #050c12;
          border: 1px solid var(--border);
          border-radius: 10px;
          padding: 0.65rem 0.75rem;
          font-size: 0.78rem;
          line-height: 1.45;
        }
        .log-panel__empty {
          color: var(--text-muted);
        }
        .log-line {
          margin-bottom: 0.2rem;
        }
        .log-line--warn {
          color: #ffd98a;
        }
        .log-line--error {
          color: #ff9aa2;
        }
        .log-line__time,
        .log-line__tag {
          color: #7fa0b8;
          margin-right: 0.35rem;
        }
      `}</style>
    </section>
  );
}

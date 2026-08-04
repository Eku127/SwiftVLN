import { useEffect, useState } from "react";

interface RegisterDeviceModalProps {
  open: boolean;
  busy: boolean;
  initialRcSn: string;
  initialDeviceSn: string;
  onClose: () => void;
  onSubmit: (rcSn: string, deviceSn: string) => void;
}

export function RegisterDeviceModal({
  open,
  busy,
  initialRcSn,
  initialDeviceSn,
  onClose,
  onSubmit,
}: RegisterDeviceModalProps) {
  const [rcSn, setRcSn] = useState(initialRcSn);
  const [deviceSn, setDeviceSn] = useState(initialDeviceSn);

  useEffect(() => {
    if (open) {
      setRcSn(initialRcSn);
      setDeviceSn(initialDeviceSn);
    }
  }, [open, initialRcSn, initialDeviceSn]);

  if (!open) {
    return null;
  }

  const handleSubmit = () => {
    const trimmedRc = rcSn.trim();
    const trimmedDevice = deviceSn.trim();
    if (!trimmedRc || !trimmedDevice) {
      return;
    }
    onSubmit(trimmedRc, trimmedDevice);
  };

  return (
    <div className="register-modal__backdrop" onClick={onClose}>
      <div
        className="register-modal panel"
        role="dialog"
        aria-labelledby="register-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="register-modal__head">
          <h2 id="register-modal-title" className="register-modal__title">
            摇控 / 无人机绑定
          </h2>
          <button
            type="button"
            className="register-modal__close"
            aria-label="关闭"
            disabled={busy}
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <p className="register-modal__hint">
          登记遥控器 SN 与飞行器 device SN，供后续获取飞行控制使用。
        </p>

        <label className="register-modal__field">
          <span className="register-modal__label">摇控 SN（rc_sn）</span>
          <input
            className="register-modal__input mono"
            value={rcSn}
            disabled={busy}
            placeholder="例如 RCPLUS2_SN_EXAMPLE"
            onChange={(event) => setRcSn(event.target.value)}
          />
        </label>

        <label className="register-modal__field">
          <span className="register-modal__label">无人机 SN（device_sn）</span>
          <input
            className="register-modal__input mono"
            value={deviceSn}
            disabled={busy}
            placeholder="例如 AIRCRAFT_SN_EXAMPLE"
            onChange={(event) => setDeviceSn(event.target.value)}
          />
        </label>

        <div className="register-modal__actions">
          <button
            type="button"
            className="register-modal__btn register-modal__btn--ghost"
            disabled={busy}
            onClick={onClose}
          >
            取消
          </button>
          <button
            type="button"
            className="register-modal__btn register-modal__btn--primary"
            disabled={busy || !rcSn.trim() || !deviceSn.trim()}
            onClick={handleSubmit}
          >
            {busy ? "绑定中…" : "确认绑定"}
          </button>
        </div>
      </div>

      <style>{`
        .register-modal__backdrop {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: grid;
          place-items: center;
          padding: 1rem;
          background: rgba(3, 10, 16, 0.72);
          backdrop-filter: blur(2px);
        }
        .register-modal {
          width: min(100%, 26rem);
          padding: 1rem 1.1rem 1.1rem;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .register-modal__head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 0.75rem;
        }
        .register-modal__title {
          margin: 0;
          font-size: 1rem;
          font-weight: 600;
        }
        .register-modal__close {
          width: 1.75rem;
          height: 1.75rem;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: #102030;
          color: var(--text-muted);
          font-size: 1.1rem;
          line-height: 1;
          padding: 0;
        }
        .register-modal__close:hover:not(:disabled) {
          color: var(--text);
        }
        .register-modal__hint {
          margin: 0;
          font-size: 0.78rem;
          line-height: 1.45;
          color: var(--text-muted);
        }
        .register-modal__field {
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }
        .register-modal__label {
          font-size: 0.78rem;
          color: var(--text-muted);
        }
        .register-modal__input {
          width: 100%;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #08131d;
          color: var(--text);
          padding: 0.62rem 0.75rem;
          font-size: 0.84rem;
        }
        .register-modal__input:disabled {
          opacity: 0.6;
        }
        .register-modal__actions {
          display: flex;
          justify-content: flex-end;
          gap: 0.5rem;
          margin-top: 0.15rem;
        }
        .register-modal__btn {
          border: none;
          border-radius: 10px;
          padding: 0.58rem 1rem;
          font-weight: 600;
          font-size: 0.86rem;
        }
        .register-modal__btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
        .register-modal__btn--ghost {
          background: #173247;
          color: var(--text);
          border: 1px solid var(--border);
        }
        .register-modal__btn--primary {
          background: linear-gradient(180deg, #2be98f, #17b86a);
          color: #042414;
        }
      `}</style>
    </div>
  );
}

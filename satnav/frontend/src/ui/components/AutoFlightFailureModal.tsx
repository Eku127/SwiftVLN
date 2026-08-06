/** Modal when a non-TIMEOUT stick-task fails during auto flight. */
interface AutoFlightFailureModalProps {
  open: boolean;
  onRetry: () => void;
  onSkip: () => void;
}

export function AutoFlightFailureModal({
  open,
  onRetry,
  onSkip,
}: AutoFlightFailureModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="auto-flight-failure-modal" role="dialog" aria-modal="true">
      <div className="auto-flight-failure-modal__backdrop" />
      <div className="auto-flight-failure-modal__card">
        <div className="auto-flight-failure-modal__title">飞控动作执行失败</div>
        <p className="auto-flight-failure-modal__body">
          当前飞控动作执行失败（非 TIMEOUT）。请选择重新执行该动作，或跳过并继续后续流程。
        </p>
        <div className="auto-flight-failure-modal__actions">
          <button type="button" className="auto-flight-failure-modal__btn" onClick={onRetry}>
            重新执行
          </button>
          <button
            type="button"
            className="auto-flight-failure-modal__btn auto-flight-failure-modal__btn--secondary"
            onClick={onSkip}
          >
            跳过
          </button>
        </div>
      </div>

      <style>{`
        .auto-flight-failure-modal {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: grid;
          place-items: center;
          padding: 1rem;
        }
        .auto-flight-failure-modal__backdrop {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.55);
        }
        .auto-flight-failure-modal__card {
          position: relative;
          width: min(28rem, 100%);
          border-radius: 12px;
          border: 1px solid #f0b429;
          background: #1a1510;
          padding: 1.1rem 1.15rem;
          box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
        }
        .auto-flight-failure-modal__title {
          font-size: 1.05rem;
          font-weight: 700;
          color: #f0b429;
          margin-bottom: 0.55rem;
        }
        .auto-flight-failure-modal__body {
          margin: 0 0 1rem;
          color: #e8f1f8;
          line-height: 1.55;
          font-size: 0.92rem;
        }
        .auto-flight-failure-modal__actions {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.55rem;
        }
        .auto-flight-failure-modal__btn {
          border: none;
          border-radius: 10px;
          padding: 0.6rem 0.9rem;
          font-weight: 600;
          background: linear-gradient(180deg, #2be98f, #17b86a);
          color: #042414;
        }
        .auto-flight-failure-modal__btn--secondary {
          background: #1e3344;
          color: #c8dce8;
          border: 1px solid #3d5f78;
        }
      `}</style>
    </div>
  );
}

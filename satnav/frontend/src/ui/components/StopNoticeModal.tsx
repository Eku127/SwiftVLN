/** Modal shown after STOP (model next_action=0 or emergency STOP takes effect). */
interface StopNoticeModalProps {
  /** Whether the dialog is visible. */
  open: boolean;
  /** Dismiss handler (backdrop or confirm button). */
  onClose: () => void;
}

/**
 * Warns that the RC link is down and flight control must be re-acquired.
 */
export function StopNoticeModal({ open, onClose }: StopNoticeModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="stop-notice-modal" role="dialog" aria-modal="true">
      <div className="stop-notice-modal__backdrop" onClick={onClose} />
      <div className="stop-notice-modal__card">
        <div className="stop-notice-modal__title">STOP</div>
        <p className="stop-notice-modal__body">
          摇控端断开连接，后续飞控需重新获取飞行控制
        </p>
        <button type="button" className="stop-notice-modal__btn" onClick={onClose}>
          知道了
        </button>
      </div>

      <style>{`
        .stop-notice-modal {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: grid;
          place-items: center;
          padding: 1rem;
        }
        .stop-notice-modal__backdrop {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.55);
        }
        .stop-notice-modal__card {
          position: relative;
          width: min(24rem, 100%);
          border-radius: 12px;
          border: 1px solid #ff6b76;
          background: #1a1014;
          padding: 1.1rem 1.15rem;
          box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
        }
        .stop-notice-modal__title {
          font-size: 1.05rem;
          font-weight: 700;
          color: #ff8a93;
          margin-bottom: 0.55rem;
        }
        .stop-notice-modal__body {
          margin: 0 0 1rem;
          color: #e8f1f8;
          line-height: 1.55;
          font-size: 0.92rem;
        }
        .stop-notice-modal__btn {
          width: 100%;
          border: none;
          border-radius: 10px;
          padding: 0.6rem 0.9rem;
          font-weight: 600;
          background: #e52f41;
          color: white;
        }
      `}</style>
    </div>
  );
}

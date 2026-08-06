/** Modal shown when enabling auto flight control without DRC ready. */
interface AutoFlightPrereqModalProps {
  open: boolean;
  onClose: () => void;
}

export function AutoFlightPrereqModal({ open, onClose }: AutoFlightPrereqModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="auto-flight-prereq-modal" role="dialog" aria-modal="true">
      <div className="auto-flight-prereq-modal__backdrop" onClick={onClose} />
      <div className="auto-flight-prereq-modal__card">
        <div className="auto-flight-prereq-modal__title">无法开启全自动飞控</div>
        <p className="auto-flight-prereq-modal__body">
          开启前请先完成飞控准备：在飞控面板依次点击「登陆飞控系统」和「获取飞行控制」，待飞行控制就绪后再开启全自动飞控。
        </p>
        <button type="button" className="auto-flight-prereq-modal__btn" onClick={onClose}>
          知道了
        </button>
      </div>

      <style>{`
        .auto-flight-prereq-modal {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: grid;
          place-items: center;
          padding: 1rem;
        }
        .auto-flight-prereq-modal__backdrop {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.55);
        }
        .auto-flight-prereq-modal__card {
          position: relative;
          width: min(26rem, 100%);
          border-radius: 12px;
          border: 1px solid #2d6f97;
          background: #0d1a24;
          padding: 1.1rem 1.15rem;
          box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
        }
        .auto-flight-prereq-modal__title {
          font-size: 1.05rem;
          font-weight: 700;
          color: #9fe0ff;
          margin-bottom: 0.55rem;
        }
        .auto-flight-prereq-modal__body {
          margin: 0 0 1rem;
          color: #e8f1f8;
          line-height: 1.55;
          font-size: 0.92rem;
        }
        .auto-flight-prereq-modal__btn {
          width: 100%;
          border: none;
          border-radius: 10px;
          padding: 0.6rem 0.9rem;
          font-weight: 600;
          background: linear-gradient(180deg, #2be98f, #17b86a);
          color: #042414;
        }
      `}</style>
    </div>
  );
}

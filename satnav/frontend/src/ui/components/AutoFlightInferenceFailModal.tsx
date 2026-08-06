/** Modal when auto-triggered inference fails. */
interface AutoFlightInferenceFailModalProps {
  open: boolean;
  onClose: () => void;
}

export function AutoFlightInferenceFailModal({
  open,
  onClose,
}: AutoFlightInferenceFailModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="auto-flight-infer-fail-modal" role="dialog" aria-modal="true">
      <div className="auto-flight-infer-fail-modal__backdrop" onClick={onClose} />
      <div className="auto-flight-infer-fail-modal__card">
        <div className="auto-flight-infer-fail-modal__title">全自动推理失败</div>
        <p className="auto-flight-infer-fail-modal__body">
          自动推理请求失败。请检查模型推理模块、RTMP 抽帧与 Deploy 服务是否正常，确认后改用手动模式继续操作。系统已触发应急
          STOP。
        </p>
        <button type="button" className="auto-flight-infer-fail-modal__btn" onClick={onClose}>
          知道了
        </button>
      </div>

      <style>{`
        .auto-flight-infer-fail-modal {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: grid;
          place-items: center;
          padding: 1rem;
        }
        .auto-flight-infer-fail-modal__backdrop {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.55);
        }
        .auto-flight-infer-fail-modal__card {
          position: relative;
          width: min(28rem, 100%);
          border-radius: 12px;
          border: 1px solid #ff6b76;
          background: #1a1014;
          padding: 1.1rem 1.15rem;
          box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
        }
        .auto-flight-infer-fail-modal__title {
          font-size: 1.05rem;
          font-weight: 700;
          color: #ff8a93;
          margin-bottom: 0.55rem;
        }
        .auto-flight-infer-fail-modal__body {
          margin: 0 0 1rem;
          color: #e8f1f8;
          line-height: 1.55;
          font-size: 0.92rem;
        }
        .auto-flight-infer-fail-modal__btn {
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

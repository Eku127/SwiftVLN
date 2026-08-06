interface InstructionPanelProps {
  instruction: string;
  onInstructionChange: (value: string) => void;
  maxLength: number;
  inferenceBusy: boolean;
  /** Whether Inference is enabled (canRunInference; separate from inferenceBusy). */
  inferenceEnabled: boolean;
  /** Calls POST /api/satnav/model/inference only. */
  onInference: () => void;
  className?: string;
}

export function InstructionPanel({
  instruction,
  onInstructionChange,
  maxLength,
  inferenceBusy,
  inferenceEnabled,
  onInference,
  className,
}: InstructionPanelProps) {
  return (
    <section
      className={["panel", "instruction-panel", className].filter(Boolean).join(" ")}
    >
      <div className="instruction-panel__title">导航指令</div>
      <div className="instruction-panel__input-wrap">
        <textarea
          className="instruction-panel__input"
          value={instruction}
          maxLength={maxLength}
          onChange={(event) => onInstructionChange(event.target.value)}
          rows={3}
          spellCheck={false}
        />
        <div className="instruction-panel__input-bar">
          <span className="instruction-panel__count mono">
            {instruction.length} / {maxLength}
          </span>
          <button
            type="button"
            className="instruction-panel__infer-btn"
            disabled={inferenceBusy || !inferenceEnabled}
            onClick={onInference}
          >
            推理
          </button>
        </div>
      </div>

      <style>{`
        .instruction-panel {
          padding: 0.85rem 1rem;
          height: 100%;
          display: flex;
          flex-direction: column;
          min-height: 0;
        }
        .instruction-panel__title {
          font-weight: 600;
          font-size: 1rem;
          line-height: 1.25;
          margin-bottom: 0.45rem;
        }
        .instruction-panel__input-wrap {
          position: relative;
          flex: 1 1 auto;
          min-height: 5.5rem;
          display: flex;
          min-width: 0;
        }
        .instruction-panel__input {
          width: 100%;
          resize: none;
          flex: 1 1 auto;
          min-height: 5.5rem;
          overflow-y: auto;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #08131d;
          color: var(--text);
          padding: 0.55rem 0.75rem 2.35rem;
          font-size: 1rem;
          line-height: 1.5;
        }
        .instruction-panel__input-bar {
          position: absolute;
          left: 0.45rem;
          right: 0.45rem;
          bottom: 0.4rem;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          padding: 0.2rem 0.35rem;
          border-radius: 0 0 8px 8px;
          background: linear-gradient(
            180deg,
            rgba(8, 19, 29, 0) 0%,
            rgba(8, 19, 29, 0.92) 35%,
            #08131d 100%
          );
          pointer-events: none;
        }
        .instruction-panel__count {
          color: var(--text-muted);
          font-size: 0.72rem;
          line-height: 1;
          white-space: nowrap;
          font-variant-numeric: tabular-nums;
          pointer-events: auto;
        }
        .instruction-panel__infer-btn {
          border: none;
          border-radius: 8px;
          padding: 0.38rem 1rem;
          font-weight: 600;
          font-size: 1rem;
          line-height: 1.25;
          flex-shrink: 0;
          pointer-events: auto;
          background: linear-gradient(180deg, #2be98f, #17b86a);
          color: #042414;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
        }
        .instruction-panel__infer-btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
      `}</style>
    </section>
  );
}

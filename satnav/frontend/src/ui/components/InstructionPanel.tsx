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
      <textarea
        className="instruction-panel__input"
        value={instruction}
        maxLength={maxLength}
        onChange={(event) => onInstructionChange(event.target.value)}
        rows={3}
      />
      <div className="instruction-panel__footer">
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

      <style>{`
        .instruction-panel {
          padding: 1rem 1.1rem;
          height: 100%;
          display: flex;
          flex-direction: column;
          min-height: 0;
        }
        .instruction-panel__title {
          font-weight: 600;
          margin-bottom: 0.65rem;
        }
        .instruction-panel__input {
          width: 100%;
          resize: none;
          flex: 1 1 auto;
          min-height: 5.5rem;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #08131d;
          color: var(--text);
          padding: 0.75rem 0.85rem;
          line-height: 1.5;
        }
        .instruction-panel__footer {
          margin-top: 0.55rem;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
        }
        .instruction-panel__count {
          color: var(--text-muted);
          font-size: 0.8rem;
        }
        .instruction-panel__infer-btn {
          border: none;
          border-radius: 10px;
          padding: 0.55rem 1.35rem;
          font-weight: 600;
          font-size: 0.9rem;
          background: linear-gradient(180deg, #2be98f, #17b86a);
          color: #042414;
        }
        .instruction-panel__infer-btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
      `}</style>
    </section>
  );
}

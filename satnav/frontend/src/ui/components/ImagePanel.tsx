import type { CSSProperties } from "react";

interface ImagePanelProps {
  title: string;
  subtitle?: string;
  imageUrl: string;
  badge?: string;
  placeholder?: string;
  className?: string;
  /** raw：按原图比例铺满视口；model：中间黑色区为正方形，宽度可调 */
  variant?: "raw" | "model";
  /** model 正方形边长，如 "280px" / "16rem"；默认随面板高度取最大正方形 */
  squareSize?: string;
}

export function ImagePanel({
  title,
  subtitle,
  imageUrl,
  badge,
  placeholder,
  className,
  variant = "raw",
  squareSize,
}: ImagePanelProps) {
  const style: CSSProperties | undefined =
    variant === "model" && squareSize
      ? ({ "--model-square-size": squareSize } as CSSProperties)
      : undefined;

  return (
    <section
      className={[
        "panel",
        "image-panel",
        `image-panel--${variant}`,
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      style={style}
    >
      <div className="image-panel__head">
        <div className="image-panel__title">{title}</div>
        {badge ? <span className="badge ok">{badge}</span> : null}
      </div>

      <div className="image-panel__viewport">
        <div className="image-panel__frame">
          {imageUrl ? (
            <img src={imageUrl} alt={title} className="image-panel__img" />
          ) : (
            <div className="image-panel__placeholder">{placeholder}</div>
          )}
        </div>
      </div>

      {subtitle ? <div className="image-panel__meta mono">{subtitle}</div> : null}

      <style>{`
        .image-panel {
          padding: 0.75rem;
          display: flex;
          flex-direction: column;
          gap: 0.45rem;
          height: 100%;
          min-height: 0;
          box-sizing: border-box;
        }
        .image-panel__head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 0.5rem;
          flex: 0 0 auto;
        }
        .image-panel__title {
          font-weight: 600;
          font-size: 0.9rem;
        }
        .image-panel__viewport {
          flex: 1 1 0;
          min-height: 8rem;
          container-type: size;
          display: grid;
          place-items: center;
        }
        .image-panel__frame {
          border-radius: 10px;
          overflow: hidden;
          border: 1px solid var(--border);
          background: #050d14;
          display: grid;
          place-items: center;
        }
        .image-panel__img {
          width: 100%;
          height: 100%;
          display: block;
        }
        .image-panel__placeholder {
          color: var(--text-muted);
          font-size: 0.85rem;
          text-align: center;
          padding: 0.5rem;
        }
        .image-panel__meta {
          color: var(--text-muted);
          font-size: 0.72rem;
          word-break: break-all;
          flex: 0 0 auto;
        }

        /* —— raw：中间显示区铺满，保持原图比例 —— */
        .image-panel--raw .image-panel__frame {
          width: 100%;
          height: 100%;
        }
        .image-panel--raw .image-panel__img {
          object-fit: contain;
        }

        /* —— model：边长 = 视口高度（与同排 raw 显示区同高）—— */
        .image-panel--model .image-panel__frame {
          width: var(--model-square-size, 100cqh);
          height: var(--model-square-size, 100cqh);
          max-height: 100%;
          aspect-ratio: 1 / 1;
          box-sizing: border-box;
        }
        .image-panel--model .image-panel__img {
          object-fit: contain;
        }
      `}</style>
    </section>
  );
}

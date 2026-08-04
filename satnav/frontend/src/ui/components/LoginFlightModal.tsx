import { useEffect, useState } from "react";

import { ApiError } from "@/api";

export interface LoginFlightForm {
  username: string;
  password: string;
  flag: number;
}

interface LoginFlightModalProps {
  open: boolean;
  busy: boolean;
  loggedIn: boolean;
  initialUsername: string;
  initialPassword: string;
  initialFlag: number;
  onClose: () => void;
  onSubmit: (form: LoginFlightForm) => Promise<void>;
}

function formatSubmitError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.detail;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function LoginFlightModal({
  open,
  busy,
  loggedIn,
  initialUsername,
  initialPassword,
  initialFlag,
  onClose,
  onSubmit,
}: LoginFlightModalProps) {
  const [username, setUsername] = useState(initialUsername);
  const [password, setPassword] = useState(initialPassword);
  const [flag, setFlag] = useState(initialFlag);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const formBusy = busy || submitting;

  useEffect(() => {
    if (open) {
      setUsername(initialUsername);
      setPassword(initialPassword);
      setFlag(initialFlag);
      setError(null);
      setSubmitting(false);
    }
  }, [open, initialUsername, initialPassword, initialFlag]);

  if (!open) {
    return null;
  }

  const canSubmit =
    loggedIn || (username.trim().length > 0 && password.trim().length > 0);

  const clearError = () => {
    if (error) {
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!canSubmit || formBusy) {
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      await onSubmit({
        username: username.trim(),
        password: password.trim(),
        flag: Number.isFinite(flag) && flag >= 1 ? flag : 1,
      });
      onClose();
    } catch (submitError) {
      setError(formatSubmitError(submitError));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="login-modal__backdrop" onClick={onClose}>
      <div
        className="login-modal panel"
        role="dialog"
        aria-labelledby="login-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="login-modal__head">
          <h2 id="login-modal-title" className="login-modal__title">
            登陆飞控系统
          </h2>
          <button
            type="button"
            className="login-modal__close"
            aria-label="关闭"
            disabled={formBusy}
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <p className="login-modal__hint">
          {loggedIn
            ? "服务端已有 token 时将自动刷新，无需填写账号密码。"
            : "首次登录需填写 CloudSDK 账号密码；token 将缓存在服务端。"}
        </p>

        {error ? (
          <div className="login-modal__error" role="alert">
            {error}
          </div>
        ) : null}

        <label className="login-modal__field">
          <span className="login-modal__label">用户名（username）</span>
          <input
            className="login-modal__input"
            value={username}
            disabled={formBusy}
            placeholder="例如 adminPC"
            autoComplete="username"
            onChange={(event) => {
              setUsername(event.target.value);
              clearError();
            }}
          />
        </label>

        <label className="login-modal__field">
          <span className="login-modal__label">密码（password）</span>
          <input
            className="login-modal__input"
            type="password"
            value={password}
            disabled={formBusy}
            placeholder="CloudSDK 密码"
            autoComplete="current-password"
            onChange={(event) => {
              setPassword(event.target.value);
              clearError();
            }}
          />
        </label>

        <label className="login-modal__field">
          <span className="login-modal__label">登录标志（flag）</span>
          <input
            className="login-modal__input mono"
            type="number"
            min={1}
            step={1}
            value={flag}
            disabled={formBusy}
            onChange={(event) => {
              setFlag(Number(event.target.value));
              clearError();
            }}
          />
        </label>

        <div className="login-modal__actions">
          <button
            type="button"
            className="login-modal__btn login-modal__btn--ghost"
            disabled={formBusy}
            onClick={onClose}
          >
            取消
          </button>
          <button
            type="button"
            className="login-modal__btn login-modal__btn--primary"
            disabled={formBusy || !canSubmit}
            onClick={() => void handleSubmit()}
          >
            {formBusy ? "登录中…" : loggedIn ? "刷新 Token" : "确认登录"}
          </button>
        </div>
      </div>

      <style>{`
        .login-modal__backdrop {
          position: fixed;
          inset: 0;
          z-index: 1000;
          display: grid;
          place-items: center;
          padding: 1rem;
          background: rgba(3, 10, 16, 0.72);
          backdrop-filter: blur(2px);
        }
        .login-modal {
          width: min(100%, 26rem);
          padding: 1rem 1.1rem 1.1rem;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .login-modal__head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 0.75rem;
        }
        .login-modal__title {
          margin: 0;
          font-size: 1rem;
          font-weight: 600;
        }
        .login-modal__close {
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
        .login-modal__close:hover:not(:disabled) {
          color: var(--text);
        }
        .login-modal__hint {
          margin: 0;
          font-size: 0.78rem;
          line-height: 1.45;
          color: var(--text-muted);
        }
        .login-modal__error {
          margin: 0;
          padding: 0.55rem 0.7rem;
          border-radius: 8px;
          border: 1px solid rgba(255, 77, 90, 0.45);
          background: rgba(255, 77, 90, 0.12);
          color: #ffb3b8;
          font-size: 0.8rem;
          line-height: 1.45;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .login-modal__field {
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
        }
        .login-modal__label {
          font-size: 0.78rem;
          color: var(--text-muted);
        }
        .login-modal__input {
          width: 100%;
          border-radius: 10px;
          border: 1px solid var(--border);
          background: #08131d;
          color: var(--text);
          padding: 0.62rem 0.75rem;
          font-size: 0.84rem;
        }
        .login-modal__input:disabled {
          opacity: 0.6;
        }
        .login-modal__actions {
          display: flex;
          justify-content: flex-end;
          gap: 0.5rem;
          margin-top: 0.15rem;
        }
        .login-modal__btn {
          border: none;
          border-radius: 10px;
          padding: 0.58rem 1rem;
          font-weight: 600;
          font-size: 0.86rem;
        }
        .login-modal__btn:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }
        .login-modal__btn--ghost {
          background: #173247;
          color: var(--text);
          border: 1px solid var(--border);
        }
        .login-modal__btn--primary {
          background: linear-gradient(180deg, #2be98f, #17b86a);
          color: #042414;
        }
      `}</style>
    </div>
  );
}

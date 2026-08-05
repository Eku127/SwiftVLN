/** Panel vs session log limits (session buffer is for full export / file stream in step 2). */

export const PANEL_LOG_MAX_LINES = 400;
export const SESSION_LOG_MAX_LINES = 20_000;

export type OperatorEventName =
  | "inference_clicked"
  | "inference_done"
  | "inference_failed"
  | "execute_clicked"
  | "flight_submit"
  | "stick_status"
  | "stick_done"
  | "step_skipped"
  | "stop";

/** ISO-8601 timestamp in Asia/Shanghai for operator event payloads. */
export function formatIsoTimestampShanghai(date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    fractionalSecondDigits: 3,
  }).formatToParts(date);

  const get = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((part) => part.type === type)?.value ?? "00";

  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}:${get("second")}.${get("fractionalSecond")}+08:00`;
}

export function formatOperatorEventMessage(
  event: OperatorEventName,
  sessionId: string | null,
  payload: Record<string, unknown>,
): string {
  return JSON.stringify({
    ts: formatIsoTimestampShanghai(),
    event,
    session_id: sessionId,
    ...payload,
  });
}

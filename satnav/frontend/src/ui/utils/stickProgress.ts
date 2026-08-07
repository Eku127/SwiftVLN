import type { StickTaskResponse } from "@/api";

export function extractStickBackendData(
  stick: StickTaskResponse | null,
): Record<string, unknown> | null {
  if (!stick?.backend || typeof stick.backend !== "object") {
    return null;
  }
  const data = (stick.backend as { data?: unknown }).data;
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return null;
  }
  return data as Record<string, unknown>;
}

function readNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function resolveKind(data: Record<string, unknown>, stick: StickTaskResponse | null): string {
  return String(data.kind ?? stick?.kind ?? "").toUpperCase();
}

export function computeStickProgressPercent(
  data: Record<string, unknown> | null,
  stick: StickTaskResponse | null,
): number | null {
  if (!data) {
    return null;
  }

  const status = String(data.status ?? stick?.status ?? "").toUpperCase();
  if (status === "COMPLETED") {
    return 100;
  }

  const kind = resolveKind(data, stick);

  if (kind.includes("YAW")) {
    const degree = readNumber(data.degree);
    // head_delta_deg: remaining yaw to target (not already turned).
    const remainingDeg = readNumber(data.head_delta_deg);
    if (degree !== null && remainingDeg !== null) {
      const degreeAbs = Math.abs(degree);
      if (degreeAbs > 0) {
        const remainingAbs = Math.abs(remainingDeg);
        return Math.min(
          100,
          Math.max(0, ((degreeAbs - remainingAbs) / degreeAbs) * 100),
        );
      }
    }
  }

  if (kind.includes("PITCH")) {
    const distance = readNumber(data.distance_m);
    const remaining = readNumber(data.remaining_distance_m);
    if (distance !== null && distance > 0 && remaining !== null) {
      return Math.min(100, Math.max(0, ((distance - remaining) / distance) * 100));
    }
  }

  return null;
}

export interface StickProgressView {
  kind: string;
  status: string;
  progress: number | null;
  headline: string;
  metrics: string[];
}

export function formatStickBackendDetail(
  data: Record<string, unknown> | null,
  stick: StickTaskResponse | null,
): string | null {
  if (!data) {
    return null;
  }

  const kind = resolveKind(data, stick);
  const status = String(data.status ?? stick?.status ?? "—");
  const elapsed = readNumber(data.elapsed_ms);
  const elapsedLine =
    elapsed !== null ? `已耗时：${Math.round(elapsed)} ms` : null;

  if (kind.includes("YAW")) {
    const degree = readNumber(data.degree);
    const start = readNumber(data.start_head);
    const target = readNumber(data.target_head);
    const current = readNumber(data.current_head);
    const remainingDeg = readNumber(data.head_delta_deg);
    const lines = [
      "任务：航向转动",
      `状态：${status}`,
      degree !== null && start !== null && target !== null
        ? `目标转角：${degree.toFixed(1)}°（${start.toFixed(1)}° → ${target.toFixed(1)}°）`
        : degree !== null
          ? `目标转角：${degree.toFixed(1)}°`
          : null,
      current !== null && remainingDeg !== null
        ? `当前航向：${current.toFixed(1)}°（剩余 ${Math.abs(remainingDeg).toFixed(1)}°）`
        : current !== null
          ? `当前航向：${current.toFixed(1)}°`
          : null,
      elapsedLine,
    ];
    return lines.filter(Boolean).join("\n");
  }

  if (kind.includes("PITCH")) {
    const distance = readNumber(data.distance_m);
    const remaining = readNumber(data.remaining_distance_m);
    const lat = readNumber(data.current_latitude);
    const lng = readNumber(data.current_longitude);
    const head = readNumber(data.locked_head_deg);
    const lines = [
      "任务：直线前进",
      `状态：${status}`,
      distance !== null && remaining !== null
        ? `目标距离：${distance.toFixed(1)} m（剩余 ${remaining.toFixed(1)} m）`
        : distance !== null
          ? `目标距离：${distance.toFixed(1)} m`
          : null,
      lat !== null && lng !== null
        ? `当前位置：${lat.toFixed(5)}, ${lng.toFixed(5)}`
        : null,
      head !== null ? `锁定航向：${head.toFixed(1)}°` : null,
      elapsedLine,
    ];
    return lines.filter(Boolean).join("\n");
  }

  return `任务：${kind || "未知"}\n状态：${status}`;
}

export function buildStickProgressView(
  stick: StickTaskResponse | null,
): StickProgressView | null {
  const data = extractStickBackendData(stick);
  if (!data) {
    return null;
  }

  const kind = resolveKind(data, stick);
  const status = String(data.status ?? stick?.status ?? "—");
  const progress = computeStickProgressPercent(data, stick);
  const metrics: string[] = [];

  if (kind.includes("YAW")) {
    const degree = readNumber(data.degree);
    // head_delta_deg: remaining yaw to target (not already turned).
    const remainingDeg = readNumber(data.head_delta_deg);
    const current = readNumber(data.current_head);
    const target = readNumber(data.target_head);
    if (degree !== null && remainingDeg !== null) {
      metrics.push(
        `剩余 ${Math.abs(remainingDeg).toFixed(1)}° / 目标 ${Math.abs(degree).toFixed(1)}°`,
      );
    }
    if (current !== null && target !== null) {
      metrics.push(`current ${current.toFixed(1)}° → target ${target.toFixed(1)}°`);
    }
    const elapsed = readNumber(data.elapsed_ms);
    if (elapsed !== null) {
      metrics.push(`elapsed ${Math.round(elapsed)} ms`);
    }
  } else if (kind.includes("PITCH")) {
    const distance = readNumber(data.distance_m);
    const remaining = readNumber(data.remaining_distance_m);
    if (distance !== null && remaining !== null) {
      metrics.push(`剩余 ${remaining.toFixed(1)} m / ${distance.toFixed(1)} m`);
    }
    const lat = readNumber(data.current_latitude);
    const lng = readNumber(data.current_longitude);
    if (lat !== null && lng !== null) {
      metrics.push(`pos ${lat.toFixed(5)}, ${lng.toFixed(5)}`);
    }
    const head = readNumber(data.locked_head_deg);
    if (head !== null) {
      metrics.push(`locked head ${head.toFixed(1)}°`);
    }
    const elapsed = readNumber(data.elapsed_ms);
    if (elapsed !== null) {
      metrics.push(`elapsed ${Math.round(elapsed)} ms`);
    }
  } else {
    metrics.push(JSON.stringify(data));
  }

  return {
    kind: kind || "—",
    status,
    progress,
    headline: `${kind || "TASK"} · ${status}`,
    metrics,
  };
}

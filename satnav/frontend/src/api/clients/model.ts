import { getJson } from "../http";
import type { ModelLogsResponse, ModelStatusResponse } from "../types";

export const modelClient = {
  getStatus(): Promise<ModelStatusResponse> {
    return getJson<ModelStatusResponse>("/api/satnav/system/model/status");
  },

  getLogs(afterSequence = 0, limit = 200): Promise<ModelLogsResponse> {
    const params = new URLSearchParams({
      after_sequence: String(afterSequence),
      limit: String(limit),
    });
    return getJson<ModelLogsResponse>(
      `/api/satnav/system/model/logs?${params.toString()}`,
    );
  },
};

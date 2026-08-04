import { getJson } from "../http";
import type { HealthResponse } from "../types";

export const healthClient = {
  getHealth(): Promise<HealthResponse> {
    return getJson<HealthResponse>("/api/satnav/health");
  },
};

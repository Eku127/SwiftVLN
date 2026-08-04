import { getJson, postJson } from "../http";
import type { RtmpStatusResponse } from "../types";

export const rtmpClient = {
  getStatus(): Promise<RtmpStatusResponse> {
    return getJson<RtmpStatusResponse>("/api/satnav/system/rtmp/status");
  },

  refresh(): Promise<RtmpStatusResponse> {
    return postJson<RtmpStatusResponse>("/api/satnav/system/rtmp/refresh", {});
  },
};

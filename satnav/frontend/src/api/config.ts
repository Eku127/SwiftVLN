export const apiBaseUrl =
  import.meta.env.VITE_API_BASE?.trim() || "http://127.0.0.1:8000";

export const defaultPollIntervals = {
  statusMs: 4000,
  /** RTMP 原始帧预览 GET /api/satnav/media/raw_img 轮询间隔 */
  rawImageMs: 1500,
  /** OSD 快照 GET .../flight/osd/latest 轮询间隔 （尽量与rawImageMs一致）*/
  osdMs: 1500,
  modelLogsMs: 2500,
  stickTaskMs: 800,
} as const;

export type NavAction = 0 | 1 | 2 | 3;

export const NAV_ACTION_LABEL: Record<NavAction, string> = {
  0: "STOP",
  1: "前进",
  2: "左转",
  3: "右转",
};

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

export interface ModelStatusResponse {
  status: "ready" | "starting" | "error";
  api: { ready: boolean };
  rtmp: Record<string, unknown>;
  model: Record<string, unknown>;
  timestamp: string;
}

export interface ModelLogEntry {
  sequence: number;
  timestamp: string;
  source: string;
  stream: string;
  level: string;
  message: string;
}

export interface ModelLogsResponse {
  logs: ModelLogEntry[];
  latest_sequence: number;
  has_more: boolean;
}

export interface RtmpStatusResponse {
  rtmp_url: string;
  connected: boolean;
  stream_active: boolean;
  probe_method: string;
  checked_at: string;
  video?: { width?: number; height?: number; codec?: string };
  error?: string;
  refreshed?: boolean;
}

export interface FlightBackendStatusResponse {
  backend_host?: string;
  backend_port?: number;
  healthy: boolean;
  checked_at: string;
  error?: string;
}

export interface LoginRequest {
  username?: string;
  password?: string;
  flag?: number;
}

export interface LoginResponse {
  auth_method: string;
  logged_in: boolean;
  workspace_id?: string | null;
  backend?: Record<string, unknown>;
  timestamp: string;
}

export interface RegisterDeviceRequest {
  rc_sn: string;
  device_sn: string;
}

export interface DeviceInfoResponse {
  logged_in?: boolean;
  workspace_id?: string | null;
  client_id?: string | null;
  drc_ready?: boolean;
  registered?: boolean;
  rc_sn?: string | null;
  device_sn?: string | null;
  registered_at?: string | null;
  timestamp: string;
}

export interface OsdLatestResponse {
  device_sn?: string | null;
  attitude_head?: number | null;
  latitude?: number | null;
  longitude?: number | null;
  height?: number | null;
  speed_x?: number | null;
  speed_y?: number | null;
  speed_z?: number | null;
  gimbal_pitch?: number | null;
  gimbal_roll?: number | null;
  gimbal_yaw?: number | null;
  received_at_ms?: number | null;
  age_ms?: number | null;
  backend?: Record<string, unknown>;
  timestamp: string;
}

export interface DrcRequest {
  expire_sec?: number;
}

export interface DrcResponse {
  drc_ready: boolean;
  client_id: string;
  workspace_id: string;
  rc_sn: string;
  device_sn: string;
  expire_sec: number;
  timestamp: string;
}

export interface FlightForwardRequest {
  distance_m?: number;
  tolerance_m?: number;
  timeout_ms?: number;
}

export interface FlightTurnRequest {
  action: 2 | 3;
  degree?: number;
  tolerance_deg?: number;
  timeout_ms?: number;
}

export interface FlightActionResponse {
  action: NavAction;
  task_id: string;
  status: string;
  parameters: Record<string, unknown>;
  backend: Record<string, unknown>;
  timestamp: string;
}

export interface StickTaskResponse {
  task_id: string;
  kind?: string | null;
  status: string;
  backend: Record<string, unknown>;
  timestamp: string;
}

export interface InferenceRequest {
  instruction: string;
}

export interface InferenceTiming {
  capture_ms: number;
  preprocess_ms: number;
  input_save_ms: number;
  deploy_response_ms: number;
  capture_to_action_response_ms: number;
}

export interface InferenceResponse {
  instruction: string;
  session_id: string;
  started_new_session: boolean;
  performed_inference: boolean;
  raw_action_text: string;
  actions: NavAction[];
  next_action: NavAction | null;
  remaining_actions: NavAction[];
  completed_action: NavAction | null;
  deploy_state: string;
  frame_sequence: number;
  image_path: string;
  timing: InferenceTiming;
  timestamp: string;
}

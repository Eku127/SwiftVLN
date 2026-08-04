import { getJson, postJson } from "../http";
import type {
  DeviceInfoResponse,
  DrcRequest,
  DrcResponse,
  FlightActionResponse,
  FlightBackendStatusResponse,
  FlightForwardRequest,
  FlightTurnRequest,
  LoginRequest,
  LoginResponse,
  OsdLatestResponse,
  RegisterDeviceRequest,
  StickTaskResponse,
} from "../types";

export const flightClient = {
  getBackendStatus(): Promise<FlightBackendStatusResponse> {
    return getJson<FlightBackendStatusResponse>(
      "/api/satnav/system/flight-rc-backend/status",
    );
  },

  login(body: LoginRequest = {}): Promise<LoginResponse> {
    return postJson<LoginResponse>(
      "/api/satnav/system/flight-rc-backend/login",
      body,
    );
  },

  registerDevice(body: RegisterDeviceRequest): Promise<DeviceInfoResponse> {
    return postJson<DeviceInfoResponse>(
      "/api/satnav/system/flight-rc-backend/register_device",
      body,
    );
  },

  getDeviceInfo(): Promise<DeviceInfoResponse> {
    return getJson<DeviceInfoResponse>(
      "/api/satnav/system/flight-rc-backend/cur_device_info",
    );
  },

  acquireDrc(body: DrcRequest = {}): Promise<DrcResponse> {
    return postJson<DrcResponse>(
      "/api/satnav/system/flight-rc-backend/drc",
      body,
    );
  },

  forward(body: FlightForwardRequest = {}): Promise<FlightActionResponse> {
    return postJson<FlightActionResponse>(
      "/api/satnav/system/flight-rc-backend/flight/forward",
      body,
    );
  },

  turn(body: FlightTurnRequest): Promise<FlightActionResponse> {
    return postJson<FlightActionResponse>(
      "/api/satnav/system/flight-rc-backend/flight/turn",
      body,
    );
  },

  getStickTask(taskId: string): Promise<StickTaskResponse> {
    return getJson<StickTaskResponse>(
      `/api/satnav/system/flight-rc-backend/flight/stick-task/${encodeURIComponent(taskId)}`,
    );
  },

  getOsdLatest(): Promise<OsdLatestResponse> {
    return getJson<OsdLatestResponse>(
      "/api/satnav/system/flight-rc-backend/flight/osd/latest",
    );
  },
};

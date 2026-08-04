import { flightClient } from "../clients/flight";
import type {
  DrcRequest,
  LoginRequest,
  RegisterDeviceRequest,
} from "../types";

export interface FlightSetupInput {
  login?: LoginRequest;
  device: RegisterDeviceRequest;
  drc?: DrcRequest;
}

export const flightSetupWorkflow = {
  async run(input: FlightSetupInput) {
    const login = await flightClient.login(input.login ?? {});
    const device = await flightClient.registerDevice(input.device);
    const drc = await flightClient.acquireDrc(input.drc ?? { expire_sec: 3600 });
    return { login, device, drc };
  },
};

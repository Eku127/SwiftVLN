/** SatNav API SDK — call-side only (no React). */

export { apiBaseUrl, defaultPollIntervals } from "./config";
export { ApiError } from "./errors";
export * from "./types";

export { healthClient } from "./clients/health";
export { modelClient } from "./clients/model";
export { rtmpClient } from "./clients/rtmp";
export { flightClient } from "./clients/flight";
export { inferenceClient } from "./clients/inference";
export { mediaClient } from "./clients/media";

export { flightSetupWorkflow } from "./workflows/flightSetup";
export { pollStickTask } from "./workflows/stickPoll";
export {
  executeFlightAction,
  runInferenceStep,
  runNavStep,
} from "./workflows/navStep";

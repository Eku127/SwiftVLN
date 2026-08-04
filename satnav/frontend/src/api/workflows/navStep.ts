import { flightClient } from "../clients/flight";
import { inferenceClient } from "../clients/inference";
import type {
  FlightActionResponse,
  InferenceResponse,
  NavAction,
  StickTaskResponse,
} from "../types";
import { pollStickTask } from "./stickPoll";

export async function runInferenceStep(
  instruction: string,
): Promise<InferenceResponse> {
  return inferenceClient.run({ instruction: instruction.trim() });
}

export async function executeFlightAction(
  action: NavAction,
): Promise<FlightActionResponse | null> {
  if (action === 0) {
    return null;
  }
  if (action === 1) {
    return flightClient.forward({});
  }
  if (action === 2) {
    return flightClient.turn({ action: 2 });
  }
  if (action === 3) {
    return flightClient.turn({ action: 3 });
  }
  throw new Error(`unsupported nav action: ${action}`);
}

export interface NavStepResult {
  inference: InferenceResponse;
  flight: FlightActionResponse | null;
  stickTask: StickTaskResponse | null;
}

export interface NavStepOptions {
  instruction: string;
  onStickUpdate?: (response: StickTaskResponse) => void;
}

/** One manual loop step: inference → flight (if needed) → poll stick-task. */
export async function runNavStep(
  options: NavStepOptions,
): Promise<NavStepResult> {
  const inference = await runInferenceStep(options.instruction);
  const nextAction = inference.next_action;

  if (nextAction === null || nextAction === 0) {
    return { inference, flight: null, stickTask: null };
  }

  const flight = await executeFlightAction(nextAction);
  if (!flight?.task_id) {
    return { inference, flight, stickTask: null };
  }

  const stickTask = await pollStickTask(flight.task_id, {
    onUpdate: options.onStickUpdate,
  });
  return { inference, flight, stickTask };
}

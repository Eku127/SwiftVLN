import { postJson } from "../http";
import type { InferenceRequest, InferenceResponse } from "../types";

export const inferenceClient = {
  run(body: InferenceRequest): Promise<InferenceResponse> {
    return postJson<InferenceResponse>("/api/satnav/model/inference", body);
  },
};

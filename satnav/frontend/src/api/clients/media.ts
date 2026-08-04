import { getBlob, mediaUrl } from "../http";

const RAW_PATH = "/api/satnav/media/raw_img";
const MODEL_INPUT_PATH = "/api/satnav/media/model_input_img";

export const mediaClient = {
  rawImageUrl(cacheBust?: number): string {
    return mediaUrl(RAW_PATH, cacheBust);
  },

  modelInputImageUrl(cacheBust?: number): string {
    return mediaUrl(MODEL_INPUT_PATH, cacheBust);
  },

  async fetchRawImage(cacheBust?: number): Promise<Blob> {
    const query = cacheBust ? `?t=${cacheBust}` : "";
    return getBlob(`${RAW_PATH}${query}`);
  },

  async fetchModelInputImage(cacheBust?: number): Promise<Blob> {
    const query = cacheBust ? `?t=${cacheBust}` : "";
    return getBlob(`${MODEL_INPUT_PATH}${query}`);
  },
};

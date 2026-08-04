#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash push_video_to_rtmp.sh <rtmp_url> <video_path>

Example:
  bash push_video_to_rtmp.sh \
    rtmp://127.0.0.1/live/satnav \
    /path/to/video.webm

Environment variables:
  LOOP=true             Loop the input video indefinitely (true/false).
  OUTPUT_SIZE=1920:1080 Output resolution. Set to "source" to keep the source size.
  VIDEO_ENCODER=libx264 FFmpeg H.264 encoder (for example libx264 or h264_nvenc).
  VIDEO_BITRATE=6M      Target video bitrate.
  GOP_SIZE=25           Keyframe interval in frames.
  FFMPEG_LOGLEVEL=info  FFmpeg log level.
EOF
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

RTMP_URL="$1"
VIDEO_PATH="$2"
LOOP="${LOOP:-true}"
OUTPUT_SIZE="${OUTPUT_SIZE:-1920:1080}"
VIDEO_ENCODER="${VIDEO_ENCODER:-libx264}"
VIDEO_BITRATE="${VIDEO_BITRATE:-6M}"
GOP_SIZE="${GOP_SIZE:-25}"
FFMPEG_LOGLEVEL="${FFMPEG_LOGLEVEL:-info}"

if [[ "$RTMP_URL" != rtmp://* && "$RTMP_URL" != rtmps://* ]]; then
  echo "RTMP URL must start with rtmp:// or rtmps://: $RTMP_URL" >&2
  exit 2
fi

for command_name in ffmpeg ffprobe; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is not installed: $command_name" >&2
    exit 1
  fi
done

if [ ! -s "$VIDEO_PATH" ]; then
  echo "Input video does not exist or is empty: $VIDEO_PATH" >&2
  exit 1
fi

if ! ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 "$VIDEO_PATH" >/dev/null; then
  echo "Input file does not contain a readable video stream: $VIDEO_PATH" >&2
  exit 1
fi

if ! ffmpeg -hide_banner -encoders 2>/dev/null |
  awk '{print $2}' |
  grep -Fxq "$VIDEO_ENCODER"; then
  echo "FFmpeg encoder is unavailable: $VIDEO_ENCODER" >&2
  exit 1
fi

INPUT_ARGS=(-re)
case "$LOOP" in
  true|1|yes)
    INPUT_ARGS+=(-stream_loop -1)
    ;;
  false|0|no)
    ;;
  *)
    echo "LOOP must be true or false, got: $LOOP" >&2
    exit 2
    ;;
esac

VIDEO_FILTER_ARGS=()
if [ "$OUTPUT_SIZE" != "source" ]; then
  if [[ ! "$OUTPUT_SIZE" =~ ^[0-9]+:[0-9]+$ ]]; then
    echo "OUTPUT_SIZE must be WIDTH:HEIGHT or source, got: $OUTPUT_SIZE" >&2
    exit 2
  fi
  VIDEO_FILTER_ARGS=(-vf "scale=${OUTPUT_SIZE}:force_original_aspect_ratio=decrease,pad=${OUTPUT_SIZE}:(ow-iw)/2:(oh-ih)/2")
fi

ENCODER_ARGS=()
case "$VIDEO_ENCODER" in
  libx264)
    ENCODER_ARGS=(-preset ultrafast -tune zerolatency)
    ;;
  h264_nvenc)
    ENCODER_ARGS=(-preset p1 -tune ll -rc cbr)
    ;;
esac

VIDEO_INFO="$(
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate \
    -of csv=p=0 "$VIDEO_PATH"
)"

echo "Starting real-time RTMP test stream"
echo "  input:        $VIDEO_PATH"
echo "  source:       $VIDEO_INFO"
echo "  output size:  $OUTPUT_SIZE"
echo "  encoder:      $VIDEO_ENCODER"
echo "  bitrate:      $VIDEO_BITRATE"
echo "  loop:         $LOOP"
echo "  destination:  $RTMP_URL"
echo "Press Ctrl+C to stop."

exec ffmpeg \
  -hide_banner \
  -loglevel "$FFMPEG_LOGLEVEL" \
  "${INPUT_ARGS[@]}" \
  -i "$VIDEO_PATH" \
  -map 0:v:0 \
  -an \
  "${VIDEO_FILTER_ARGS[@]}" \
  -c:v "$VIDEO_ENCODER" \
  "${ENCODER_ARGS[@]}" \
  -pix_fmt yuv420p \
  -b:v "$VIDEO_BITRATE" \
  -maxrate "$VIDEO_BITRATE" \
  -bufsize "$VIDEO_BITRATE" \
  -g "$GOP_SIZE" \
  -keyint_min "$GOP_SIZE" \
  -bf 0 \
  -flvflags no_duration_filesize \
  -f flv \
  "$RTMP_URL"

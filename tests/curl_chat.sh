#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:7891/v1/chat/completions}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PATH="${SCRIPT_DIR}/../assets/demo/images/episode0/cam_high/0.jpg"

encode_image() {
  base64 <"$1" | tr -d '\n'
}

IMAGE="$(encode_image "${IMAGE_PATH}")"

curl -sS -X POST "${BASE_URL}" \
  -H 'Content-Type: application/json' \
  --data @- <<EOF
{
  "model": "dm05",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the scene in this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE}"}}
      ]
    }
  ],
  "max_tokens": 128,
  "temperature": 0
}
EOF
echo

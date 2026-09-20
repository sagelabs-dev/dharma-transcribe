#!/usr/bin/env bash
# Fetch WhisperX alignment models with validated resume (flaky-network hardened).
set -uo pipefail
CKPT="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CKPT"
declare -A MODELS=(
  ["wav2vec2_fairseq_base_ls960_asr_ls960.pth"]="https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
)
for f in "${!MODELS[@]}"; do
  URL="${MODELS[$f]}"; OUT="$CKPT/$f"
  EXPECTED=$(curl -sIL "$URL" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')
  EXPECTED=${EXPECTED:-0}
  echo "=== $f expected=$EXPECTED ==="
  for attempt in $(seq 1 40); do
    HAVE=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
    if [ "$HAVE" = "$EXPECTED" ] && [ "$EXPECTED" != "0" ]; then
      echo "COMPLETE after $attempt checks: $HAVE bytes"; break
    fi
    echo "attempt $attempt: have=$HAVE / $EXPECTED — resuming..."
    curl -L -C - --retry 5 --retry-delay 2 --retry-all-errors --max-time 900 \
         --connect-timeout 20 -o "$OUT" "$URL" 2>&1 | tail -1
  done
  FINAL=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$FINAL" = "$EXPECTED" ] && [ "$EXPECTED" != "0" ]; then
    echo "OK $f size=$FINAL"
  else
    echo "FAIL $f size=$FINAL expected=$EXPECTED"
  fi
  ls -la "$OUT"
done
echo "ALIGN_FETCH_DONE"

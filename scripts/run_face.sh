#!/usr/bin/env bash
set -e
INPUT="$1"; OUTPUT="$2"; MODE="${3:-opt}"
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: ./scripts/run_face.sh <input_dir> <output_dir> [strict|opt]"; exit 1
fi
if [ "$MODE" = "strict" ]; then
  python -m face.cli --strict --input "$INPUT" --output "$OUTPUT"
else
  python -m face.cli --input "$INPUT" --output "$OUTPUT" --save-debug
fi

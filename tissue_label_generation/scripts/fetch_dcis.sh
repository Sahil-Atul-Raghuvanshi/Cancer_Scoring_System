#!/usr/bin/env bash
# Fetch the 790 BRACS DCIS ROI PNGs into benchmarks/ (NON-COMMERCIAL asset -
# research/benchmark use only; nothing under backend/ may ever import this).
# Resumable: re-run it and it skips files already present at the right size.
set -uo pipefail

FTP="ftp://utente_reader:ICAR-CNR@histoimage.na.icar.cnr.it/BRACS_RoI/latest_version"
# The BRACS regions are a source download, so they land in the workspace's
# read-only source tree - `data/original/bracs/<lesion>/` - and not beside this
# script. This file lives in scripts/ because it is code; what it fetches is data.
BRACS_ROOT="$(dirname "$0")/../../storage/data/original/bracs"
mkdir -p "$BRACS_ROOT"
BRACS_ROOT="$(cd "$BRACS_ROOT" && pwd)"
DEST="$BRACS_ROOT/dcis"
LOG="$DEST/download.log"

mkdir -p "$DEST"
echo "=== start $(date -Is) ===" >> "$LOG"

total_ok=0
total_fail=0

for s in train val test; do
  mkdir -p "$DEST/$s"
  man="$DEST/$s/.manifest"

  # name + size, so a re-run can tell complete files from partial ones
  curl -s --connect-timeout 30 --max-time 300 "$FTP/$s/5_DCIS/" \
    | awk '/\.png$/{print $9, $5}' > "$man"

  n=$(wc -l < "$man")
  echo "[$s] manifest: $n files" >> "$LOG"

  while read -r name size; do
    [ -z "${name:-}" ] && continue
    out="$DEST/$s/$name"

    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo -1)" = "$size" ]; then
      continue
    fi

    if curl -s --connect-timeout 30 --max-time 900 -C - \
         -o "$out" "$FTP/$s/5_DCIS/$name"; then
      total_ok=$((total_ok + 1))
    else
      # a complete file makes -C - return 416; retry once from scratch
      if curl -s --connect-timeout 30 --max-time 900 \
           -o "$out" "$FTP/$s/5_DCIS/$name"; then
        total_ok=$((total_ok + 1))
      else
        echo "FAIL $s/$name" >> "$LOG"
        total_fail=$((total_fail + 1))
      fi
    fi
  done < "$man"

  have=$(find "$DEST/$s" -name '*.png' | wc -l)
  echo "[$s] on disk: $have png" >> "$LOG"
done

echo "=== done $(date -Is) : ok=$total_ok fail=$total_fail ===" >> "$LOG"
find "$DEST" -name '*.png' | wc -l | xargs -I{} echo "TOTAL PNG: {}" >> "$LOG"
du -sh "$DEST" >> "$LOG" 2>/dev/null

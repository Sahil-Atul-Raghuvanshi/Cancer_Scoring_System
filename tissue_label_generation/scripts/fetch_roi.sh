#!/usr/bin/env bash
# Fetch one BRACS lesion type's ROI PNGs into this folder (NON-COMMERCIAL asset -
# research/benchmark use only; nothing under backend/ may ever import this).
#
#   ./fetch_roi.sh 6_IC ic 2      # <ftp dir> <local dir> [regions per patient]
#   ./fetch_roi.sh 6_IC ic        # every IC region: 649 files, 15.4 GiB
#
# **Why the per-patient limit is the default way to run this.** The whole 6_IC tree is
# 649 regions and 15.4 GiB, and the DCIS run that produced approach 1's borrowed class
# used 153 regions to keep 120 - so downloading everything would spend 10 GiB and an
# extra day of teacher time on regions that were never going to be segmented. The limit
# is applied HERE, against the FTP listing, rather than after the download, because the
# download is the expensive half.
#
# Two regions per patient, largest first, matches what `export_to_approach1.py
# --per-case 2` will then select locally, so the disk holds what the pipeline wants and
# not a superset. Sampling per patient also stops the set becoming a study of the few
# patients with the most regions - 42 of BRACS's 665 DCIS training regions are case
# 1247's alone.
#
# Size is the proxy for area, because the FTP listing gives bytes and not dimensions.
# For PNGs of one tissue type at one resolution it is a good one, and it only has to
# rank regions within a patient.
#
# Resumable: re-run it and it skips files already present at the right size.
set -uo pipefail

FTP_DIR="${1:?usage: fetch_roi.sh <ftp dir, e.g. 6_IC> <local dir, e.g. ic> [per-case]}"
LOCAL="${2:?usage: fetch_roi.sh <ftp dir> <local dir> [per-case]}"
PER_CASE="${3:-}"

FTP="ftp://utente_reader:ICAR-CNR@histoimage.na.icar.cnr.it/BRACS_RoI/latest_version"
# The BRACS regions are a source download, so they land in the workspace's
# read-only source tree - `data/original/bracs/<lesion>/` - and not beside this
# script. This file lives in scripts/ because it is code; what it fetches is data.
BRACS_ROOT="$(dirname "$0")/../../storage/data/original/bracs"
mkdir -p "$BRACS_ROOT"
BRACS_ROOT="$(cd "$BRACS_ROOT" && pwd)"
DEST="$BRACS_ROOT/$LOCAL"
LOG="$DEST/download.log"

mkdir -p "$DEST"
echo "=== start $(date -Is)  $FTP_DIR -> $LOCAL  per-case=${PER_CASE:-all} ===" >> "$LOG"

total_ok=0
total_fail=0

for s in train val test; do
  mkdir -p "$DEST/$s"
  man="$DEST/$s/.manifest"
  full="$DEST/$s/.manifest_full"

  # name + size, so a re-run can tell complete files from partial ones
  curl -s --connect-timeout 30 --max-time 300 "$FTP/$s/$FTP_DIR/" \
    | awk '/\.png$/{print $9, $5}' > "$full"

  if [ -z "$PER_CASE" ]; then
    cp "$full" "$man"
  else
    # `BRACS_1247_IC_3.png` -> patient `BRACS_1247`. Keep the PER_CASE largest per
    # patient. Sorting by patient then by descending size makes the choice a head.
    awk '{split($1, f, "_"); print f[1]"_"f[2], $1, $2}' "$full" \
      | sort -k1,1 -k3,3nr \
      | awk -v k="$PER_CASE" '{ if ($1 != last) { n = 0; last = $1 }
                                if (++n <= k) print $2, $3 }' > "$man"
  fi

  n=$(wc -l < "$man")
  echo "[$s] manifest: $n of $(wc -l < "$full") files" >> "$LOG"

  while read -r name size; do
    [ -z "${name:-}" ] && continue
    out="$DEST/$s/$name"

    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo -1)" = "$size" ]; then
      continue
    fi

    if curl -s --connect-timeout 30 --max-time 900 -C - \
         -o "$out" "$FTP/$s/$FTP_DIR/$name"; then
      total_ok=$((total_ok + 1))
    else
      # a complete file makes -C - return 416; retry once from scratch
      if curl -s --connect-timeout 30 --max-time 900 \
           -o "$out" "$FTP/$s/$FTP_DIR/$name"; then
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

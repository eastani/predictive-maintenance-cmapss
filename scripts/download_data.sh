#!/usr/bin/env bash
#
# download_data.sh — Fetch the NASA CMAPSS turbofan dataset into data/raw/.
#
# CMAPSS is hosted by NASA's Prognostics Center of Excellence. Direct URLs
# have changed over the years, so this script tries a list of known mirrors
# in turn and falls back to clear manual instructions if all of them fail.
#
# After a successful run, data/raw/ should contain at least:
#   train_FD001.txt  test_FD001.txt  RUL_FD001.txt
#   train_FD002.txt  test_FD002.txt  RUL_FD002.txt
#   train_FD003.txt  test_FD003.txt  RUL_FD003.txt
#   train_FD004.txt  test_FD004.txt  RUL_FD004.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/raw"
mkdir -p "${DEST}"

CANDIDATES=(
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
    "https://github.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/raw/main/CMAPSSData.zip"
)

ARCHIVE="${DEST}/CMAPSSData.zip"

if compgen -G "${DEST}/train_FD00*.txt" > /dev/null; then
    echo "✓ CMAPSS files already present in ${DEST}; nothing to do."
    exit 0
fi

for url in "${CANDIDATES[@]}"; do
    echo "→ Trying: ${url}"
    if curl -fsSL --retry 2 --max-time 60 -o "${ARCHIVE}" "${url}"; then
        echo "  Download OK."
        break
    fi
    echo "  Failed, trying next mirror…"
done

if [[ ! -s "${ARCHIVE}" ]]; then
    cat <<'MSG'
✗ Automatic download failed.

Manual fallback:
  1. Visit NASA's Prognostics Data Repository:
       https://www.nasa.gov/intelligent-systems-division/
  2. Download the "Turbofan Engine Degradation Simulation" data.
  3. Unzip the archive and copy all train_*.txt, test_*.txt and
     RUL_*.txt files into:
       data/raw/

Then re-run this script (or simply re-run the test suite).
MSG
    exit 1
fi

echo "→ Extracting…"
unzip -o "${ARCHIVE}" -d "${DEST}" > /dev/null

# Some archives nest the txt files inside a CMAPSSData/ folder; flatten if so.
if [[ -d "${DEST}/CMAPSSData" ]]; then
    mv "${DEST}/CMAPSSData/"*.txt "${DEST}/" 2>/dev/null || true
    rmdir "${DEST}/CMAPSSData" 2>/dev/null || true
fi

rm -f "${ARCHIVE}"

echo "✓ CMAPSS files extracted to ${DEST}:"
ls -1 "${DEST}"/train_FD00*.txt 2>/dev/null | sed 's/^/    /' || true

#!/usr/bin/env bash
#
# download_data.sh — Fetch the NASA CMAPSS turbofan dataset into data/raw/.
#
# CMAPSS is hosted by NASA's Prognostics Center of Excellence. Direct URLs
# at NASA have changed over the years and the official zip is occasionally
# unreachable, so this script first tries individual files from a known
# good GitHub mirror and falls back to clear manual instructions.
#
# After a successful run, data/raw/ should contain:
#   train_FD001.txt  test_FD001.txt  RUL_FD001.txt
#   train_FD002.txt  test_FD002.txt  RUL_FD002.txt
#   train_FD003.txt  test_FD003.txt  RUL_FD003.txt
#   train_FD004.txt  test_FD004.txt  RUL_FD004.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/raw"
mkdir -p "${DEST}"

if compgen -G "${DEST}/train_FD00*.txt" > /dev/null \
   && compgen -G "${DEST}/test_FD00*.txt" > /dev/null \
   && compgen -G "${DEST}/RUL_FD00*.txt" > /dev/null; then
    echo "✓ CMAPSS files already present in ${DEST}; nothing to do."
    exit 0
fi

# A community mirror that hosts each file individually under data/.
MIRROR_BASE="https://raw.githubusercontent.com/egehanyorulmaz/nasa-turbofan-engine-rul-prediction/main/data"

ALL_FILES=(
    train_FD001 train_FD002 train_FD003 train_FD004
    test_FD001  test_FD002  test_FD003  test_FD004
    RUL_FD001   RUL_FD002   RUL_FD003   RUL_FD004
)

failed=0
for stem in "${ALL_FILES[@]}"; do
    target="${DEST}/${stem}.txt"
    if [[ -s "${target}" ]]; then
        echo "  ✓ ${stem}.txt already present"
        continue
    fi
    if curl -fsSL --retry 2 --max-time 60 -o "${target}" "${MIRROR_BASE}/${stem}.txt"; then
        echo "  ✓ ${stem}.txt"
    else
        echo "  ✗ ${stem}.txt"
        rm -f "${target}"
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    cat <<'MSG'

✗ One or more files failed to download.

Manual fallback:
  1. Visit NASA's Prognostics Data Repository:
       https://www.nasa.gov/intelligent-systems-division/
     or any community mirror that hosts the original "Turbofan Engine
     Degradation Simulation Data Set" zip.
  2. Unzip the archive and copy all train_*.txt, test_*.txt and
     RUL_*.txt files into:
       data/raw/

Then re-run this script (or the test suite).
MSG
    exit 1
fi

echo
echo "✓ CMAPSS files in ${DEST}:"
ls -1 "${DEST}"/*.txt | sed 's/^/    /'

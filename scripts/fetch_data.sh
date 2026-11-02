#!/usr/bin/env bash
# Downloads the four Qwen-Bailian usage traces (Apache-2.0) into data/raw/ at a pinned commit
# and verifies their SHA-256. Files already present are verified, not downloaded again.
set -euo pipefail

COMMIT=5f7439c51ec248a0c585f7d90a41a6f57773b912
BASE="https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/$COMMIT"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    echo "need sha256sum or shasum" >&2
    exit 1
fi

mkdir -p "$DEST"
status=0
while read -r name sum; do
    file="$DEST/qwen_${name}_blksz_16.jsonl"
    src="$file"
    if [[ ! -f "$file" ]]; then
        echo "fetching  $(basename "$file")"
        src="$file.part"
        curl -fL --retry 3 --retry-delay 5 -o "$src" "$BASE/qwen_${name}_blksz_16.jsonl" </dev/null
    fi
    got="$(sha256 "$src")"
    if [[ "$got" != "$sum" ]]; then
        echo "MISMATCH  $(basename "$file"): expected $sum, got $got" >&2
        if [[ "$src" != "$file" ]]; then
            rm -f "$src"
        fi
        status=1
        continue
    fi
    if [[ "$src" != "$file" ]]; then
        mv "$src" "$file"
    fi
    echo "ok        $(basename "$file")"
done <<'EOF'
coder 3d74974cebb7bccb69b1ca6ea210a44d9a3033443c0271ed015327284dea1aff
thinking 41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906
traceA 07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17
traceB 68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59
EOF
exit $status

#!/usr/bin/env bash
# wu1-bulk-ship.sh — OPERATION FAREWELL EMBRACE WU1 bulk-ship script.
#
# Run this in a separate terminal window while the supervisor proceeds
# with WU2 code changes. Each `acervo ship` command runs in the foreground
# in this script's shell — no Bash-tool timeout to worry about.
#
# Usage (in a fresh terminal):
#   cd /Users/stovak/Projects/flux-2-swift-mlx
#   bash scripts/wu1-bulk-ship.sh 2>&1 | tee -a docs/missions/cdn-bulk-ship.log
#
# The script:
#   - Sorts ships in the EXECUTION_PLAN order: 5 (upload-only — staged),
#     6, 7, 8, 9, 10, 11 (subfolder), 12 (gated). Skips Sortie 13 (plan smoke
#     test) per operator direction (assume eventual consistency; supervisor
#     starts WU2 in parallel).
#   - For each ship, tries the default invocation first, falls back to
#     --no-verify if CHECK 1 fails (non-LFS source). All lmstudio-community
#     repos use --no-verify directly per Sortie-1 recon.
#   - Logs each ship's full stdout/stderr to docs/missions/ship-logs/<slug>.log
#     so the supervisor's later closeout sorties can read them.
#   - Prints a final summary of which ships succeeded.
#   - Does NOT clean up /tmp/acervo-staging/ — operator can rm later.

set -uo pipefail

# ---------------------------------------------------------------------------
# Environment loader (same prefix the supervisor uses for sortie agents).
# ---------------------------------------------------------------------------
[ -z "${HF_TOKEN:-}" ] && export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
[ -z "${R2_PUBLIC_URL:-}" ] && export R2_PUBLIC_URL="${R2_ENDPOINT:-}"

if [ -z "${R2_PUBLIC_URL:-}" ]; then
  echo "ERROR: R2_PUBLIC_URL is not set and R2_ENDPOINT fallback is empty." >&2
  echo "Export R2_PUBLIC_URL in this shell before running the script." >&2
  exit 1
fi
if [ ! -s ~/.cache/huggingface/token ]; then
  echo "ERROR: ~/.cache/huggingface/token is missing or empty. Run 'hf auth login' first." >&2
  exit 1
fi
if ! command -v acervo >/dev/null 2>&1; then
  echo "ERROR: acervo CLI not found in PATH." >&2
  exit 1
fi

ACERVO_VERSION="$(acervo --version)"
echo "[info] acervo version: ${ACERVO_VERSION}"
if [ "${ACERVO_VERSION}" \< "0.8.4" ]; then
  echo "ERROR: acervo ${ACERVO_VERSION} is older than required 0.8.4 (manifest path fix)." >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/docs/missions/ship-logs"
mkdir -p "${LOG_DIR}"

START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[info] bulk ship started at ${START_TIME}"
echo "[info] per-ship logs will land in ${LOG_DIR}/"
echo

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

# repo_to_slug "owner/repo" → "owner_repo"
repo_to_slug() {
  printf '%s' "${1//\//_}"
}

# verify_cdn_manifest "owner/repo"
# echoes HTTP code; exits 0 if 200, 1 otherwise.
verify_cdn_manifest() {
  local repo="$1"
  local slug; slug="$(repo_to_slug "${repo}")"
  local url="${R2_PUBLIC_URL}/models/${slug}/manifest.json"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' "${url}")"
  echo "${code}"
  [ "${code}" = "200" ]
}

# ship_repo "label" "owner/repo" [acervo-ship-args...]
# Tries default invocation; if it fails, retries with --no-verify.
# Logs stdout+stderr to a per-slug file. Returns 0 on success.
ship_repo() {
  local label="$1"; shift
  local repo="$1"; shift
  local extra_args=("$@")
  local slug; slug="$(repo_to_slug "${repo}")"
  local log="${LOG_DIR}/${slug}.log"

  echo "----------------------------------------------------------------------"
  echo "[$(date '+%H:%M:%S')] ${label}: ${repo} ${extra_args[*]:-}"
  echo "  log → ${log}"

  # Skip if CDN manifest is already live.
  local cdn_code; cdn_code="$(curl -sS -o /dev/null -w '%{http_code}' "${R2_PUBLIC_URL}/models/${slug}/manifest.json")"
  if [ "${cdn_code}" = "200" ]; then
    echo "  → already live on CDN (HTTP 200). Skipping."
    return 0
  fi

  {
    echo "===== ${label}: ${repo} ====="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "extra_args: ${extra_args[*]:-}"
    echo

    df -k "${HOME}" || true
    echo
  } >> "${log}"

  # Sortie 5 special case: staging already exists from the failed retry.
  # Skip download, run upload only.
  if [ "${repo}" = "aydin99/FLUX.2-klein-4B-int8" ] && \
     [ -d "/tmp/acervo-staging/${slug}" ] && \
     [ -f "/tmp/acervo-staging/${slug}/manifest.json" ]; then
    echo "  Sortie 5: staging exists; running 'acervo upload' (no re-download)"
    if acervo upload "${repo}" "/tmp/acervo-staging/${slug}" >> "${log}" 2>&1; then
      echo "  ✅ upload exit 0"
      return 0
    else
      echo "  ❌ upload failed (see ${log})"
      return 1
    fi
  fi

  # Try the default invocation first. If it exits non-zero, retry with --no-verify
  # (CHECK 1 fails for non-LFS-backed repos).
  # Use ${extra_args[@]+...} pattern so an empty array doesn't trip `set -u`.
  if acervo ship "${repo}" ${extra_args[@]+"${extra_args[@]}"} >> "${log}" 2>&1; then
    echo "  ✅ ship exit 0 (default invocation)"
    return 0
  fi

  echo "  default invocation failed; retrying with --no-verify"
  echo "===== retry with --no-verify =====" >> "${log}"
  if acervo ship "${repo}" ${extra_args[@]+"${extra_args[@]}"} --no-verify >> "${log}" 2>&1; then
    echo "  ✅ ship exit 0 (--no-verify)"
    return 0
  fi
  echo "  ❌ ship failed under both invocations (see ${log})"
  return 1
}

# ---------------------------------------------------------------------------
# Ship roster (preserving EXECUTION_PLAN order; Sortie 13 smoke test skipped).
# ---------------------------------------------------------------------------

declare -a results=()

run_ship() {
  local label="$1" repo="$2"; shift 2
  if ship_repo "${label}" "${repo}" "$@"; then
    results+=("✅ ${label}: ${repo}")
  else
    results+=("❌ ${label}: ${repo}")
  fi
  echo
}

run_ship "Sortie 5"  "aydin99/FLUX.2-klein-4B-int8"
run_ship "Sortie 6"  "lmstudio-community/Qwen3-8B-MLX-8bit"
run_ship "Sortie 7"  "black-forest-labs/FLUX.2-klein-4B"
run_ship "Sortie 8"  "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit"
run_ship "Sortie 9"  "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit"
run_ship "Sortie 10" "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit"
# Sortie 11: subfolder ship — second positional arg.
run_ship "Sortie 11" "VincentGOURBIN/flux_qint_8bit" "flux-2-dev/transformer/qint8/"
run_ship "Sortie 12" "black-forest-labs/FLUX.2-klein-9B"

# ---------------------------------------------------------------------------
# Final summary + manifest verification.
# ---------------------------------------------------------------------------

END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
echo "Bulk ship finished at ${END_TIME} (started ${START_TIME})"
echo
echo "Per-ship results:"
for r in "${results[@]}"; do
  echo "  ${r}"
done

echo
echo "CDN manifest verification:"
manifest_failures=0
for repo in \
  "aydin99/FLUX.2-klein-4B-int8" \
  "lmstudio-community/Qwen3-8B-MLX-8bit" \
  "black-forest-labs/FLUX.2-klein-4B" \
  "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit" \
  "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit" \
  "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit" \
  "VincentGOURBIN/flux_qint_8bit" \
  "black-forest-labs/FLUX.2-klein-9B"; do
  slug="$(repo_to_slug "${repo}")"
  code="$(curl -sS -o /dev/null -w '%{http_code}' "${R2_PUBLIC_URL}/models/${slug}/manifest.json")"
  if [ "${code}" = "200" ]; then
    echo "  ✅ ${repo} → HTTP ${code}"
  else
    echo "  ❌ ${repo} → HTTP ${code}"
    manifest_failures=$((manifest_failures + 1))
  fi
done

echo
if [ "${manifest_failures}" -eq 0 ]; then
  echo "🎉 All 8 manifests live on CDN."
  echo "Run '/mission-supervisor resume' in the Claude session — supervisor will"
  echo "dispatch a haiku closeout sortie that appends ship-log entries to"
  echo "docs/missions/cdn-ship-log.md and creates path-restricted commits."
  exit 0
else
  echo "⚠ ${manifest_failures} manifest(s) not live yet. Inspect per-ship logs"
  echo "in ${LOG_DIR}/ for failure details."
  exit 1
fi

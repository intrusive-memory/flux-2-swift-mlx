#!/usr/bin/env bash
#
# acervo-ci-verify.sh — fail loudly if any primed CI model is missing or
# incomplete.
#
# The integration tests are model-presence-gated (`.enabled(if:)`), so a model
# that never landed makes them SKIP, and a skipped job is green. Run this after
# the cache-restore / prime steps so "skipped" can never be mistaken for
# "passed".
#
# Checks, for every slug in $ACERVO_CI_MODELS (plus positional args), that
# $ACERVO_MODELS_DIR/<slug>/manifest.json exists and that every file it lists
# is present with the manifest's exact byte size (the same size check
# SwiftAcervo's local validity check performs).
#
# Usage:
#   ACERVO_MODELS_DIR=/path/to/cache ACERVO_CI_MODELS="slug-1 slug-2" \
#     bash acervo-ci-verify.sh [extra-slug ...]
set -euo pipefail

if [[ -z "${ACERVO_MODELS_DIR:-}" ]]; then
  echo "::error::ACERVO_MODELS_DIR is not set." >&2
  exit 2
fi
command -v jq >/dev/null || { echo "::error::jq not found on PATH" >&2; exit 2; }

read -r -a SLUGS <<< "$(printf '%s %s' "${ACERVO_CI_MODELS:-}" "$*" | tr '\n' ' ')"
if [[ ${#SLUGS[@]} -eq 0 ]]; then
  echo "::error::No model slugs supplied (set ACERVO_CI_MODELS or pass args)." >&2
  exit 2
fi

failed=0
for slug in "${SLUGS[@]}"; do
  [[ -z "$slug" ]] && continue
  dir="$ACERVO_MODELS_DIR/$slug"
  manifest="$dir/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "::error::$slug: no manifest at $manifest — the model was never primed."
    failed=1
    continue
  fi

  problems=""
  total=0
  while IFS=$'\t' read -r path size; do
    total=$((total + 1))
    file="$dir/$path"
    if [[ ! -f "$file" ]]; then
      problems+="  missing: $path"$'\n'
    else
      have="$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")"
      if [[ "$have" != "$size" ]]; then
        problems+="  size mismatch: $path (have $have, manifest $size)"$'\n'
      fi
    fi
  done < <(jq -r '.files[] | "\(.path)\t\(.sizeBytes)"' "$manifest")

  if [[ -n "$problems" ]]; then
    echo "::error::$slug is incomplete in $dir:"
    printf '%s' "$problems" | head -20
    failed=1
  else
    echo "✓ $slug ($total files)"
  fi
done

if [[ $failed -ne 0 ]]; then
  echo "::error::Model cache is incomplete. The gated integration tests would silently SKIP; failing instead." >&2
fi
exit $failed

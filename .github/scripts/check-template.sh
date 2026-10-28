#!/usr/bin/env bash
# Checks the pull request body keeps the template's required sections and
# that each has something in it once the template's comments are removed.
set -euo pipefail

required=(What Checks)
fail=0

body=$(printf '%s' "${BODY:-}" | tr -d '\r' | perl -0pe 's/<!--.*?-->//gs')

for heading in "${required[@]}"; do
  if ! grep -qx "## $heading" <<<"$body"; then
    echo "::error::the $heading section is missing"
    fail=1
    continue
  fi
  section=$(awk -v h="## $heading" '$0 == h { on = 1; next } /^## / { on = 0 } on' <<<"$body")
  if ! [[ "$section" =~ [^[:space:]] ]]; then
    echo "::error::the $heading section is empty"
    fail=1
  fi
done

exit "$fail"

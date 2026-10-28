#!/usr/bin/env bash
# Checks the pull request title and every commit header between BASE and
# HEAD: a conventional type, a subject, and at most 50 characters. A trailing
# pull request number, as a squash into dev leaves it, is not counted.
set -euo pipefail

re='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^)]+\))?!?: [^ ]'
fail=0

check() {
  local what=$1 header=$2 bare
  bare=$(sed -E 's/ \(#[0-9]+\)$//' <<<"$header")
  if ! [[ "$bare" =~ $re ]]; then
    echo "::error::$what is not conventional: $header"
    fail=1
  fi
  if [ "${#bare}" -gt 50 ]; then
    echo "::error::$what is over 50 characters (${#bare}): $header"
    fail=1
  fi
}

check "the title" "$TITLE"
while IFS= read -r line; do
  check "commit ${line%% *}" "${line#* }"
done < <(git log --format='%h %s' "$BASE..$HEAD")

exit "$fail"

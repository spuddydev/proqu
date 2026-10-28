#!/bin/sh
# Installs or updates proqu for the current user.
#
#   curl -fsSL https://raw.githubusercontent.com/spuddydev/proqu/main/install.sh | sh
#
# The files go to ~/.local/share/proqu and the command is linked into
# ~/.local/bin. When Claude Code is installed, the plugin is installed from that
# same folder, so the agent and the owner run the same copy.
#
# Settings, all optional:
#   PROQU_VERSION   a release tag such as v1.0.0, default the latest release
#   PROQU_DIR       where the files go, default ~/.local/share/proqu
#   PROQU_BIN_DIR   where the command is linked, default ~/.local/bin
#   PROQU_ARCHIVE   install from this local archive instead of downloading
#   PROQU_NO_PLUGIN set to 1 to skip the Claude Code plugin

set -eu

repo="spuddydev/proqu"
home="${PROQU_DIR:-$HOME/.local/share/proqu}"
bin_dir="${PROQU_BIN_DIR:-$HOME/.local/bin}"
version="${PROQU_VERSION:-latest}"

say() { printf 'proqu: %s\n' "$*"; }
fail() {
	printf 'proqu: %s\n' "$*" >&2
	exit 1
}

case "$(uname -s)" in
Linux) ;;
*) fail "only Linux is supported for now, this is $(uname -s)" ;;
esac

command -v python3 >/dev/null 2>&1 || fail "python3 is needed, version 3.10 or newer"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
	fail "python3 is $(python3 -c 'import platform; print(platform.python_version())'), proqu needs 3.10 or newer"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -n "${PROQU_ARCHIVE:-}" ]; then
	cp "$PROQU_ARCHIVE" "$work/proqu.tar.gz"
else
	if [ "$version" = latest ]; then
		url="https://github.com/$repo/releases/latest/download/proqu.tar.gz"
	else
		url="https://github.com/$repo/releases/download/$version/proqu.tar.gz"
	fi
	say "downloading $url"
	if command -v curl >/dev/null 2>&1; then
		curl -fsSL "$url" -o "$work/proqu.tar.gz" || fail "download failed"
	elif command -v wget >/dev/null 2>&1; then
		wget -q "$url" -O "$work/proqu.tar.gz" || fail "download failed"
	else
		fail "curl or wget is needed to download proqu"
	fi
fi

tar -xzf "$work/proqu.tar.gz" -C "$work"
[ -x "$work/proqu/bin/proqu" ] || fail "the archive does not hold proqu"

# swap the new files in whole, so a failed install never leaves half of each
mkdir -p "$(dirname "$home")"
rm -rf "$home.new" "$home.old"
mv "$work/proqu" "$home.new"
[ -e "$home" ] && mv "$home" "$home.old"
mv "$home.new" "$home"
rm -rf "$home.old"

mkdir -p "$bin_dir"
ln -sfn "$home/bin/proqu" "$bin_dir/proqu"
say "installed $("$home/bin/proqu" --version) to $home"
say "linked the command at $bin_dir/proqu"
case ":$PATH:" in
*":$bin_dir:"*) ;;
*) say "add $bin_dir to your PATH to run proqu yourself" ;;
esac

if [ "${PROQU_NO_PLUGIN:-0}" = 1 ]; then
	exit 0
fi
if ! command -v claude >/dev/null 2>&1; then
	say "Claude Code was not found. Once it is installed, run this again or:"
	say "  claude plugin marketplace add $home"
	say "  claude plugin install proqu@proqu"
	exit 0
fi

if claude plugin marketplace list 2>/dev/null | grep -q proqu; then
	claude plugin marketplace update proqu >/dev/null
	claude plugin update proqu@proqu >/dev/null || claude plugin install proqu@proqu >/dev/null
	say "updated the Claude Code plugin, restart Claude Code to use it"
else
	claude plugin marketplace add "$home" >/dev/null
	claude plugin install proqu@proqu >/dev/null
	say "installed the Claude Code plugin, restart Claude Code to use it"
fi

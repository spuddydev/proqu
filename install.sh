#!/bin/sh
# Installs or updates proqu for the current user.
#
#   curl -fsSL https://raw.githubusercontent.com/spuddydev/proqu/main/install.sh | sh
#
# The files go to ~/.local/share/proqu and the command is linked into
# ~/.local/bin. When Claude Code is installed, the plugin is installed from that
# same folder, so the agent and the owner always run the same release.
#
# Settings, all optional:
#   PROQU_VERSION   a release tag such as v1.0.0, default the latest release
#   PROQU_DIR       where the files go, default ~/.local/share/proqu
#   PROQU_BIN_DIR   where the command is linked, default ~/.local/bin
#   PROQU_ARCHIVE   install from this local archive instead of downloading
#   PROQU_NO_PLUGIN set to 1 to skip the Claude Code plugin

set -eu

repo="spuddydev/proqu"
# written into the install folder, so an update only ever replaces a folder this script made
marker=".proqu-install"

say() { printf 'proqu: %s\n' "$*"; }
fail() {
	printf 'proqu: %s\n' "$*" >&2
	exit 1
}

# the source of the marketplace called proqu, empty if there is none
marketplace_source() {
	claude plugin marketplace list --json </dev/null 2>/dev/null | python3 -c '
import json, sys
try:
    entries = json.load(sys.stdin)
except ValueError:
    entries = []
for entry in entries:
    if entry.get("name") == "proqu":
        print(entry.get("path") or entry.get("repo") or entry.get("installLocation") or "elsewhere")
'
}

install_plugin() {
	home="$1"
	if ! command -v claude >/dev/null 2>&1; then
		say "Claude Code was not found. Once it is installed, run this again or:"
		say "  claude plugin marketplace add $home"
		say "  claude plugin install proqu@proqu"
		return
	fi
	current="$(marketplace_source)"
	if [ -z "$current" ]; then
		claude plugin marketplace add "$home" </dev/null >/dev/null || fail "could not add the plugin marketplace"
		claude plugin install proqu@proqu </dev/null >/dev/null || fail "could not install the plugin"
		say "installed the Claude Code plugin, restart Claude Code to use it"
	elif [ "$current" = "$home" ]; then
		claude plugin marketplace update proqu </dev/null >/dev/null || fail "could not update the plugin marketplace"
		claude plugin update proqu@proqu </dev/null >/dev/null ||
			claude plugin install proqu@proqu </dev/null >/dev/null || fail "could not update the plugin"
		say "updated the Claude Code plugin, restart Claude Code to use it"
	else
		say "the Claude Code plugin comes from $current, so it was left as it is"
		say "to use this install instead, run:"
		say "  claude plugin marketplace remove proqu"
		say "  claude plugin marketplace add $home"
		say "  claude plugin install proqu@proqu"
	fi
}

main() {
	home="${PROQU_DIR:-$HOME/.local/share/proqu}"
	bin_dir="${PROQU_BIN_DIR:-$HOME/.local/bin}"
	version="${PROQU_VERSION:-latest}"

	case "$(uname -s)" in
	Linux) ;;
	*) fail "only Linux is supported for now, this is $(uname -s)" ;;
	esac

	command -v python3 >/dev/null 2>&1 || fail "python3 is needed, version 3.10 or newer"
	python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
		fail "python3 is $(python3 -c 'import platform; print(platform.python_version())'), proqu needs 3.10 or newer"

	if [ -e "$home" ] && [ ! -e "$home/$marker" ]; then
		fail "$home already exists and was not made by this installer, move it or set PROQU_DIR"
	fi

	work="$(mktemp -d)"
	trap 'rm -rf "$work"' EXIT
	trap 'exit 130' INT
	trap 'exit 143' TERM
	trap 'exit 129' HUP

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

	tar -xzf "$work/proqu.tar.gz" -C "$work" || fail "the archive could not be unpacked"
	[ -x "$work/proqu/bin/proqu" ] || fail "the archive does not hold proqu"
	: >"$work/proqu/$marker"

	# swap the new files in whole, so a failed install never leaves half of each
	mkdir -p "$(dirname "$home")"
	rm -rf "$home.new" "$home.old"
	mv "$work/proqu" "$home.new"
	if [ -e "$home" ]; then
		mv "$home" "$home.old"
	fi
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

	if [ "${PROQU_NO_PLUGIN:-0}" != 1 ]; then
		install_plugin "$home"
	fi
}

# nothing runs until the whole script has arrived, so a cut off download does nothing
main "$@"

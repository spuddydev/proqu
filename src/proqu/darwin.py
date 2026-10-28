"""The macOS process and machine queries proqu needs, in place of /proc.

Everything here is standard library only. The native calls go through ctypes into
libproc and libc, and each query has a fallback that runs a stock command such as ps.
No library is loaded at import time, so the module imports cleanly on Linux too.
"""

import ctypes
import ctypes.util
import os
import re
import struct
import subprocess
import sys
import time

PROC_PIDTBSDINFO = 3
# asks proc_pidinfo for a zombie too, which it otherwise reports as missing
FIND_ZOMBIE = 1
RUSAGE_INFO_V0 = 0
SZOMB = 5
CTL_KERN = 1
KERN_ARGMAX = 8
KERN_PROCARGS2 = 49
# ri_phys_footprint sits at this offset in every rusage_info version
_FOOTPRINT_OFFSET = 72
_RUSAGE_BUFFER = 512
_DEFAULT_ARGMAX = 1024 * 1024
_MAX_ARGMAX = 16 * 1024 * 1024
_MAX_PID = 2**31 - 1
_COMMAND_TIMEOUT = 5

_CALL_ERRORS = (OSError, ctypes.ArgumentError, ValueError, OverflowError)


class _ProcBsdInfo(ctypes.Structure):
    """struct proc_bsdinfo from libproc, 136 bytes."""

    _fields_ = (
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    )


class _Lib:
    """The native functions in use. Every buffer is passed as an address, or None for NULL."""

    def __init__(self, libproc, libc):
        vp = ctypes.c_void_p
        self.proc_pidinfo = libproc.proc_pidinfo
        self.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, vp, ctypes.c_int]
        self.proc_pidinfo.restype = ctypes.c_int
        self.proc_listallpids = libproc.proc_listallpids
        self.proc_listallpids.argtypes = [vp, ctypes.c_int]
        self.proc_listallpids.restype = ctypes.c_int
        self.proc_pid_rusage = libproc.proc_pid_rusage
        self.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, vp]
        self.proc_pid_rusage.restype = ctypes.c_int
        self.sysctl = libc.sysctl
        self.sysctl.argtypes = [vp, ctypes.c_uint, vp, vp, vp, ctypes.c_size_t]
        self.sysctl.restype = ctypes.c_int
        self.sysctlbyname = libc.sysctlbyname
        self.sysctlbyname.argtypes = [ctypes.c_char_p, vp, vp, vp, ctypes.c_size_t]
        self.sysctlbyname.restype = ctypes.c_int


_loaded = False
_cached: _Lib | None = None


def _lib() -> _Lib | None:
    """The native functions, loaded once, or None where they are not available.

    Tests replace this to run the queries against a fake.
    """
    global _loaded, _cached
    if _loaded:
        return _cached
    _loaded = True
    try:
        try:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        except OSError:
            # the same functions are also exported by libSystem
            libproc = ctypes.CDLL(None, use_errno=True)
        try:
            libc = ctypes.CDLL(None, use_errno=True)
        except OSError:
            path = ctypes.util.find_library("c")
            if not path:
                return None
            libc = ctypes.CDLL(path, use_errno=True)
        _cached = _Lib(libproc, libc)
    except (OSError, AttributeError):
        _cached = None
    return _cached


def _run(command: list[str]) -> str | None:
    """The output of a stock command, or None if it is missing, fails or hangs."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return result.stdout


def _by_name(lib: _Lib, name: str) -> bytes | None:
    """The raw value of a named sysctl, or None."""
    key = name.encode()
    size = ctypes.c_size_t(0)
    try:
        if lib.sysctlbyname(key, None, ctypes.addressof(size), None, 0) != 0:
            return None
        if not 0 < size.value <= 65536:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if lib.sysctlbyname(key, ctypes.addressof(buf), ctypes.addressof(size), None, 0) != 0:
            return None
        return buf.raw[: size.value]
    except _CALL_ERRORS:
        return None


def _int_by_name(name: str) -> int | None:
    lib = _lib()
    raw = _by_name(lib, name) if lib else None
    if raw is None or len(raw) not in (4, 8):
        return None
    return int.from_bytes(raw, sys.byteorder)


def _command_int(name: str) -> int | None:
    out = _run(["sysctl", "-n", name])
    try:
        return int(out.strip()) if out else None
    except ValueError:
        return None


def total_ram() -> int:
    """Physical RAM in bytes. Raises OSError when the machine cannot be read."""
    value = _int_by_name("hw.memsize")
    if value is None:
        value = _command_int("hw.memsize")
    if not value or value <= 0:
        raise OSError("cannot read hw.memsize")
    return value


def free_ram() -> int | None:
    """Memory available for new work right now, in bytes, or None if vm_stat cannot be read.

    This is an estimate, free plus inactive plus speculative pages, standing in for the
    Linux MemAvailable. It is not verified against what Activity Monitor reports.
    """
    out = _run(["vm_stat"])
    if not out:
        return None
    header = re.search(r"page size of (\d+) bytes", out)
    if not header:
        return None
    pages = {}
    for match in re.finditer(r"^Pages (free|inactive|speculative):\s+(\d+)\.", out, re.M):
        pages[match.group(1)] = int(match.group(2))
    if len(pages) != 3:
        return None
    return sum(pages.values()) * int(header.group(1))


def cpu_count() -> int:
    """Logical CPUs, performance and efficiency cores alike, and at least 1."""
    count = os.cpu_count()
    if not count:
        count = _int_by_name("hw.logicalcpu") or _command_int("hw.logicalcpu")
    return max(1, count or 1)


def boot_id() -> str:
    """An id that changes on every boot, so pids from before a restart are never trusted."""
    lib = _lib()
    if lib:
        raw = _by_name(lib, "kern.bootsessionuuid")
        if raw:
            text = raw.split(b"\0", 1)[0].decode(errors="replace").strip()
            if text:
                return text
        raw = _by_name(lib, "kern.boottime")
        # a struct timeval, whose first field is the boot time in whole seconds
        if raw and len(raw) >= 8:
            return str(int.from_bytes(raw[:8], sys.byteorder, signed=True))
    out = _run(["sysctl", "-n", "kern.bootsessionuuid"])
    return out.strip() if out else ""


def _valid_pid(pid: int) -> bool:
    return isinstance(pid, int) and 0 < pid <= _MAX_PID


def _process_from_ps(pid: int) -> tuple[int, int, bool] | None:
    out = _run(["ps", "-o", "uid=,stat=,lstart=", "-p", str(pid)])
    if not out:
        return None
    parts = out.split(None, 2)
    if len(parts) != 3:
        return None
    try:
        uid = int(parts[0])
        started = time.strptime(" ".join(parts[2].split()), "%a %b %d %H:%M:%S %Y")
        # whole seconds, to agree exactly with pbi_start_tvsec for the same process
        start = int(time.mktime((*started[:8], -1)))
    except (ValueError, OverflowError):
        return None
    return uid, start, parts[1].startswith("Z")


def process(pid: int) -> tuple[int, int, bool] | None:
    """(uid, start, zombie) for a pid, or None if it does not exist or cannot be read.

    The start is in whole seconds since the epoch. It is stored and compared later to
    spot a reused pid, so the libproc and ps routes must give the same number.
    """
    if not _valid_pid(pid):
        return None
    lib = _lib()
    if lib is None:
        return _process_from_ps(pid)
    info = _ProcBsdInfo()
    size = ctypes.sizeof(info)
    try:
        filled = lib.proc_pidinfo(pid, PROC_PIDTBSDINFO, FIND_ZOMBIE, ctypes.addressof(info), size)
    except _CALL_ERRORS:
        return None
    if filled != size:
        return None
    return info.pbi_uid, int(info.pbi_start_tvsec), info.pbi_status == SZOMB


def start_time(pid: int) -> int | None:
    """The start of a process as process() reports it, or None."""
    found = process(pid)
    return found[1] if found else None


def pids() -> list[int]:
    """Every pid on the machine, or an empty list if they cannot be listed."""
    lib = _lib()
    if lib is None:
        out = _run(["ps", "-axo", "pid="])
        found = []
        for word in (out or "").split():
            if word.isdigit():
                found.append(int(word))
        return found
    try:
        count = lib.proc_listallpids(None, 0)
        if count <= 0:
            return []
        # processes can start between the two calls, so leave room
        room = count + 64
        buf = (ctypes.c_int * room)()
        filled = lib.proc_listallpids(ctypes.addressof(buf), ctypes.sizeof(buf))
    except _CALL_ERRORS:
        return []
    if filled <= 0:
        return []
    return [pid for pid in buf[: min(filled, room)] if pid > 0]


_argmax_cached: int | None = None


def _argmax(lib: _Lib) -> int:
    """The largest argument and environment block, which does not change while running."""
    global _argmax_cached
    if _argmax_cached is None:
        _argmax_cached = _read_argmax(lib)
    return _argmax_cached


def _read_argmax(lib: _Lib) -> int:
    mib = (ctypes.c_int * 2)(CTL_KERN, KERN_ARGMAX)
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    try:
        ok = lib.sysctl(
            ctypes.addressof(mib), 2, ctypes.addressof(value), ctypes.addressof(size), None, 0
        )
    except _CALL_ERRORS:
        return _DEFAULT_ARGMAX
    if ok != 0 or not 0 < value.value <= _MAX_ARGMAX:
        return _DEFAULT_ARGMAX
    return value.value


def parse_procargs(data: bytes) -> list[bytes] | None:
    """The environment entries in a KERN_PROCARGS2 result, or None if there is no environment.

    The layout is a native int argc, the executable path, padding NULs, argc arguments and
    then the environment, every string NUL terminated. A last string cut off by the end of
    the data is dropped, as it could be a partial value.
    """
    if len(data) < 4:
        return None
    argc = int.from_bytes(data[:4], sys.byteorder)
    if argc > len(data):
        return None
    end = data.find(b"\0", 4)
    if end < 0:
        return None
    pos = end
    while pos < len(data) and data[pos] == 0:
        pos += 1
    for _ in range(argc):
        end = data.find(b"\0", pos)
        if end < 0:
            return None
        pos = end + 1
    entries = []
    while pos < len(data):
        end = data.find(b"\0", pos)
        if end < 0:
            break
        if end == pos:
            break
        entries.append(data[pos:end])
        pos = end + 1
    return entries


def _environ_from_ps(pid: int) -> list[bytes] | None:
    out = _run(["ps", "-E", "-ww", "-o", "command=", "-p", str(pid)])
    if out is None:
        return None
    # weaker than the sysctl route, as an argument can look like a variable
    return [word.encode() for word in out.split() if "=" in word]


def environ(pid: int) -> list[bytes] | None:
    """The environment entries (b"NAME=value") of a process of the same user, or None."""
    if not _valid_pid(pid):
        return None
    lib = _lib()
    if lib is None:
        return _environ_from_ps(pid)
    room = _argmax(lib)
    mib = (ctypes.c_int * 3)(CTL_KERN, KERN_PROCARGS2, pid)
    buf = ctypes.create_string_buffer(room)
    size = ctypes.c_size_t(room)
    try:
        ok = lib.sysctl(
            ctypes.addressof(mib), 3, ctypes.addressof(buf), ctypes.addressof(size), None, 0
        )
    except _CALL_ERRORS:
        return None
    if ok != 0:
        return None
    # only the part the kernel filled, as copying the whole buffer is costly
    return parse_procargs(ctypes.string_at(buf, min(size.value, room)))


def _rss_from_ps(pid: int) -> int:
    out = _run(["ps", "-o", "rss=", "-p", str(pid)])
    try:
        # leaves out compressed memory, so this undercounts
        return int(out.strip()) * 1024 if out else 0
    except ValueError:
        return 0


def footprint(pid: int) -> int:
    """The physical footprint of a process in bytes, or 0 when it is gone.

    This is what Activity Monitor shows as Memory, including compressed and GPU memory.
    """
    if not _valid_pid(pid):
        return 0
    lib = _lib()
    if lib is None:
        return _rss_from_ps(pid)
    # far larger than the v0 struct, so a kernel that writes more cannot overrun it
    buf = ctypes.create_string_buffer(_RUSAGE_BUFFER)
    try:
        if lib.proc_pid_rusage(pid, RUSAGE_INFO_V0, ctypes.addressof(buf)) != 0:
            return 0
    except _CALL_ERRORS:
        return 0
    return int(struct.unpack_from("=Q", buf.raw, _FOOTPRINT_OFFSET)[0])

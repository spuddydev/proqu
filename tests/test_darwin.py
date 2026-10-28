import calendar
import ctypes
import os
import stat
import struct
import subprocess
import sys
import time

import pytest

from proqu import darwin

MAC = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
MIB = 1024**2
PS_START = calendar.timegm((2026, 10, 28, 10, 0, 0))


class FakeLib:
    """Stands in for the native functions, filling the buffers it is given from canned values."""

    def __init__(self, procs=None, names=None, mibs=None, footprints=None, pidlist=()):
        self.procs = procs or {}
        self.names = names or {}
        self.mibs = mibs or {}
        self.footprints = footprints or {}
        self.pidlist = list(pidlist)
        self.pidinfo_size = None

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        if flavor != darwin.PROC_PIDTBSDINFO or pid not in self.procs:
            return 0
        uid, start, status = self.procs[pid]
        # as the kernel does, a zombie is only found when asked for with arg 1
        if status == darwin.SZOMB and arg != 1:
            return 0
        info = darwin._ProcBsdInfo.from_address(buf)
        info.pbi_pid, info.pbi_uid, info.pbi_status = pid, uid, status
        info.pbi_start_tvsec = start
        info.pbi_start_tvusec = 999999
        return ctypes.sizeof(info) if self.pidinfo_size is None else self.pidinfo_size

    def proc_listallpids(self, buf, size):
        if buf is None:
            return len(self.pidlist)
        room = size // ctypes.sizeof(ctypes.c_int)
        found = self.pidlist[:room]
        (ctypes.c_int * len(found)).from_address(buf)[:] = found
        return len(found)

    def proc_pid_rusage(self, pid, flavor, buf):
        if pid not in self.footprints:
            return -1
        ctypes.memmove(buf + 72, struct.pack("=Q", self.footprints[pid]), 8)
        return 0

    @staticmethod
    def _answer(value, oldp, oldlenp):
        if value is None:
            return -1
        length = ctypes.c_size_t.from_address(oldlenp)
        if oldp is None:
            length.value = len(value)
            return 0
        if length.value < len(value):
            return -1
        ctypes.memmove(oldp, value, len(value))
        length.value = len(value)
        return 0

    def sysctlbyname(self, name, oldp, oldlenp, newp, newlen):
        return self._answer(self.names.get(name.decode()), oldp, oldlenp)

    def sysctl(self, mib, namelen, oldp, oldlenp, newp, newlen):
        key = tuple((ctypes.c_int * namelen).from_address(mib))
        return self._answer(self.mibs.get(key), oldp, oldlenp)


def use_lib(monkeypatch, lib):
    monkeypatch.setattr(darwin, "_lib", lambda: lib)
    monkeypatch.setattr(darwin, "_argmax_cached", None)


def fake_bin(tmp_path, monkeypatch, **scripts):
    """Fake commands as the only things on PATH."""
    for name, body in scripts.items():
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))


def procargs(argc, path, args, env, pad=3):
    parts = struct.pack("=i", argc) + path + b"\0" * (pad + 1)
    for arg in args:
        parts += arg + b"\0"
    for entry in env:
        parts += entry + b"\0"
    return parts


@pytest.fixture
def utc(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_struct_size():
    assert ctypes.sizeof(darwin._ProcBsdInfo) == 136


def test_imports_without_loading_native_libraries():
    assert darwin._cached is None or isinstance(darwin._cached, darwin._Lib)


def test_total_ram_from_sysctl(monkeypatch):
    raw = struct.pack("=Q", 16 * 1024**3)
    use_lib(monkeypatch, FakeLib(names={"hw.memsize": raw}))
    assert darwin.total_ram() == 16 * 1024**3


def test_total_ram_falls_back_to_command(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, sysctl='[ "$2" = hw.memsize ] && echo 17179869184')
    assert darwin.total_ram() == 17179869184


def test_total_ram_raises_when_unreadable(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, sysctl="exit 1")
    with pytest.raises(OSError):
        darwin.total_ram()


def test_total_ram_raises_without_tools(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(OSError):
        darwin.total_ram()


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                            200000.
Pages inactive:                           50000.
Pages speculative:                         2000.
Pages throttled:                              0.
Pages wired down:                         90000.
"""


def test_free_ram_parses_vm_stat(tmp_path, monkeypatch):
    fake_bin(tmp_path, monkeypatch, vm_stat=f"printf '%s' '{VM_STAT}'")
    assert darwin.free_ram() == (10000 + 50000 + 2000) * 16384


def test_free_ram_none_without_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert darwin.free_ram() is None


@pytest.mark.parametrize(
    "text",
    [
        "nonsense",
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 5.\n",
        "Pages free: 5.\nPages inactive: 5.\nPages speculative: 5.\n",
    ],
)
def test_free_ram_none_on_odd_output(tmp_path, monkeypatch, text):
    fake_bin(tmp_path, monkeypatch, vm_stat=f"printf '%s' '{text}'")
    assert darwin.free_ram() is None


def test_cpu_count_uses_os(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    assert darwin.cpu_count() == 10


def test_cpu_count_falls_back_to_sysctl(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    use_lib(monkeypatch, FakeLib(names={"hw.logicalcpu": struct.pack("=i", 8)}))
    assert darwin.cpu_count() == 8


def test_cpu_count_falls_back_to_command(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, sysctl='[ "$2" = hw.logicalcpu ] && echo 12')
    assert darwin.cpu_count() == 12


def test_cpu_count_at_least_one(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    use_lib(monkeypatch, None)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert darwin.cpu_count() == 1


def test_boot_id_prefers_session_uuid(monkeypatch):
    names = {
        "kern.bootsessionuuid": b"ABCD-1234\0",
        "kern.boottime": struct.pack("=qi4x", 1700000000, 5),
    }
    use_lib(monkeypatch, FakeLib(names=names))
    assert darwin.boot_id() == "ABCD-1234"


def test_boot_id_falls_back_to_boottime_seconds(monkeypatch):
    names = {"kern.boottime": struct.pack("=qi4x", 1700000000, 5)}
    use_lib(monkeypatch, FakeLib(names=names))
    assert darwin.boot_id() == "1700000000"


def test_boot_id_falls_back_to_command(tmp_path, monkeypatch):
    use_lib(monkeypatch, FakeLib())
    fake_bin(tmp_path, monkeypatch, sysctl='[ "$2" = kern.bootsessionuuid ] && echo ZZZ-1')
    assert darwin.boot_id() == "ZZZ-1"


def test_boot_id_empty_on_failure(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert darwin.boot_id() == ""


def test_process_from_libproc(monkeypatch):
    lib = FakeLib(procs={42: (501, PS_START, 2), 43: (0, PS_START + 1, darwin.SZOMB)})
    use_lib(monkeypatch, lib)
    assert darwin.process(42) == (501, PS_START, False)
    assert darwin.process(43) == (0, PS_START + 1, True)
    assert darwin.start_time(42) == PS_START
    assert darwin.process(99) is None
    assert darwin.start_time(99) is None


def test_process_short_fill_is_failure(monkeypatch):
    lib = FakeLib(procs={42: (501, PS_START, 2)})
    lib.pidinfo_size = 100
    use_lib(monkeypatch, lib)
    assert darwin.process(42) is None


@pytest.mark.parametrize("pid", [0, -1, 2**40])
def test_process_rejects_bad_pids(monkeypatch, pid):
    use_lib(monkeypatch, FakeLib())
    assert darwin.process(pid) is None
    assert darwin.environ(pid) is None
    assert darwin.footprint(pid) == 0


def test_process_from_ps(tmp_path, monkeypatch, utc):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps='echo "  501 Ss   Wed Oct 28 10:00:00 2026"')
    assert darwin.process(42) == (501, PS_START, False)


def test_process_from_ps_zombie_and_padded_day(tmp_path, monkeypatch, utc):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps='echo "  0 Z+   Sun Nov  1 09:05:07 2026"')
    uid, start, zombie = darwin.process(42)
    assert (uid, zombie) == (0, True)
    assert start == calendar.timegm((2026, 11, 1, 9, 5, 7))


def test_process_from_ps_missing_process(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps="exit 1")
    assert darwin.process(42) is None


def test_process_from_ps_garbage(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps="echo 'what is this'")
    assert darwin.process(42) is None


def test_libproc_and_ps_agree_on_start(tmp_path, monkeypatch, utc):
    use_lib(monkeypatch, FakeLib(procs={42: (501, PS_START, 2)}))
    native = darwin.process(42)
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps='echo "  501 Ss   Wed Oct 28 10:00:00 2026"')
    assert darwin.process(42) == native


def test_pids_from_libproc(monkeypatch):
    use_lib(monkeypatch, FakeLib(pidlist=[1, 50, 51, 900]))
    assert darwin.pids() == [1, 50, 51, 900]


def test_pids_empty_on_failure(monkeypatch):
    use_lib(monkeypatch, FakeLib())
    assert darwin.pids() == []


def test_pids_from_ps(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps='printf "    1\\n  345\\n99999\\n"')
    assert darwin.pids() == [1, 345, 99999]


def test_pids_empty_without_ps(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert darwin.pids() == []


def test_parse_procargs_normal():
    data = procargs(2, b"/usr/bin/python3", [b"python3", b"x.py"], [b"A=1", b"B=two=2"])
    assert darwin.parse_procargs(data) == [b"A=1", b"B=two=2"]


def test_parse_procargs_stops_at_empty_string():
    data = procargs(1, b"/bin/sh", [b"sh"], [b"A=1"]) + b"\0\0/bin/sh\0"
    assert darwin.parse_procargs(data) == [b"A=1"]


def test_parse_procargs_empty_env():
    assert darwin.parse_procargs(procargs(1, b"/bin/sh", [b"sh"], [])) == []


def test_parse_procargs_drops_cut_last_entry():
    data = procargs(1, b"/bin/sh", [b"sh"], [b"A=1"]) + b"B=par"
    assert darwin.parse_procargs(data) == [b"A=1"]


def test_parse_procargs_truncated_in_args():
    data = procargs(3, b"/bin/sh", [b"sh"], [])
    assert darwin.parse_procargs(data) is None


@pytest.mark.parametrize(
    "data",
    [b"", b"\x01", struct.pack("=i", 1), struct.pack("=i", 1) + b"/bin/sh", b"\xff" * 40],
)
def test_parse_procargs_odd_data_never_raises(data):
    assert darwin.parse_procargs(data) is None


def test_environ_from_sysctl(monkeypatch):
    data = procargs(1, b"/bin/sh", [b"sh"], [b"PROQU_JOB_ID=xyz", b"HOME=/h"])
    mibs = {
        (darwin.CTL_KERN, darwin.KERN_ARGMAX): struct.pack("=i", 262144),
        (darwin.CTL_KERN, darwin.KERN_PROCARGS2, 42): data,
    }
    use_lib(monkeypatch, FakeLib(mibs=mibs))
    assert darwin.environ(42) == [b"PROQU_JOB_ID=xyz", b"HOME=/h"]


def test_environ_default_argmax_and_unreadable(monkeypatch):
    data = procargs(1, b"/bin/sh", [b"sh"], [b"A=1"])
    use_lib(monkeypatch, FakeLib(mibs={(darwin.CTL_KERN, darwin.KERN_PROCARGS2, 42): data}))
    assert darwin.environ(42) == [b"A=1"]
    assert darwin.environ(43) is None


def test_environ_from_ps(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(
        tmp_path,
        monkeypatch,
        ps='echo "/bin/sh -c x HOME=/Users/me PROQU_JOB_ID=xyz TERM=xterm-256color"',
    )
    assert darwin.environ(42) == [b"HOME=/Users/me", b"PROQU_JOB_ID=xyz", b"TERM=xterm-256color"]


def test_environ_from_ps_missing_process(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps="exit 1")
    assert darwin.environ(42) is None


def test_footprint_from_rusage(monkeypatch):
    use_lib(monkeypatch, FakeLib(footprints={42: 123 * MIB}))
    assert darwin.footprint(42) == 123 * MIB
    assert darwin.footprint(43) == 0


def test_footprint_from_ps(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps='echo "  2048"')
    assert darwin.footprint(42) == 2048 * 1024


def test_footprint_zero_when_gone(tmp_path, monkeypatch):
    use_lib(monkeypatch, None)
    fake_bin(tmp_path, monkeypatch, ps="exit 1")
    assert darwin.footprint(42) == 0


def test_lib_none_when_symbols_missing(monkeypatch):
    class Empty:
        pass

    monkeypatch.setattr(darwin, "_loaded", False)
    monkeypatch.setattr(darwin, "_cached", None)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: Empty())
    assert darwin._lib() is None


@MAC
def test_real_machine_queries():
    assert darwin.total_ram() > 1024**3
    assert darwin.cpu_count() == os.cpu_count()
    assert darwin.boot_id()
    assert darwin.boot_id() == darwin.boot_id()
    free = darwin.free_ram()
    assert free is not None
    assert 0 < free < darwin.total_ram()


@MAC
def test_real_process_of_ourselves():
    uid, start, zombie = darwin.process(os.getpid())
    assert uid == os.getuid()
    assert time.time() - 86400 < start <= time.time()
    assert not zombie
    assert os.getpid() in darwin.pids()
    assert darwin.footprint(os.getpid()) > MIB


@MAC
def test_real_zombie_child():
    child = subprocess.Popen(["true"])
    try:
        time.sleep(0.3)
        assert darwin.process(child.pid)[2] is True
    finally:
        child.wait()


@MAC
def test_real_environ_of_child():
    env = {**os.environ, "PROQU_JOB_ID": "xyz"}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], env=env)
    try:
        assert b"PROQU_JOB_ID=xyz" in darwin.environ(child.pid)
    finally:
        child.kill()
        child.wait()


@MAC
def test_real_footprint_of_child():
    code = (
        "import time\n"
        "b = bytearray(100 * 1024 * 1024)\n"
        "for i in range(0, len(b), 4096):\n"
        "    b[i] = 1\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        assert darwin.footprint(child.pid) > 100 * MIB
    finally:
        child.kill()
        child.wait()


@MAC
def test_real_libproc_and_ps_agree_on_start(monkeypatch):
    native = darwin.process(os.getpid())
    monkeypatch.setattr(darwin, "_lib", lambda: None)
    assert darwin.process(os.getpid()) == native

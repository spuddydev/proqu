from proqu import report


def write(tmp_path, data: bytes):
    path = tmp_path / "job.log"
    path.write_bytes(data)
    return path


def test_tail_takes_the_last_lines(tmp_path):
    path = write(tmp_path, b"".join(b"%d\n" % i for i in range(1, 301)))
    assert report.tail_lines(path, 3) == ["298", "299", "300"]
    assert report.tail_lines(path, 500)[0] == "1"


def test_tail_edges(tmp_path):
    assert report.tail_lines(tmp_path / "missing.log", 5) is None
    assert report.tail_lines(write(tmp_path, b""), 5) == []
    assert report.tail_lines(write(tmp_path, b"a\n\nb"), 5) == ["a", "", "b"]
    assert report.tail_lines(write(tmp_path, b"\xffok\n"), 5) == ["�ok"]


def test_tail_keeps_the_last_drawing_of_a_progress_bar(tmp_path):
    path = write(tmp_path, b"10%\r50%\r100%\ndone\r\n")
    assert report.tail_lines(path, 5) == ["100%", "done"]


def test_tail_reads_only_the_end_of_a_huge_log(tmp_path):
    path = write(tmp_path, b"x" * (report.TAIL_BYTES * 2) + b"\nend\n")
    assert report.tail_lines(path, 5) == ["end"]
    one_line = write(tmp_path, b"y" * (report.TAIL_BYTES * 2))
    assert report.tail_lines(one_line, 5) == ["y" * report.TAIL_BYTES]

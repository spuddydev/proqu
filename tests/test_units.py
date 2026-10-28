import pytest

from proqu.units import format_duration, format_size, parse_duration, parse_size


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("512M", 512 * 1024**2),
        ("6G", 6 * 1024**3),
        ("6g", 6 * 1024**3),
        ("6GB", 6 * 1024**3),
        ("6GiB", 6 * 1024**3),
        ("1.5T", int(1.5 * 1024**4)),
        (" 64K ", 64 * 1024),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize(
    "text", ["6", "6X", "G", "", "-1G", "0G", "1.G", "nan", "1e3G", "1_0G", "\u0664G"]
)
def test_parse_size_refuses(text):
    with pytest.raises(ValueError):
        parse_size(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("90s", 90), ("20m", 1200), ("2h", 7200), ("1h30m", 5400), ("1d", 86400), ("1.5h", 5400)],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["90", "", "m", "1x", "1h30", "0s", "h1", "20M", "\u0664m"])
def test_parse_duration_refuses(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_format_size():
    assert format_size(4 * 1024**3) == "4.0G"
    assert format_size(1536 * 1024**2) == "1.5G"
    assert format_size(500) == "500B"


def test_format_duration():
    assert format_duration(5400) == "1h30m"
    assert format_duration(45) == "45s"
    assert format_duration(90061) == "1d1h1m"
    assert format_duration(-3) == "0s"


@pytest.mark.parametrize("text", ["20m", "1h30m", "2h", "1d1h1m"])
def test_duration_round_trip(text):
    assert format_duration(parse_duration(text)) == text


def test_uppercase_duration_gets_a_hint():
    with pytest.raises(ValueError, match="try 1h20m"):
        parse_duration("1H20M")


def test_format_size_rounds_down():
    shown = format_size(16541204480)
    assert shown == "15.4G"
    assert parse_size(shown) <= 16541204480

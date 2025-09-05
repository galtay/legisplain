from pathlib import Path
import re


BILLSTATUS_CONGRESS_NUMS = [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
TEXTVERSION_CONGRESS_NUMS = [113, 114, 115, 116, 117, 118, 119]

BILLSTATUS_PATH_PATTERN = re.compile(
    r"""data/
    (?P<congress_num>\d{3})/
    (?P<legis_class>\w+)/
    (?P<legis_type>\w+)/
    ([a-zA-Z]+)
    (?P<legis_num>\d+)/
    fdsys_billstatus\.xml
    """,
    re.VERBOSE,
)

TEXTVERSION_BILLS_PATH_PATTERN = re.compile(
    r"""data/govinfo/BILLS/
    (?P<congress_num_p>\d{3})/
    (?P<congress_session_p>\d{1})/
    (?P<legis_type_p>[a-zA-Z]+)/
    BILLS-
    (?P<congress_num>\d{3})
    (?P<legis_type>[a-zA-Z]+)
    (?P<legis_num>\d+)
    (?P<legis_version>\w+)
    \.xml
    """,
    re.VERBOSE,
)

TEXTVERSION_PLAW_PATH_PATTERN = re.compile(
    r"""data/govinfo/PLAW/
    (?P<congress_num_p>\d{3})/
    (?P<pvtl_or_publ_p>[a-zA-Z]+)/
    PLAW-
    (?P<congress_num>\d{3})
    (?P<legis_type>[a-zA-Z]+)
    (?P<legis_num>\d+)
    \.xml
    """,
    re.VERBOSE,
)


def get_paths_from_download_logs(congress_bulk_path: Path, log_path: Path):
    patterns = {
        "bs": BILLSTATUS_PATH_PATTERN,
        "tvb": TEXTVERSION_BILLS_PATH_PATTERN,
        "tvp": TEXTVERSION_PLAW_PATH_PATTERN,
    }

    with log_path.open("r") as fp:
        all_lines = fp.readlines()

    matches = {"bs": [], "tvb": [], "tvp": []}
    paths = {"bs": [], "tvb": [], "tvp": []}
    lines = {"bs": [], "tvb": [], "tvp": []}

    for line in all_lines:
        for key, pattern in patterns.items():
            mm = re.search(pattern, line)
            if not mm:
                continue

            matches[key].append(mm)
            lines[key].append(line)
            pp = congress_bulk_path / mm.string[mm.span()[0] : mm.span()[1]]
            if pp.exists():
                paths[key].append(pp)

    return paths


TEXTVERSION_BILLS_PATTERN = re.compile(
    r"""BILLS-
    (?P<congress_num>\d{3})
    (?P<legis_type>[a-zA-Z]+)
    (?P<legis_num>\d+)
    (?P<legis_version>\w+)
    \.xml
    """,
    re.VERBOSE,
)

TEXTVERSION_PLAW_PATTERN = re.compile(
    r"""PLAW-
    (?P<congress_num>\d{3})
    (?P<legis_type>[a-zA-Z]+)
    (?P<legis_num>\d+)
    \.xml
    """,
    re.VERBOSE,
)

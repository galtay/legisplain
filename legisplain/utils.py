import re

BILLSTATUS_CONGRESS_NUMS = [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
TEXTVERSION_CONGRESS_NUMS = [113, 114, 115, 116, 117, 118]

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

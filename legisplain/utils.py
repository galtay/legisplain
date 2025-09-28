from pathlib import Path
import re

import pandas as pd
import rich
from langchain_core.documents import Document


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


def get_langchain_docs_from_unified(df: pd.DataFrame) -> list[Document]:
    skipped = []
    docs = []
    for _, u_row in df.iterrows():
        if len(u_row["tvs"]) == 0:
            skipped.append(u_row["legis_id"])
            continue
        doc = Document(
            page_content=u_row["tvs"][0]["tv_txt"],
            metadata={
                "tv_id": u_row["tvs"][0]["tv_id"],
                "legis_version": u_row["tvs"][0]["legis_version"],
                "legis_class": u_row["tvs"][0]["legis_class"],
                "legis_id": u_row["legis_id"],
                "congress_num": u_row["congress_num"],
                "legis_type": u_row["legis_type"],
                "legis_num": u_row["legis_num"],
                "text_date": u_row["tvs"][0]["bs_tv"]["date"],
                "title": u_row["bs_json"]["title"],
                "sponsor_bioguide_id": u_row["bs_json"]["sponsor"]["bioguide_id"],
                "sponsor_full_name": u_row["bs_json"]["sponsor"]["full_name"],
                "sponsor_party": u_row["bs_json"]["sponsor"]["party"],
                "sponsor_state": u_row["bs_json"]["sponsor"]["state"],
                "introduced_date": u_row["bs_json"]["introduced_date"],
                "policy_area": u_row["bs_json"]["policy_area"],
            },
        )
        docs.append(doc)
    if len(skipped) > 0:
        rich.print(f"skipped {len(skipped)} rows with no text versions")
    return docs

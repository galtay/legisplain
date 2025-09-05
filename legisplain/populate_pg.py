"""
"""

from collections import OrderedDict
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Union
from typing import Optional
import yaml

from bs4 import BeautifulSoup
import pandas as pd
import rich
from sqlalchemy import create_engine
from sqlalchemy import text

from legisplain import utils
from legisplain.billstatus_mod import BillStatus
from legisplain.textversion_mod import get_legis_text_v1


logger = logging.getLogger(__name__)


sql_drop_billstatus = """
DROP TABLE IF EXISTS billstatus
"""
sql_create_billstatus = """
CREATE TABLE IF NOT EXISTS billstatus (
  legis_id varchar PRIMARY KEY
  ,congress_num integer NOT NULL
  ,legis_type varchar NOT NULL
  ,legis_num integer NOT NULL
  ,bulk_path varchar NOT NULL
  ,lastmod timestamp without time zone NOT NULL
  ,bs_xml XML NOT NULL
  ,bs_json JSONB NOT NULL
)
"""

sql_add_index_billstatus = """
CREATE INDEX IF NOT EXISTS billstatus_bs_json_gin
  ON billstatus
  USING GIN (bs_json)
"""

sql_drop_textversion = """
DROP TABLE IF EXISTS textversion
"""
sql_create_textversion = """
CREATE TABLE IF NOT EXISTS textversion (
  tv_id varchar PRIMARY KEY
  ,legis_id varchar NOT NULL
  ,congress_num integer NOT NULL
  ,legis_type varchar NOT NULL
  ,legis_num integer NOT NULL
  ,legis_version varchar NOT NULL
  ,legis_class varchar NOT NULL
  ,bulk_path varchar NOT NULL
  ,file_name varchar NOT NULL
  ,lastmod timestamp without time zone NOT NULL
  ,xml_type varchar NOT NULL
  ,root_tag varchar NOT NULL
  ,tv_xml XML NOT NULL
  ,tv_txt varchar NOT NULL
)
"""

sql_drop_textversion_tag = """
DROP TABLE IF EXISTS textversion_tag
"""
sql_create_textversion_tag = """
CREATE TABLE IF NOT EXISTS textversion_tag (
  tvt_id varchar PRIMARY KEY
  ,tv_id varchar NOT NULL
  ,legis_id varchar NOT NULL
  ,legis_version varchar NOT NULL
  ,congress_num integer NOT NULL
  ,legis_type varchar NOT NULL
  ,file_name varchar NOT NULL
  ,root_name varchar NOT NULL
  ,root_attrs JSON NOT NULL
  ,child_indx integer NOT NULL
  ,child_name varchar NOT NULL
  ,child_xml XML NOT NULL
)
"""


sql_drop_unified = """
DROP TABLE IF EXISTS unified
"""


def create_tables(conn_str: str, echo=False):
    engine = create_engine(conn_str, echo=echo)
    with engine.connect() as conn:
        conn.execute(text(sql_create_billstatus))
        conn.execute(text(sql_create_textversion))
        conn.execute(text(sql_create_textversion_tag))
        conn.commit()

    with engine.connect() as conn:
        conn.execute(text(sql_add_index_billstatus))
        conn.commit()


def reset_tables(conn_str: str, echo=False):
    engine = create_engine(conn_str, echo=echo)
    with engine.connect() as conn:
        conn.execute(text(sql_drop_billstatus))
        conn.execute(text(sql_drop_textversion))
        conn.execute(text(sql_drop_textversion_tag))
        conn.execute(text(sql_drop_unified))
        conn.commit()
    create_tables(conn_str, echo=echo)

def _normalize_lastmod(value) -> Optional[str]:
    """
    Normalize the GovInfo-style lastmod strings (e.g., '2025-08-08T20:08:12.920Z')
    or DB datetime objects to a tz-naive, second-precision string:
        'YYYY-MM-DD HH:MM:SS'
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        # Drop tz and microseconds to consistent seconds precision.
        return value.replace(tzinfo=None, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    s = str(value).strip()
    # Try robust ISO8601 parsing first (handles Z and offsets).
    try:
        # Python 3.11+ has fromisoformat but doesn't accept 'Z'; handle 'Z' -> '+00:00'
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1] + "+00:00")
        else:
            dt = datetime.fromisoformat(s)
        # Convert to naive UTC seconds (since DB column is without TZ).
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        dt = dt.replace(microsecond=0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # Fallback: manual cleanup (T->space, strip trailing Z/offset, drop fraction)
        s = s.replace("T", " ").replace("Z", "")
        # Drop offset if present (e.g., '+00:00'/'-05:00')
        for tz_sep in ("+", "-"):
            # Keep first 19 chars 'YYYY-MM-DD HH:MM:SS' if offset exists
            if tz_sep in s and len(s) >= 19:
                s = s[:19]
                break
        # Drop fractional seconds
        if "." in s:
            s = s.split(".")[0]
        # Final sanity slice to seconds
        return s[:19]


def _load_known_lastmods(engine, table: str, key_col: str) -> dict[str, str]:
    """
    Load {key: normalized_lastmod_string} from the DB to avoid per-row queries.
    """
    known: dict[str, str] = {}
    with engine.connect() as conn:
        rows = conn.execute(text(f"select {key_col}, lastmod from {table}")).fetchall()
        for k, lm in rows:
            known[str(k)] = _normalize_lastmod(lm)
    return known

def _flush_billstatus_batch(engine, rows: list[dict]):
    """
    Batch UPSERT to billstatus. Only updates when lastmod changed.
    """
    cols = [
        "legis_id",
        "congress_num",
        "legis_type",
        "legis_num",
        "bulk_path",
        "lastmod",
        "bs_xml",
        "bs_json",
    ]
    placeholders = "(" + ", ".join(f":{c}" for c in cols) + ")"

    update_set = """
      congress_num = EXCLUDED.congress_num,
      legis_type   = EXCLUDED.legis_type,
      legis_num    = EXCLUDED.legis_num,
      bulk_path    = EXCLUDED.bulk_path,
      lastmod      = EXCLUDED.lastmod,
      bs_xml       = EXCLUDED.bs_xml,
      bs_json      = EXCLUDED.bs_json
    """

    sql = f"""
    INSERT INTO billstatus ({", ".join(cols)}) VALUES {placeholders}
    ON CONFLICT (legis_id) DO UPDATE SET
      {update_set}
    WHERE billstatus.lastmod IS DISTINCT FROM EXCLUDED.lastmod
    """

    with engine.connect() as conn:
        conn.execute(text(sql), rows)
        conn.commit()


def upsert_billstatus(
    congress_bulk_path: Union[str, Path],
    conn_str: str,
    batch_size: int = 50,
    echo: bool = False,
    paths: Optional[list[Path]] = None,
):
    """Upsert billstatus xml files into postgres

    Args:
        congress_bulk_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
        paths: if present use these instead of walking bulk path
    """

    data_path = Path(congress_bulk_path) / "data"
    engine = create_engine(conn_str, echo=echo)
    known_lastmods = _load_known_lastmods(engine, table="billstatus", key_col="legis_id")

    rows: list[OrderedDict] = []
    ibatch = 0
    scanned = 0
    skipped = 0
    prepared = 0

    if paths is None:
        logger.info("walking entire bulk data path")
        path_iter = data_path.rglob("fdsys_billstatus.xml")
    else:
        logger.info(f"walking {len(paths)} input paths")
        path_iter = paths

    for path_object in path_iter:
        scanned += 1
        path_str = str(path_object.relative_to(congress_bulk_path))
        match = re.match(utils.BILLSTATUS_PATH_PATTERN, path_str)
        if match is None:
            rich.print("billstatus oops: {}".format(path_object))
            continue

        congress_num = match.groupdict()["congress_num"]
        legis_type = match.groupdict()["legis_type"]
        legis_num = match.groupdict()["legis_num"]
        legis_id = f"{congress_num}-{legis_type}-{legis_num}"

        lastmod_path = path_object.parent / "fdsys_billstatus-lastmod.txt"
        if lastmod_path.exists():
            lastmod_raw = lastmod_path.read_text().strip()
            lastmod_norm = _normalize_lastmod(lastmod_raw)
            # Cheap skip if unchanged in DB
            if known_lastmods.get(legis_id) == lastmod_norm:
                skipped += 1
                continue
            # We'll store the normalized (tz-naive) string.
            lastmod_store = lastmod_norm
        else:
            rich.print("no lastmod file found oops: {}".format(lastmod_path))
            continue

        xml = path_object.read_text().strip()
        soup = BeautifulSoup(xml, "xml")
        xml_pretty = (
            soup.prettify()
        )  # note this also fixes invalid xml that bs leniently parsed
        bs = BillStatus.from_xml_str(xml)

        row = OrderedDict(
            {
                "legis_id": legis_id,
                "congress_num": int(congress_num),
                "legis_type": legis_type,
                "legis_num": int(legis_num),
                "bulk_path": path_str,
                "lastmod": lastmod_store,  # normalized, tz-naive seconds
                "bs_xml": xml_pretty,
                "bs_json": bs.model_dump_json(),
            }
        )
        rows.append(row)
        prepared += 1

        if len(rows) >= batch_size:
            rich.print(f"upserting billstatus batch {ibatch} with {len(rows)} rows.")
            _flush_billstatus_batch(engine, rows)
            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting billstatus batch {ibatch} with {len(rows)} rows.")
        _flush_billstatus_batch(engine, rows)


    logger.info(
        f"billstatus scan complete: scanned={scanned}, skipped_unchanged={skipped}, prepared={prepared}"
    )
    return {"scanned": scanned, "skipped": skipped, "prepared": prepared}




def upsert_textversion(
    congress_bulk_path: Union[str, Path],
    conn_str: str,
    batch_size: int = 50,
    echo: bool = False,
    paths: Optional[list[Path]] = None,
):
    """Upsert textversion xml files into postgres

    Args:
        congress_bulk_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
        paths: if present use these instead of walking bulk path
    """

    data_path = Path(congress_bulk_path) / "data"
    engine = create_engine(conn_str, echo=echo)
    missed = Counter()

    rows = []
    ibatch = 0

    if paths is None:
        logger.info("walking entire bulk data path")
        path_iter = data_path.rglob("*.xml")
    else:
        logger.info(f"walking {len(paths)} input paths")
        path_iter = paths

    for path_object in path_iter:
        path_str = str(path_object.relative_to(congress_bulk_path))

        if "/uslm/" in path_str:
            xml_type = "uslm"
        else:
            xml_type = "dtd"

        if match := re.match(utils.TEXTVERSION_BILLS_PATH_PATTERN, path_str):
            legis_class = "bills"
            legis_version = match.groupdict()["legis_version"]

        elif match := re.match(utils.TEXTVERSION_PLAW_PATH_PATTERN, path_str):
            legis_class = "plaw"
            legis_version = "plaw"

        else:
            missed[path_object.name] += 1
            continue

        lastmod_path = path_object.parent / (
            path_object.name.split(".")[0] + "-lastmod.txt"
        )
        lastmod_str = lastmod_path.read_text()

        xml = path_object.read_text().strip()
        soup = BeautifulSoup(xml, "xml")
        xml_pretty = (
            soup.prettify()
        )  # note this also fixes invalid xml that bs leniently parsed
        tv_txt = get_legis_text_v1(xml)

        root_tags = [el.name for el in soup.contents if el.name]
        if len(root_tags) != 1:
            rich.print("more than one non null root tag: ", root_tags)
        else:
            root_tag = root_tags[0]
            root_tag = root_tag.replace("{http://schemas.gpo.gov/xml/uslm}", "")

        if root_tag not in (
            "bill",
            "resolution",
            "amendment-doc",
            "pLaw",
        ):
            print(f"root tag not recognized: {root_tag}")

        row = {
            "tv_id": "{}-{}-{}-{}-{}".format(
                match.groupdict()["congress_num"],
                match.groupdict()["legis_type"],
                match.groupdict()["legis_num"],
                legis_version,
                xml_type,
            ),
            "legis_id": "{}-{}-{}".format(
                match.groupdict()["congress_num"],
                match.groupdict()["legis_type"],
                match.groupdict()["legis_num"],
            ),
            "congress_num": int(match.groupdict()["congress_num"]),
            "legis_type": match.groupdict()["legis_type"],
            "legis_num": int(match.groupdict()["legis_num"]),
            "legis_version": legis_version,
            "legis_class": legis_class,
            "bulk_path": path_str,
            "file_name": Path(path_str).name,
            "lastmod": lastmod_str,
            "xml_type": xml_type,
            "root_tag": root_tag,
            "tv_xml": xml_pretty,
            "tv_txt": tv_txt,
        }
        rows.append(row)

        if len(rows) >= batch_size:
            rich.print(f"upserting textversion batch {ibatch} with {len(rows)} rows.")
            pt1 = "({})".format(", ".join(row.keys()))
            pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
            pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
            sql = f"""
            INSERT INTO textversion {pt1} VALUES {pt2}
            ON CONFLICT (tv_id) DO UPDATE SET
            {pt3}
            """
            with engine.connect() as conn:
                conn.execute(text(sql), rows)
                conn.commit()

            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting textversion batch {ibatch} with {len(rows)} rows.")
        pt1 = "({})".format(", ".join(row.keys()))
        pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
        pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
        sql = f"""
        INSERT INTO textversion {pt1} VALUES {pt2}
        ON CONFLICT (tv_id) DO UPDATE SET
        {pt3}
        """
        with engine.connect() as conn:
            conn.execute(text(sql), rows)
            conn.commit()

    return missed


def get_root_tag(soup):
    for child in soup.children:
        if child.name is not None:
            return child


def upsert_textversion_tag(
    congress_bulk_path: Union[str, Path],
    conn_str: str,
    batch_size: int = 50,
    echo: bool = False,
    paths: Optional[list[Path]] = None,
):
    """Upsert textversion xml tags into postgres

    Args:
        congress_bulk_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
        paths: if present use these instead of walking bulk path
    """

    data_path = Path(congress_bulk_path) / "data"
    engine = create_engine(conn_str, echo=echo)

    rows = []
    ibatch = 0

    if paths is None:
        logger.info("walking entire bulk data path")
        path_iter = data_path.rglob("*.xml")
    else:
        logger.info(f"walking {len(paths)} input paths")
        path_iter = paths

    for path_object in path_iter:
        path_str = str(path_object.relative_to(congress_bulk_path))

        if "/uslm/" in path_str:
            continue
        xml_type = "dtd"

        if match := re.match(utils.TEXTVERSION_BILLS_PATH_PATTERN, path_str):
            legis_class = "bills"
            legis_version = match.groupdict()["legis_version"]

        elif match := re.match(utils.TEXTVERSION_PLAW_PATH_PATTERN, path_str):
            legis_class = "plaw"
            legis_version = "plaw"

        else:
            continue

        xml = path_object.read_text().strip()
        soup = BeautifulSoup(xml, "xml")
        root_tag = get_root_tag(soup)
        root_name = root_tag.name
        root_name = root_name.replace("{http://schemas.gpo.gov/xml/uslm}", "")
        if root_name not in (
            "bill",
            "resolution",
            "amendment-doc",
            "pLaw",
        ):
            print(f"root name not recognized: {root_name}")

        xml_pretty = (
            soup.prettify()
        )  # note this also fixes invalid xml that bs leniently parsed
        tv_txt = get_legis_text_v1(xml)

        ii = 0
        for child in root_tag.children:
            if child.get_text().strip() == "":
                continue
            row = {
                "tvt_id": "{}-{}-{}-{}-{}-{}".format(
                    match.groupdict()["congress_num"],
                    match.groupdict()["legis_type"],
                    match.groupdict()["legis_num"],
                    legis_version,
                    ii,
                    xml_type,
                ),
                "tv_id": "{}-{}-{}-{}-{}".format(
                    match.groupdict()["congress_num"],
                    match.groupdict()["legis_type"],
                    match.groupdict()["legis_num"],
                    legis_version,
                    xml_type,
                ),
                "legis_id": "{}-{}-{}".format(
                    match.groupdict()["congress_num"],
                    match.groupdict()["legis_type"],
                    match.groupdict()["legis_num"],
                ),
                "legis_version": legis_version,
                "legis_type": match.groupdict()["legis_type"],
                "congress_num": int(match.groupdict()["congress_num"]),
                "file_name": Path(path_str).name,
                "root_name": root_name,
                "root_attrs": json.dumps(root_tag.attrs),
                "child_indx": ii,
                "child_name": child.name,
                "child_xml": child.prettify().strip(),
            }
            rows.append(row)
            ii += 1

        if len(rows) >= batch_size:
            rich.print(
                f"upserting textversion_tag batch {ibatch} with {len(rows)} rows."
            )
            pt1 = "({})".format(", ".join(row.keys()))
            pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
            pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
            sql = f"""
            INSERT INTO textversion_tag {pt1} VALUES {pt2}
            ON CONFLICT (tvt_id) DO UPDATE SET
            {pt3}
            """
            with engine.connect() as conn:
                conn.execute(text(sql), rows)
                conn.commit()

            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting textversion_tag batch {ibatch} with {len(rows)} rows.")
        pt1 = "({})".format(", ".join(row.keys()))
        pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
        pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
        sql = f"""
        INSERT INTO textversion_tag {pt1} VALUES {pt2}
        ON CONFLICT (tvt_id) DO UPDATE SET
        {pt3}
        """
        with engine.connect() as conn:
            conn.execute(text(sql), rows)
            conn.commit()


def create_unified(conn_str: str):
    """Join billstatus and textversion data.
    Note that this uses the dtd xml text version not the uslm xml versions.

    BS = billstatus
    TV = textversion

    billstatus xml files have an array of textversion info (not the actual text).
    each textversion xml file has one version of the text for a bill.

    Args:
        conn_str: postgres connection string
    """

    sql = """
    drop table if exists unified;
    create table unified as (

    with

    -- turn BS textversion array with N entries into N rows
    bs_tvs_v1 as (
      select
        legis_id,
        json_array_elements(bs_json->'text_versions') as bs_tv
      from billstatus
    ),

    -- pull file name from BS textversion for joining to TV
    bs_tvs_v2 as (
      select
        legis_id,
        bs_tv,
        bs_tv->'url' as url,
        split_part(bs_tv->>'url', '/', -1) as file_name
      from bs_tvs_v1
    ),

    -- join BS and TV textversions. keep only dtd xml text versions
    jnd_tvs as (
      select
        textversion.*,
        bs_tv
      from bs_tvs_v2
      join textversion
      on bs_tvs_v2.file_name = textversion.file_name
      where xml_type = 'dtd'
    ),

    -- group TV info by legis_id
    tvs as (
      select
        legis_id,
        json_agg(
          json_build_object(
            'tv_id', tv_id,
            'legis_id', legis_id,
            'congress_num', congress_num,
            'legis_type', legis_type,
            'legis_num', legis_num,
            'legis_version', legis_version,
            'legis_class', legis_class,
            'bulk_path', bulk_path,
            'file_name', file_name,
            'lastmod', lastmod,
            'xml_type', xml_type,
            'root_tag', root_tag,
            'tv_xml', tv_xml,
            'tv_txt', tv_txt,
            'bs_tv', bs_tv
          ) order by lastmod desc
        ) as tvs
      from jnd_tvs
      group by legis_id
    )

    -- join billstatus info with text versions
    select billstatus.*, tvs.tvs from billstatus join tvs
    on billstatus.legis_id = tvs.legis_id
    )
    """

    engine = create_engine(conn_str, echo=True)
    with engine.connect() as conn:
        with conn.begin():
            result = conn.execute(text(sql))

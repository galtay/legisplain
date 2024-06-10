"""
"""

from collections import OrderedDict
from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Union
import yaml

from bs4 import BeautifulSoup
import pandas as pd
import rich
from sqlalchemy import create_engine
from sqlalchemy import text

from legisplain import utils
from legisplain.bill_status_mod import BillStatus


logger = logging.getLogger(__name__)


sql_drop_billstatus = """
DROP TABLE IF EXISTS billstatus
"""
sql_create_billstatus = """
CREATE TABLE billstatus (
  legis_id varchar PRIMARY KEY
  ,congress_num integer NOT NULL
  ,legis_type varchar NOT NULL
  ,legis_num integer NOT NULL
  ,scrape_path varchar NOT NULL
  ,lastmod timestamp without time zone NOT NULL
  ,bs_xml XML NOT NULL
  ,bs_json JSON NOT NULL
)
"""

sql_drop_textversion_xml = """
DROP TABLE IF EXISTS textversion_xml
"""
sql_create_textversion_xml = """
CREATE TABLE textversion_xml (
  tv_id varchar PRIMARY KEY
  ,legis_id varchar NOT NULL
  ,congress_num integer NOT NULL
  ,legis_type varchar NOT NULL
  ,legis_num integer NOT NULL
  ,legis_version varchar NOT NULL
  ,legis_class varchar NOT NULL
  ,scrape_path varchar NOT NULL
  ,file_name varchar NOT NULL
  ,lastmod timestamp without time zone NOT NULL
  ,xml_type varchar NOT NULL
  ,root_tag varchar NOT NULL
  ,tv_xml XML NOT NULL
)
"""

sql_drop_unified_xml = """
DROP TABLE IF EXISTS unified_xml
"""


def reset_tables(conn_str: str, echo=False):
    engine = create_engine(conn_str, echo=echo)
    with engine.connect() as conn:
        conn.execute(text(sql_drop_billstatus))
        conn.execute(text(sql_create_billstatus))
        conn.execute(text(sql_drop_textversion_xml))
        conn.execute(text(sql_create_textversion_xml))
        conn.execute(text(sql_drop_unified_xml))
        conn.commit()


def upsert_billstatus_xml(
    congress_bulk_path: Union[str, Path],
    conn_str: str,
    batch_size: int = 50,
    echo: bool = False,
):
    """Upsert billstatus xml files into postgres

    Args:
        congress_bulk_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
    """

    data_path = Path(congress_bulk_path) / "data"
    engine = create_engine(conn_str, echo=echo)

    rows = []
    ibatch = 0

    for path_object in data_path.rglob("fdsys_billstatus.xml"):
        path_str = str(path_object.relative_to(congress_bulk_path))
        if (match := re.match(utils.BILLSTATUS_PATH_PATTERN, path_str)) is None:
            rich.print("billstatus oops: {}".format(path_object))
            continue

        lastmod_path = path_object.parent / "fdsys_billstatus-lastmod.txt"
        lastmod_str = lastmod_path.read_text()
        xml = path_object.read_text().strip()
        soup = BeautifulSoup(xml, "xml")
        xml_pretty = (
            soup.prettify()
        )  # note this also fixes invalid xml that bs leniently parsed
        bs = BillStatus.from_xml_str(xml)

        row = OrderedDict(
            {
                "legis_id": "{}-{}-{}".format(
                    match.groupdict()["congress_num"],
                    match.groupdict()["legis_type"],
                    match.groupdict()["legis_num"],
                ),
                "congress_num": int(match.groupdict()["congress_num"]),
                "legis_type": match.groupdict()["legis_type"],
                "legis_num": int(match.groupdict()["legis_num"]),
                "scrape_path": path_str,
                "lastmod": lastmod_str,
                "bs_xml": xml_pretty,
                "bs_json": bs.model_dump_json(),
            }
        )
        rows.append(row)

        if len(rows) >= batch_size:
            rich.print(f"upserting billstatus batch {ibatch} with {len(rows)} rows.")
            pt1 = "({})".format(", ".join(row.keys()))
            pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
            pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
            sql = f"""
            INSERT INTO billstatus {pt1} VALUES {pt2}
            ON CONFLICT (legis_id) DO UPDATE SET
            {pt3}
            """
            with engine.connect() as conn:
                conn.execute(text(sql), rows)
                conn.commit()

            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting billstatus batch {ibatch} with {len(rows)} rows.")
        pt1 = "({})".format(", ".join(row.keys()))
        pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
        sql = f"""
        INSERT INTO billstatus {pt1} VALUES {pt2}
        """
        with engine.connect() as conn:
            conn.execute(text(sql), rows)
            conn.commit()


def upsert_textversion_xml(
    congress_bulk_path: Union[str, Path],
    conn_str: str,
    batch_size: int = 50,
    echo: bool = False,
):
    """Upsert textversion xml files into postgres

    Args:
        congress_bulk_path: should have "cache" and "data" as subdirectories
        conn_str: postgres connection string
        batch_size: number of billstatus files to upsert at once
    """

    data_path = Path(congress_bulk_path) / "data"
    engine = create_engine(conn_str, echo=echo)
    missed = Counter()

    rows = []
    ibatch = 0

    for path_object in data_path.rglob("*.xml"):
        path_str = str(path_object.relative_to(congress_bulk_path))

        if "/uslm/" in path_str:
            xml_type = "uslm"
        else:
            xml_type = "dtd"

        if match := re.match(utils.TEXTVERSION_BILLS_PATTERN, path_object.name):
            legis_class = "bills"
            legis_version = match.groupdict()["legis_version"]

        elif match := re.match(utils.TEXTVERSION_PLAW_PATTERN, path_object.name):
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
            "parse_failed",
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
            "scrape_path": path_str,
            "file_name": Path(path_str).name,
            "lastmod": lastmod_str,
            "xml_type": xml_type,
            "root_tag": root_tag,
            "tv_xml": xml_pretty,
        }
        rows.append(row)

        if len(rows) >= batch_size:
            rich.print(
                f"upserting textversion_xml batch {ibatch} with {len(rows)} rows."
            )
            pt1 = "({})".format(", ".join(row.keys()))
            pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
            pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
            sql = f"""
            INSERT INTO textversion_xml {pt1} VALUES {pt2}
            ON CONFLICT (tv_id) DO UPDATE SET
            {pt3}
            """
            with engine.connect() as conn:
                conn.execute(text(sql), rows)
                conn.commit()

            rows = []
            ibatch += 1

    if len(rows) > 0:
        rich.print(f"upserting textversion_xml batch {ibatch} with {len(rows)} rows.")
        pt1 = "({})".format(", ".join(row.keys()))
        pt2 = "({})".format(", ".join([f":{key}" for key in row.keys()]))
        pt3 = ", ".join(f"{key} = EXCLUDED.{key}" for key in row.keys())
        sql = f"""
        INSERT INTO textversion_xml {pt1} VALUES {pt2}
        ON CONFLICT (tv_id) DO UPDATE SET
        {pt3}
        """
        with engine.connect() as conn:
            conn.execute(text(sql), rows)
            conn.commit()

    return missed


def create_unified_xml(conn_str: str):
    """Join billstatus and textversion_xml data.
    Note that this uses the dtd xml text version not the uslm xml versions.

    BS = billstatus
    TV = textversion

    billstatus xml files have an array of textversion info (not the actual text).
    each textversion xml file has one version of the text for a bill.

    Args:
        conn_str: postgres connection string
    """

    sql = """
    drop table if exists unified_xml;
    create table unified_xml as (

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
        textversion_xml.*,
        bs_tv
      from bs_tvs_v2
      join textversion_xml
      on bs_tvs_v2.file_name = textversion_xml.file_name
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
            'scrape_path', scrape_path,
            'file_name', file_name,
            'lastmod', lastmod,
            'xml_type', xml_type,
            'root_tag', root_tag,
            'tv_xml', tv_xml,
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

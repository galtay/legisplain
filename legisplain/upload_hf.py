"""
Upload local data to HF
"""

from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import rich
from sqlalchemy import create_engine


def upload_billstatus(congress_hf_path: Union[str, Path], conn_str: str):

    rich.print(f"{congress_hf_path=}")

    ds_tag = "billstatus"
    ds_name = f"usc-{ds_tag}"

    upload_folder = Path(congress_hf_path) / ds_name
    data_folder = upload_folder / "data"
    data_folder.mkdir(exist_ok=True, parents=True)
    repo_id = f"hyperdemocracy/{ds_name}"
    rich.print(f"{repo_id=}")

    tmpl_readme_path = Path("templates") / "hf" / ds_name / "README.md"
    readme_path = upload_folder / "README.md"
    with tmpl_readme_path.open("r") as fp:
        readme_txt = fp.read()
    with readme_path.open("w") as fp:
        fp.write(readme_txt)

    engine = create_engine(conn_str, echo=False)
    df = pd.read_sql(
        f"select * from billstatus order by congress_num, legis_type, legis_num",
        con=engine,
    )
    df["lastmod"] = df["lastmod"].astype(str)
    for key in df.iloc[0]["bs_json"].keys():
        df[key] = df["bs_json"].apply(lambda x: x[key])

    table = pa.Table.from_pandas(df)

    for cn in df["congress_num"].unique():
        tf = table.filter((df["congress_num"] == cn).values)
        out_path = data_folder / f"usc-{cn}-{ds_tag}.parquet"
        rich.print(f"{out_path=}")
        pq.write_table(tf, out_path)

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    rich.print(f"{upload_folder=}")
    api.upload_folder(
        folder_path=upload_folder,
        repo_id=repo_id,
        repo_type="dataset",
    )


def upload_textversion(congress_hf_path: Union[str, Path], conn_str: str):

    rich.print(f"{congress_hf_path=}")

    ds_tag = "textversion"
    ds_name = f"usc-{ds_tag}"

    upload_folder = Path(congress_hf_path) / ds_name
    data_folder = upload_folder / "data"
    data_folder.mkdir(exist_ok=True, parents=True)
    repo_id = f"hyperdemocracy/{ds_name}"
    rich.print(f"{repo_id=}")

    tmpl_readme_path = Path("templates") / "hf" / ds_name / "README.md"
    readme_path = upload_folder / "README.md"
    with tmpl_readme_path.open("r") as fp:
        readme_txt = fp.read()
    with readme_path.open("w") as fp:
        fp.write(readme_txt)

    engine = create_engine(conn_str, echo=False)
    df = pd.read_sql(
        f"""select * from textversion
        where xml_type = 'dtd'
        order by congress_num, legis_type, legis_num, legis_version
        """,
        con=engine,
    )
    df["lastmod"] = df["lastmod"].astype(str)
    table = pa.Table.from_pandas(df)

    for cn in df["congress_num"].unique():
        tf = table.filter((df["congress_num"] == cn).values)
        out_path = data_folder / f"usc-{cn}-{ds_tag}.parquet"
        rich.print(f"{out_path=}")
        pq.write_table(tf, out_path)

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    rich.print(f"{upload_folder=}")
    api.upload_folder(
        folder_path=upload_folder,
        repo_id=repo_id,
        repo_type="dataset",
    )


def upload_unified(congress_hf_path: Union[str, Path], conn_str: str):

    rich.print(f"{congress_hf_path=}")

    ds_tag = "unified"
    ds_name = f"usc-{ds_tag}"

    upload_folder = Path(congress_hf_path) / ds_name
    data_folder = upload_folder / "data"
    data_folder.mkdir(exist_ok=True, parents=True)
    repo_id = f"hyperdemocracy/{ds_name}"
    rich.print(f"{repo_id=}")

    tmpl_readme_path = Path("templates") / "hf" / ds_name / "README.md"
    readme_path = upload_folder / "README.md"
    with tmpl_readme_path.open("r") as fp:
        readme_txt = fp.read()
    with readme_path.open("w") as fp:
        fp.write(readme_txt)

    engine = create_engine(conn_str, echo=False)
    df = pd.read_sql(
        f"select * from unified order by congress_num, legis_type, legis_num",
        con=engine,
    )
    df["lastmod"] = df["lastmod"].astype(str)
    table = pa.Table.from_pandas(df)

    for cn in df["congress_num"].unique():
        tf = table.filter((df["congress_num"] == cn).values)
        out_path = data_folder / f"usc-{cn}-{ds_tag}.parquet"
        rich.print(f"{out_path=}")
        pq.write_table(tf, out_path)

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    rich.print(f"{upload_folder=}")
    api.upload_folder(
        folder_path=upload_folder,
        repo_id=repo_id,
        repo_type="dataset",
    )

from enum import Enum
import logging
from pathlib import Path
import subprocess
from typing import Annotated
import yaml

import rich
from rich.logging import RichHandler
import typer

from legisplain import utils
from legisplain import populate_pg
from legisplain import upload_hf


logger = logging.getLogger(__name__)


app = typer.Typer()


class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"

LOG_LEVEL_ANNOTATED = Annotated[
    LogLevel,
    typer.Option(help="Log level"),
]



def execute_shell_cmnd(cmnd_parts):
    logger.info(f"{cmnd_parts=}")
    process = subprocess.Popen(
        cmnd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode(), stderr.decode()
    logger.info(f"stdout=\n{stdout}")
    logger.info(f"stderr=\n{stderr}")

    return stdout, stderr



@app.command()
def write_scripts(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
):

    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])

    # read main config
    #--------------------------------------------------
    with Path("config.yml").open("r") as fp:
        config = yaml.safe_load(fp)
    logger.info(config)


    # set base path and create log dir
    #--------------------------------------------------
    base_path = Path("scripts") / "bulk_download"
    logs_path = base_path / "logs"
    logs_path.mkdir(exist_ok=True)


    # write usc-run config
    #--------------------------------------------------
    usc_config = {
        "output": {
            "data": config["bulk_path"] + "/data",
            "cache": config["bulk_path"] + "/cache",
        }
    }
    with (base_path / "config.yml").open("w") as fp:
        yaml.dump(usc_config, fp)
    logger.info(usc_config)


    # write usc-run download commands
    #--------------------------------------------------
    download_lines = []
    for cn in utils.BILLSTATUS_CONGRESS_NUMS:
        download_lines.append(f"usc-run govinfo --bulkdata=BILLSTATUS --congress={cn}")

    for cn in utils.TEXTVERSION_CONGRESS_NUMS:
        download_lines.append(f"usc-run govinfo --bulkdata=BILLS --congress={cn}")
        download_lines.append(f"usc-run govinfo --bulkdata=PLAW --congress={cn}")

    with (base_path / "download.sh").open("w") as fp:
        for line in download_lines:
            fp.write(f"{line}\n")


    # write s3 sync commands
    #--------------------------------------------------
    sync_lines = []
    for suffix in ["data", "cache"]:
        sync_lines.append("aws s3 sync {} s3://{}/congress-bulk/{} --quiet".format(
            config["bulk_path"] + "/" + suffix,
            config["s3_bucket"],
            suffix,
        ))

    with (base_path / "sync.sh").open("w") as fp:
        for line in sync_lines:
            fp.write(f"{line}\n")



@app.command()
def pg_reset_tables(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    with Path("config.yml").open("r") as fp:
        config = yaml.safe_load(fp)
    logger.info(config)
    conn_str = config["pg_conn_str"]
    congress_bulk_path = config["bulk_path"]
    populate_pg.reset_tables(conn_str, echo=True)


@app.command()
def pg_populate_billstatus(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    batch_size: int=100,
    echo: bool=False,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    with Path("config.yml").open("r") as fp:
        config = yaml.safe_load(fp)
    logger.info(config)
    conn_str = config["pg_conn_str"]
    congress_bulk_path = config["bulk_path"]
    populate_pg.upsert_billstatus_xml(
        congress_bulk_path,
        conn_str,
        batch_size=batch_size,
        echo=echo,
    )

@app.command()
def pg_populate_textversion_xml(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    batch_size: int=100,
    echo: bool=False,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    with Path("config.yml").open("r") as fp:
        config = yaml.safe_load(fp)
    logger.info(config)
    conn_str = config["pg_conn_str"]
    congress_bulk_path = config["bulk_path"]
    populate_pg.upsert_textversion_xml(
        congress_bulk_path,
        conn_str,
        batch_size=batch_size,
        echo=echo,
    )

@app.command()
def pg_populate_unified_xml(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    batch_size: int=100,
    echo: bool=False,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    with Path("config.yml").open("r") as fp:
        config = yaml.safe_load(fp)
    logger.info(config)
    conn_str = config["pg_conn_str"]
    congress_bulk_path = config["bulk_path"]
    populate_pg.create_unified_xml(conn_str)


@app.command()
def hf_upload_billstatus(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    echo: bool=False,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    with Path("config.yml").open("r") as fp:
        config = yaml.safe_load(fp)
    logger.info(config)
    conn_str = config["pg_conn_str"]
    congress_hf_path = config["hf_path"]
    upload_hf.upload_billstatus(
        congress_hf_path,
        conn_str,
    )


if __name__ == "__main__":
    app()

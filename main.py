from enum import Enum
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Annotated
from typing import Optional
import yaml

import rich
from rich.logging import RichHandler
import typer

from legisplain import utils
from legisplain import populate_pg
from legisplain import upload_hf
from legisplain import chunking_v1
from legisplain import embedding_v1
from legisplain import chroma
from legisplain.config import Config, CANONICAL_CHUNK_PARAMS


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


def load_config(config_path: Path = Path("config.yml")) -> Config:
    """Load configuration and return Config instance."""
    if config_path.exists():
        return Config.from_yaml_file(config_path)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return Config()


def create_usc_config(config: Config) -> dict:
    """Create USC configuration for the usc-run tool."""
    congress_bulk_path = config.get_congress_bulk_path()
    return {
        "output": {
            "data": str(congress_bulk_path / "data"),
            "cache": str(congress_bulk_path / "cache"),
        }
    }


def require_pg_connection(config: Config) -> str:
    """Check that PostgreSQL connection string is configured and return it."""
    if config.pg_conn_str is None:
        logger.error("PostgreSQL connection string (pg_conn_str) is required for this command")
        logger.info("Please add 'pg_conn_str' to your config.yml file")
        raise typer.Exit(1)
    return config.pg_conn_str


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
def usc_download(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    congress_nums: Optional[str] = typer.Option(None, help="Comma-separated congress numbers (e.g., '117,118,119'). If not provided, uses default lists."),
    billstatus_only: bool = typer.Option(False, help="Download only BILLSTATUS data"),
    textversion_only: bool = typer.Option(False, help="Download only BILLS and PLAW data"),
    quiet: bool = typer.Option(False, help="Suppress real-time output from usc-run (capture and log only on completion/error)"),
):
    """Download congressional data using usc-run."""
    
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    
    # Load config
    config = load_config()
    logger.info(f"Using base path: {config.hyperdemocracy_data_path}")
    
    # Create directories
    congress_bulk_path = config.get_congress_bulk_path()
    congress_bulk_path.mkdir(parents=True, exist_ok=True)
    
    # Create USC config directory in the repo (not in user data directory)
    usc_config_dir = Path("usc-config")
    usc_config_dir.mkdir(exist_ok=True)
    
    # Create USC config file in the repo's usc-config directory
    # usc-run expects config.yml in the current working directory
    usc_config = create_usc_config(config)
    usc_config_path = usc_config_dir / "config.yml"
    with usc_config_path.open("w") as fp:
        yaml.dump(usc_config, fp)
    logger.info(f"Created USC config at: {usc_config_path}")
    logger.info(f"Data will be written to: {congress_bulk_path}")
    
    # Determine congress numbers to download
    if congress_nums:
        congress_nums_list = [int(x.strip()) for x in congress_nums.split(",")]
        billstatus_nums = congress_nums_list
        textversion_nums = congress_nums_list
    else:
        billstatus_nums = utils.BILLSTATUS_CONGRESS_NUMS
        textversion_nums = utils.TEXTVERSION_CONGRESS_NUMS
    
    # Build download commands (usc-run will find config.yml in the working directory)
    download_commands = []
    
    if not textversion_only:
        for cn in billstatus_nums:
            download_commands.append([
                "uv", "run", "usc-run", 
                "govinfo", f"--bulkdata=BILLSTATUS", f"--congress={cn}"
            ])
    
    if not billstatus_only:
        for cn in textversion_nums:
            download_commands.extend([
                [
                    "uv", "run", "usc-run", 
                    "govinfo", f"--bulkdata=BILLS", f"--congress={cn}"
                ],
                [
                    "uv", "run", "usc-run", 
                    "govinfo", f"--bulkdata=PLAW", f"--congress={cn}"
                ]
            ])
    
    # Execute download commands with usc_config_dir as working directory
    total_commands = len(download_commands)
    logger.info(f"Executing {total_commands} download commands...")
    logger.info(f"Working directory: {usc_config_dir.absolute()}")
    
    for i, cmd in enumerate(download_commands, 1):
        logger.info(f"[{i}/{total_commands}] Running: {' '.join(cmd)}")
        
        try:
            if quiet:
                # Capture output and only show on completion/error
                result = subprocess.run(
                    cmd,
                    cwd=usc_config_dir,  # usc-run will find config.yml here
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"✓ Command completed successfully")
                if result.stdout.strip():
                    logger.debug(f"stdout: {result.stdout.strip()}")
            else:
                # Run with real-time output (don't capture stdout/stderr)
                result = subprocess.run(
                    cmd,
                    cwd=usc_config_dir,  # usc-run will find config.yml here
                    check=True
                )
                logger.info(f"✓ Command completed successfully")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Command failed with exit code {e.returncode}")
            if quiet and hasattr(e, 'stderr') and e.stderr:
                logger.error(f"stderr: {e.stderr}")
            if quiet and hasattr(e, 'stdout') and e.stdout and e.stdout.strip():
                logger.error(f"stdout: {e.stdout}")
            # Continue with next command rather than stopping
    
    logger.info("Download process completed!")


@app.command()
def usc_sync(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    data_only: bool = typer.Option(False, help="Sync only the data directory"),
    cache_only: bool = typer.Option(False, help="Sync only the cache directory"),
    dry_run: bool = typer.Option(False, help="Show what would be synced without actually syncing"),
    quiet: bool = typer.Option(False, help="Suppress real-time output from aws s3 sync (capture and log only on completion/error)"),
):
    """Sync congressional bulk data to AWS S3."""
    
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    
    # Load config
    config = load_config()
    logger.info(f"Using base path: {config.hyperdemocracy_data_path}")
    logger.info(f"S3 bucket: {config.s3_bucket}")
    
    congress_bulk_path = config.get_congress_bulk_path()
    
    # Check if bulk data directory exists
    if not congress_bulk_path.exists():
        logger.error(f"Congress bulk directory does not exist: {congress_bulk_path}")
        logger.info("Run 'download' command first to create the data.")
        raise typer.Exit(1)
    
    # Build sync commands
    sync_commands = []
    
    if not cache_only:
        data_dir = congress_bulk_path / "data"
        if data_dir.exists():
            cmd = [
                "aws", "s3", "sync", 
                str(data_dir),
                f"s3://{config.s3_bucket}/congress-bulk/data"
            ]
            if dry_run:
                cmd.append("--dryrun")
            sync_commands.append(("data", cmd))
        else:
            logger.warning(f"Data directory does not exist: {data_dir}")
    
    if not data_only:
        cache_dir = congress_bulk_path / "cache"
        if cache_dir.exists():
            cmd = [
                "aws", "s3", "sync",
                str(cache_dir), 
                f"s3://{config.s3_bucket}/congress-bulk/cache"
            ]
            if dry_run:
                cmd.append("--dryrun")
            sync_commands.append(("cache", cmd))
        else:
            logger.warning(f"Cache directory does not exist: {cache_dir}")
    
    if not sync_commands:
        logger.error("No directories to sync!")
        raise typer.Exit(1)
    
    # Execute sync commands
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Syncing {len(sync_commands)} directories to S3...")
    
    for dir_name, cmd in sync_commands:
        logger.info(f"Syncing {dir_name}: {' '.join(cmd)}")
        
        try:
            if quiet:
                # Capture output and only show on completion/error
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"✓ {dir_name} sync completed successfully")
                if result.stdout.strip():
                    # S3 sync can produce a lot of output, so we'll just show a summary
                    lines = result.stdout.strip().split('\n')
                    logger.debug(f"Synced {len(lines)} items")
                    logger.debug(f"Full output: {result.stdout}")
            else:
                # Run with real-time output (don't capture stdout/stderr)
                result = subprocess.run(
                    cmd,
                    check=True
                )
                logger.info(f"✓ {dir_name} sync completed successfully")
                        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {dir_name} sync failed with exit code {e.returncode}")
            if quiet and hasattr(e, 'stderr') and e.stderr:
                logger.error(f"stderr: {e.stderr}")
            if quiet and hasattr(e, 'stdout') and e.stdout and e.stdout.strip():
                logger.error(f"stdout: {e.stdout}")
            # Continue with next command rather than stopping
    
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Sync process completed!")





# Main workflow commands (in typical processing order)
# ====================================================

@app.command()
def pg_populate_billstatus(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    batch_size: int = 100,
    echo: bool = False,
    log_path: Optional[Path] = None,
):
    """Populate billstatus table in postgres."""
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info("pg_populate_billstatus")
    logger.info(config)
    conn_str = require_pg_connection(config)
    congress_bulk_path = config.get_congress_bulk_path()

    if log_path is not None:
        paths = utils.get_paths_from_download_logs(congress_bulk_path, log_path)
        bs_paths = paths["bs"]
    else:
        bs_paths = None

    populate_pg.upsert_billstatus(
        congress_bulk_path,
        conn_str,
        batch_size=batch_size,
        echo=echo,
        paths=bs_paths,
    )


@app.command()
def pg_populate_textversion(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    batch_size: int = 100,
    echo: bool = False,
    log_path: Optional[Path] = None,
):
    """Populate textversion table in postgres."""
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info("pg_populate_textversion")
    logger.info(config)
    conn_str = require_pg_connection(config)
    congress_bulk_path = config.get_congress_bulk_path()

    if log_path is not None:
        paths = utils.get_paths_from_download_logs(congress_bulk_path, log_path)
        tv_paths = paths["tvb"] + paths["tvp"]
    else:
        tv_paths = None

    populate_pg.upsert_textversion(
        congress_bulk_path,
        conn_str,
        batch_size=batch_size,
        echo=echo,
        paths=tv_paths,
    )


@app.command()
def pg_populate_unified(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    echo: bool = False,
):
    """Populate unified table in postgres."""
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info("pg_populate_unified")
    logger.info(config)
    conn_str = require_pg_connection(config)
    populate_pg.create_unified(conn_str, echo=echo)


@app.command()
def hf_upload_billstatus(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    echo: bool = False,
):
    """Upload billstatus table to HuggingFace."""
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info("hf_upload_billstatus")
    logger.info(config)
    conn_str = require_pg_connection(config)
    congress_hf_path = str(config.get_congress_hf_path())
    upload_hf.upload_billstatus(
        congress_hf_path,
        conn_str,
    )


@app.command()
def hf_upload_textversion(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    echo: bool = False,
):
    """Upload textversion table to HuggingFace."""
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info("hf_upload_textversion")
    logger.info(config)
    conn_str = require_pg_connection(config)
    congress_hf_path = str(config.get_congress_hf_path())
    upload_hf.upload_textversion(
        congress_hf_path,
        conn_str,
    )


@app.command()
def hf_upload_unified(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    echo: bool = False,
):
    """Upload unified table to HuggingFace."""
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info("hf_upload_unified")
    logger.info(config)
    conn_str = require_pg_connection(config)
    congress_hf_path = str(config.get_congress_hf_path())
    upload_hf.upload_unified(
        congress_hf_path,
        conn_str,
    )


@app.command()
def chunk_v1(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    chunk_size: Optional[int] = typer.Option(None, help="Chunk size in characters. If not provided, processes all canonical chunk sizes."),
    chunk_overlap: Optional[int] = typer.Option(None, help="Chunk overlap in characters. Required if chunk_size is specified."),
    upload: bool = typer.Option(True, help="Upload chunked datasets to HuggingFace"),
):
    """Create chunked datasets from unified congressional data."""
    
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    
    # Load base config
    config = load_config()
    logger.info(f"Using base path: {config.hyperdemocracy_data_path}")
    
    # Determine chunk parameters to process
    if chunk_size is not None:
        if chunk_overlap is None:
            logger.error("chunk_overlap is required when chunk_size is specified")
            raise typer.Exit(1)
        chunk_params = [(chunk_size, chunk_overlap)]
        logger.info(f"Processing single chunk configuration: size={chunk_size}, overlap={chunk_overlap}")
    else:
        chunk_params = CANONICAL_CHUNK_PARAMS
        logger.info(f"Processing all canonical chunk configurations: {chunk_params}")
    
    # Process each chunk configuration
    total_configs = len(chunk_params)
    for i, (cs, co) in enumerate(chunk_params, 1):
        logger.info(f"[{i}/{total_configs}] Processing chunk_size={cs}, chunk_overlap={co}")
        
        # Create config for this chunk configuration
        chunk_config = Config(
            hyperdemocracy_data_path=config.hyperdemocracy_data_path,
            pg_conn_str=config.pg_conn_str,
            s3_bucket=config.s3_bucket,
            congress_nums=config.congress_nums,
            chunk_size=cs,
            chunk_overlap=co,
            chunking_version=config.chunking_version,
            embedding_model_name=config.embedding_model_name,
            embedding_version=config.embedding_version,
        )
        
        try:
            # Create chunked datasets
            logger.info("Creating chunked datasets...")
            chunking_v1.write_local(chunk_config)
            logger.info("✓ Chunked datasets created successfully")
            
            # Upload to HuggingFace if requested
            if upload:
                logger.info("Uploading to HuggingFace...")
                chunking_v1.upload_dataset(chunk_config)
                logger.info("✓ Upload completed successfully")
            else:
                logger.info("Skipping HuggingFace upload (--no-upload specified)")
                
        except Exception as e:
            logger.error(f"✗ Failed to process chunk configuration {cs}/{co}: {e}")
            # Continue with next configuration rather than stopping
    
    logger.info("Chunking process completed!")


@app.command()
def embed_v1(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    chunk_size: Optional[int] = typer.Option(None, help="Chunk size in characters. If not provided, processes all canonical chunk sizes."),
    chunk_overlap: Optional[int] = typer.Option(None, help="Chunk overlap in characters. Required if chunk_size is specified."),
    embedding_model: Optional[str] = typer.Option(None, help="Embedding model name. If not provided, uses default from config."),
    write_local: bool = typer.Option(True, help="Generate embeddings locally"),
    write_readme: bool = typer.Option(True, help="Write README file for the dataset"),
    upload: bool = typer.Option(True, help="Upload embedding datasets to HuggingFace"),
):
    """Create embedding datasets from chunked congressional data."""
    
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    
    # Load base config
    config = load_config()
    logger.info(f"Using base path: {config.hyperdemocracy_data_path}")
    
    # Determine chunk parameters to process
    if chunk_size is not None:
        if chunk_overlap is None:
            logger.error("chunk_overlap is required when chunk_size is specified")
            raise typer.Exit(1)
        chunk_params = [(chunk_size, chunk_overlap)]
        logger.info(f"Processing single chunk configuration: size={chunk_size}, overlap={chunk_overlap}")
    else:
        chunk_params = CANONICAL_CHUNK_PARAMS
        logger.info(f"Processing all canonical chunk configurations: {chunk_params}")
    
    # Determine embedding model
    model_name = embedding_model if embedding_model is not None else config.embedding_model_name
    logger.info(f"Using embedding model: {model_name}")
    
    # Process each chunk configuration
    total_configs = len(chunk_params)
    for i, (cs, co) in enumerate(chunk_params, 1):
        logger.info(f"[{i}/{total_configs}] Processing chunk_size={cs}, chunk_overlap={co}")
        
        # Create config for this embedding configuration
        embed_config = Config(
            hyperdemocracy_data_path=config.hyperdemocracy_data_path,
            pg_conn_str=config.pg_conn_str,
            s3_bucket=config.s3_bucket,
            congress_nums=config.congress_nums,
            chunk_size=cs,
            chunk_overlap=co,
            chunking_version=config.chunking_version,
            embedding_model_name=model_name,
            embedding_version=config.embedding_version,
        )
        
        try:
            # Write README if requested
            if write_readme:
                logger.info("Writing README file...")
                embedding_v1.write_readme(embed_config)
                logger.info("✓ README file created successfully")
            
            # Generate embeddings if requested
            if write_local:
                logger.info("Generating embeddings...")
                embedding_v1.write_local(embed_config)
                logger.info("✓ Embeddings generated successfully")
            else:
                logger.info("Skipping local embedding generation (--no-write-local specified)")
            
            # Upload to HuggingFace if requested
            if upload:
                logger.info("Uploading to HuggingFace...")
                embedding_v1.upload_hf(embed_config)
                logger.info("✓ Upload completed successfully")
            else:
                logger.info("Skipping HuggingFace upload (--no-upload specified)")
                
        except Exception as e:
            logger.error(f"✗ Failed to process embedding configuration {cs}/{co}: {e}")
            # Continue with next configuration rather than stopping
    
    logger.info("Embedding process completed!")


@app.command()
def chroma_populate(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
    chunk_size: Optional[int] = typer.Option(None, help="Chunk size in characters. If not provided, processes all canonical chunk sizes."),
    chunk_overlap: Optional[int] = typer.Option(None, help="Chunk overlap in characters. Required if chunk_size is specified."),
    embedding_model: Optional[str] = typer.Option(None, help="Embedding model name. If not provided, uses default from config."),
    upload: bool = typer.Option(True, help="Upload ChromaDB datasets to HuggingFace"),
    reset: bool = typer.Option(True, help="Reset/delete existing ChromaDB before creating new one"),
):
    """Create ChromaDB from embedding datasets."""
    
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    
    # Load base config
    config = load_config()
    logger.info(f"Using base path: {config.hyperdemocracy_data_path}")
    
    # Determine chunk parameters to process
    if chunk_size is not None:
        if chunk_overlap is None:
            logger.error("chunk_overlap is required when chunk_size is specified")
            raise typer.Exit(1)
        chunk_params = [(chunk_size, chunk_overlap)]
        logger.info(f"Processing single chunk configuration: size={chunk_size}, overlap={chunk_overlap}")
    else:
        chunk_params = CANONICAL_CHUNK_PARAMS
        logger.info(f"Processing all canonical chunk configurations: {chunk_params}")
    
    # Determine embedding model
    model_name = embedding_model if embedding_model is not None else config.embedding_model_name
    logger.info(f"Using embedding model: {model_name}")
    
    # Process each chunk configuration
    total_configs = len(chunk_params)
    for i, (cs, co) in enumerate(chunk_params, 1):
        logger.info(f"[{i}/{total_configs}] Processing chunk_size={cs}, chunk_overlap={co}")
        
        # Create config for this ChromaDB configuration
        chroma_config = Config(
            hyperdemocracy_data_path=config.hyperdemocracy_data_path,
            pg_conn_str=config.pg_conn_str,
            s3_bucket=config.s3_bucket,
            congress_nums=config.congress_nums,
            chunk_size=cs,
            chunk_overlap=co,
            chunking_version=config.chunking_version,
            embedding_model_name=model_name,
            embedding_version=config.embedding_version,
        )
        
        try:
            # Get vector paths for this configuration
            vec_paths = [chroma_config.get_vec_path(cn) for cn in chroma_config.congress_nums]
            
            # Check if vector files exist
            missing_files = [p for p in vec_paths if not p.exists()]
            if missing_files:
                logger.error(f"Missing embedding files for configuration {cs}/{co}:")
                for p in missing_files:
                    logger.error(f"  - {p}")
                logger.error("Run 'embed-v1' command first to generate embeddings")
                continue
            
            chroma_persist_directory = chroma_config.get_chroma_persist_directory()
            
            # Reset ChromaDB if requested
            if reset and chroma_persist_directory.exists():
                logger.info("Resetting existing ChromaDB...")
                shutil.rmtree(chroma_persist_directory)
                logger.info("✓ Existing ChromaDB deleted")
            
            # Create ChromaDB
            logger.info("Creating ChromaDB...")
            client = chroma.create_chroma_client(chroma_config)
            collection = chroma.load_dataset_to_chroma(client, vec_paths, collection_name="usc", n_lim=None)
            logger.info("✓ ChromaDB created successfully")
            
            # Upload to HuggingFace if requested
            if upload:
                logger.info("Uploading ChromaDB to HuggingFace...")
                success = chroma.upload_chroma_to_hf(chroma_persist_directory)
                if success:
                    logger.info("✓ Upload completed successfully")
                    dataset_name = chroma_config.get_chroma_dataset_name()
                    logger.info(f"Dataset URL: https://huggingface.co/datasets/hyperdemocracy/{dataset_name}")
                else:
                    logger.error("✗ Upload failed")
            else:
                logger.info("Skipping HuggingFace upload (--no-upload specified)")
                logger.info(f"ChromaDB created at: {chroma_persist_directory}")
                
        except Exception as e:
            logger.error(f"✗ Failed to process ChromaDB configuration {cs}/{co}: {e}")
            # Continue with next configuration rather than stopping
    
    logger.info("ChromaDB process completed!")


# Utility commands (one-off tools)
# =================================

@app.command()
def pg_reset_tables(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info(f"Using config: hyperdemocracy_data_path={config.hyperdemocracy_data_path}")
    conn_str = require_pg_connection(config)
    populate_pg.reset_tables(conn_str, echo=True)


@app.command()
def pg_create_tables(
    log_level: LOG_LEVEL_ANNOTATED = LogLevel.info,
):
    logging.basicConfig(level=log_level.value, handlers=[RichHandler()])
    config = load_config()
    logger.info(f"Using config: hyperdemocracy_data_path={config.hyperdemocracy_data_path}")
    conn_str = require_pg_connection(config)
    populate_pg.create_tables(conn_str, echo=True)


if __name__ == "__main__":
    app()

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
import yaml


CONGRESS_NUMS = [119, 118, 117, 116, 115, 114, 113]
CANONICAL_CHUNK_PARAMS = [(8192, 512), (4096, 512), (2048, 256)]
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/static-retrieval-mrl-en-v1"
DEFAULT_CHUNKING_VERSION = 1
DEFAULT_EMBEDDING_VERSION = 1


class Config(BaseModel):
    # Core paths
    hyperdemocracy_data_path: Path = Field(default_factory=lambda: Path.home() / "hyperdemocracy-data")
    
    # Database connection (optional)
    pg_conn_str: Optional[str] = None
    
    # S3 configuration
    s3_bucket: str = "hyperdemocracy"
    
    # Congress processing
    congress_nums: list[int] = CONGRESS_NUMS
    
    # Chunking configuration (for embedding workflows)
    chunk_size: int = CANONICAL_CHUNK_PARAMS[0][0]
    chunk_overlap: int = CANONICAL_CHUNK_PARAMS[0][1]
    chunking_version: int = DEFAULT_CHUNKING_VERSION
    
    # Embedding configuration (for embedding workflows)
    embedding_model_name: Optional[str] = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_version: int = DEFAULT_EMBEDDING_VERSION
    
    @classmethod
    def from_yaml_file(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with config_path.open("r") as fp:
            yaml_data = yaml.safe_load(fp) or {}
        return cls(**yaml_data)
    
    def get_congress_bulk_path(self) -> Path:
        """Get the congress-bulk directory path."""
        return self.hyperdemocracy_data_path / "congress-bulk"
    
    def get_congress_hf_path(self) -> Path:
        """Get the congress-hf directory path."""
        return self.hyperdemocracy_data_path / "congress-hf"

    def get_chunking_tag(self):
        return f"chunks-v{self.chunking_version}-s{self.chunk_size}-o{self.chunk_overlap}"

    def get_embedding_model_tag(self):
        return self.embedding_model_name.replace("/", "-")

    def get_vecs_tag(self):
        return f"vecs-v{self.embedding_version}-{self.get_chunking_tag()}-{self.get_embedding_model_tag()}"

    def get_unified_path(self, congress_num: int):
        return self.get_congress_hf_path() / "usc-unified" / "data" / f"usc-{congress_num}-unified.parquet"

    def get_chunk_path(self, congress_num: int):
        return self.get_congress_hf_path() / f"usc-{self.get_chunking_tag()}" / "data" / f"usc-{congress_num}-{self.get_chunking_tag()}.parquet"

    def get_vec_path(self, congress_num: int):
        return (
            self.get_congress_hf_path() 
            / f"usc-{self.get_vecs_tag()}" 
            / "data" 
            / f"usc-{congress_num}-{self.get_vecs_tag()}.parquet"
        )

    def get_chroma_persist_directory(self):
        return self.get_congress_hf_path() / f"usc-chroma-{self.get_vecs_tag()}" / "chromadb"
    
    def get_chroma_dataset_name(self):
        """Get the HF dataset name for ChromaDB following the naming convention"""
        return f"usc-chroma-{self.get_vecs_tag()}"
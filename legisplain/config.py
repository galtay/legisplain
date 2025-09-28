from pathlib import Path
from pydantic import BaseModel


CONGRESS_NUMS = [119, 118, 117, 116, 115, 114, 113]

class Config(BaseModel):
    congress_hf_path: Path
    congress_nums: list[int] = CONGRESS_NUMS
    chunk_size: int
    chunk_overlap: int
    chunking_version: int
    embedding_model_name: str | None = None
    embedding_version: int

    def get_chunking_tag(self):
        return f"chunks-v{self.chunking_version}-s{self.chunk_size}-o{self.chunk_overlap}"

    def get_embedding_model_tag(self):
        return self.embedding_model_name.replace("/", "-")

    def get_vecs_tag(self):
        return f"vecs-v{self.embedding_version}-{self.get_chunking_tag()}-{self.get_embedding_model_tag()}"

    def get_unified_path(self, congress_num: int):
        return self.congress_hf_path / "usc-unified" / "data" / f"usc-{congress_num}-unified.parquet"

    def get_chunk_path(self, congress_num: int):
        return self.congress_hf_path / f"usc-{self.get_chunking_tag()}" / "data" / f"usc-{congress_num}-{self.get_chunking_tag()}.parquet"

    def get_vec_path(self, congress_num: int):
        return (
            self.congress_hf_path 
            / f"usc-{self.get_vecs_tag()}" 
            / "data" 
            / f"usc-{congress_num}-{self.get_vecs_tag()}.parquet"
        )

    def get_chroma_persist_directory(self):
        return self.congress_hf_path / f"usc-chroma-{self.get_vecs_tag()}" / "chromadb"
    
    def get_chroma_dataset_name(self):
        """Get the HF dataset name for ChromaDB following the naming convention"""
        return f"usc-chroma-{self.get_vecs_tag()}"
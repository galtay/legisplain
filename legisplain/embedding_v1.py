import os
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi
import numpy as np
import pandas as pd
import rich
from sentence_transformers import SentenceTransformer
import yaml

from legisplain.config import Config
from legisplain import utils


os.environ["TOKENIZERS_PARALLELISM"] = "false"
VEC_DTYPE = "float32"
EMBEDDING_VERSION = 1


def get_readme_str(config: Config):

    model_tag = config.get_embedding_model_tag()
    yaml_dict = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": str(cn),
                        "path": f"data/usc-{cn}-{config.get_vecs_tag()}.parquet",
                    }
                    for cn in config.congress_nums
                ],
            }
        ],
        "dataset_info": {
            "features": [
                {"name": "chunk_id", "dtype": "string"},
                {"name": "tv_id", "dtype": "string"},
                {"name": "legis_id", "dtype": "string"},
                {"name": "text", "dtype": "string"},
                {"name": "vec", "list": {"dtype": VEC_DTYPE}},
                {
                    "name": "metadata",
                    "struct": [
                        {"name": "tv_id", "dtype": "string"},
                        {"name": "legis_version", "dtype": "string"},
                        {"name": "legis_class", "dtype": "string"},
                        {"name": "legis_id", "dtype": "string"},
                        {"name": "congress_num", "dtype": "int32"},
                        {"name": "legis_type", "dtype": "string"},
                        {"name": "legis_num", "dtype": "int32"},
                        {"name": "text_date", "dtype": "string"},
                        {"name": "title", "dtype": "string"},
                        {"name": "sponsor_bioguide_id", "dtype": "string"},
                        {"name": "sponsor_full_name", "dtype": "string"},
                        {"name": "sponsor_party", "dtype": "string"},
                        {"name": "sponsor_state", "dtype": "string"},
                        {"name": "introduced_date", "dtype": "string"},
                        {"name": "policy_area", "dtype": "string"},
                        {"name": "chunk_index", "dtype": "int32"},
                        {"name": "chunk_id", "dtype": "string"},
                        {"name": "start_index", "dtype": "int32"},
                    ],
                },
            ]
        },
    }
    readme_str = "---\n{}---".format(yaml.safe_dump(yaml_dict))
    return readme_str


def write_local(config: Config):

    rich.print("EMBEDDING (write local)")
    rich.print(config)

    model = get_embedding_model(config)
    model.load_model()

    for congress_num in config.congress_nums:

        c_fpath = config.get_chunk_path(congress_num)
        df_c = pd.read_parquet(c_fpath)
        vecs = model.encode_docs([el.strip() for el in df_c["text"].tolist()])

        df_c["vec"] = [row for row in vecs]
        df_c["chunk_id"] = df_c["metadata"].apply(lambda x: x["chunk_id"])
        df_c["tv_id"] = df_c["metadata"].apply(lambda x: x["tv_id"])
        df_c["legis_id"] = df_c["metadata"].apply(lambda x: x["legis_id"])
        col_order = ["chunk_id", "tv_id", "legis_id", "text", "metadata", "vec"]
        df_c = df_c[col_order]

        out_path = config.get_vec_path(congress_num)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_c.to_parquet(out_path)


def upload_hf(config: Config):

    rich.print("EMBEDDING (upload hf)")
    rich.print(config)

    tag = f"usc-{config.get_vecs_tag()}"
    upload_folder = config.congress_hf_path / tag
    repo_id = f"hyperdemocracy/{tag}"
    rich.print(f"{repo_id=}")
    rich.print(f"{upload_folder=}")

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=upload_folder,
        repo_id=repo_id,
        repo_type="dataset",
    )



class StaticRetrievalMrl:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def encode_docs(self, docs: list[str], batch_size: int = 64):
        vecs = self.model.encode(
            docs,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size,
        )
        return vecs

    def encode_query(self, query: str):
        vec = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return vec


def get_embedding_model(config: Config):
    if config.embedding_model_name == "sentence-transformers/static-retrieval-mrl-en-v1":
        return StaticRetrievalMrl(config.embedding_model_name)
    else:
        raise NotImplementedError(f"Model not implemented: {config.embedding_model_name}")


def write_readme(config: Config):
    out_path = config.congress_hf_path / f"usc-{config.get_vecs_tag()}"
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = Path(out_path) / "README.md"
    readme_str = get_readme_str(config)
    with open(fpath, "w") as fp:
        fp.write(readme_str)






if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/repos/legisplain/congress-hf")
    chunks = [(8192, 512), (4096, 512), (2048, 256), (1024, 256)]
    chunking_version = 1
    embedding_model_name = "sentence-transformers/static-retrieval-mrl-en-v1"

    do_write_local = False
    do_upload_hf = True

    for chunk_size, chunk_overlap in chunks:
        config = Config(
            congress_hf_path=congress_hf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_version=chunking_version,
            embedding_version=EMBEDDING_VERSION,
            embedding_model_name=embedding_model_name,
        )

        if do_write_local:
            write_readme(config)
            write_local(config)

        if do_upload_hf:
            upload_hf(config)

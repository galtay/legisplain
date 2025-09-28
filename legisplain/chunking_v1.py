from pathlib import Path
from typing import Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import HfApi
import rich
import pandas as pd

from legisplain.config import Config
from legisplain import utils


CHUNKING_VERSION = 1


def add_chunk_index(split_docs: list[Document]) -> list[Document]:
    chunk_index = -1
    cur_text_id = split_docs[0].metadata["tv_id"]
    for doc in split_docs:
        if cur_text_id == doc.metadata["tv_id"]:
            chunk_index += 1
        else:
            chunk_index = 0
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["chunk_id"] = "{}-{}".format(
            doc.metadata["tv_id"],
            chunk_index,
        )
        cur_text_id = doc.metadata["tv_id"]
    return split_docs


def write_local(config: Config):

    rich.print("CHUNKING (write local)")
    rich.print(config)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    for congress_num in config.congress_nums:

        u_path = config.get_unified_path(congress_num)
        rich.print(u_path)
        df_u = pd.read_parquet(u_path)

        docs = utils.get_langchain_docs_from_unified(df_u)
        chunked_docs = splitter.split_documents(docs)
        chunked_docs = add_chunk_index(chunked_docs)
        df_c = pd.DataFrame.from_records(
            [
                {
                    "chunk_id": doc.metadata["chunk_id"],
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in chunked_docs
            ]
        )

        df_c["tv_id"] = df_c["metadata"].apply(lambda x: x["tv_id"])
        df_c["legis_id"] = df_c["metadata"].apply(lambda x: x["legis_id"])

        # reorder columns
        cols = ["chunk_id", "tv_id", "legis_id", "text", "metadata"]
        assert set(cols) == set(df_c.columns)
        df_c = df_c[cols]

        out_path = config.get_chunk_path(congress_num)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rich.print(f"{out_path=}")
        df_c.to_parquet(out_path)


def upload_dataset(config: Config):
    chunking_tag = config.get_chunking_tag()
    ds_name = f"usc-{chunking_tag}"
    repo_id = f"hyperdemocracy/{ds_name}"
    rich.print(f"{repo_id=}")

    upload_folder = config.congress_hf_path / ds_name

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



if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/repos/legisplain/congress-hf")
    chunks = [(8192, 512), (4096, 512), (2048, 256), (1024, 256)]
    
    for chunk_size, chunk_overlap in chunks:
        config = Config(
            congress_hf_path=congress_hf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_version=CHUNKING_VERSION,
        )
        write_local(config)
        upload_dataset(config)

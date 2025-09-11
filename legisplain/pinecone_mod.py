import os
from pathlib import Path
from typing import Optional, Union

from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import rich
from tqdm import tqdm


def upsert_data(
    congress_hf_path: Union[str, Path],
    congress_nums: list[int],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    batch_size: int = 1000,
    nlim: Optional[int] = None,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")

    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_nums=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = get_index_name(model_name, chunk_size, chunk_overlap)
    index = pc.Index(index_name)

    for cn in congress_nums:

        dir_tag = f"usc-vecs-s{chunk_size}-o{chunk_overlap}-{model_tag}"
        file_tag = f"usc-{cn}-vecs-s{chunk_size}-o{chunk_overlap}-{model_tag}"
        fpath = congress_hf_path / dir_tag / "data" / f"{file_tag}.parquet"
        rich.print(f"{fpath=}")
        df_vec = pd.read_parquet(fpath)
        df_vec = df_vec.rename(columns={"metadata": "chunk_metadata"})
        if nlim is not None:
            df_vec = df_vec.head(nlim)

        dir_tag = "usc-unified"
        file_tag = f"usc-{cn}-unified"
        fpath = congress_hf_path / dir_tag / "data" / f"{file_tag}.parquet"
        rich.print(f"{fpath=}")
        df_uni = pd.read_parquet(fpath)
        assert df_uni.shape[0] == df_uni['legis_id'].nunique()

        df = pd.merge(df_vec, df_uni, on="legis_id")

        df["metadata"] = df.apply(
            lambda x: {
                "text": x["text"],
                "sponsor_bioguide_id": x["bs_json"]["sponsor"]["bioguide_id"],
                "sponsor_full_name": x["bs_json"]["sponsor"]["full_name"],
                "sponsor_party": x["bs_json"]["sponsor"]["party"],
                "sponsor_state": x["bs_json"]["sponsor"]["state"],
                "cosponsor_bioguide_ids": [el["bioguide_id"] for el in x["bs_json"]["cosponsors"]],
                "introduced_date": x["bs_json"]["introduced_date"],
                "policy_area": x["bs_json"]["policy_area"],
                "title": x["bs_json"]["title"],
                "subjects": x["bs_json"]["subjects"].tolist(),
                **x["chunk_metadata"],
            },
            axis=1,
        )

        assert df["chunk_id"].nunique() == df.shape[0]
        ii_los = list(range(0, df.shape[0], batch_size))
        for ii_lo in tqdm(ii_los):
            df_batch = df.iloc[ii_lo : ii_lo + batch_size]
            vectors = []
            for _, row in df_batch.iterrows():
                meta = {k: v for k, v in row["metadata"].items() if v is not None}
                vector = {
                    "id": row["chunk_id"],
                    "values": row["vec"].tolist(),
                    "metadata": meta,
                }
                vectors.append(vector)
            index.upsert(vectors=vectors)


def get_index_name(model_name: str, chunk_size: int, chunk_overlap: int):
    model_tag = model_name.split("/")[-1].replace(".","p")
    index_name = f"usc-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    return index_name


def create_index(model_name: str, chunk_size: int, chunk_overlap: int, dimension: int):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = get_index_name(model_name, chunk_size, chunk_overlap)
    print(index_name)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    return index_name



if __name__ == "__main__":
    congress_hf_path = Path("/Users/galtay/repos/legisplain/congress-hf")
    cns = [113, 114, 115, 116, 117, 118, 119]
    chunk_size = 8192
    chunk_overlap = 512
    batch_size = 50
    nlim = None

    model_name = "google/embeddinggemma-300m"
    dimension=768

    create_index(model_name, chunk_size, chunk_overlap, dimension)
    upsert_data(
        congress_hf_path,
        cns,
        chunk_size,
        chunk_overlap,
        model_name,
        batch_size,
        nlim=nlim,
    )



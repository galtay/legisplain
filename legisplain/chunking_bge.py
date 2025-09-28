from pathlib import Path
from typing import Union

from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from huggingface_hub import HfApi
import rich
import pandas as pd


CHUNKING_VERSION = "bge"


def get_langchain_docs_from_unified(df_u: pd.DataFrame) -> list[Document]:
    skipped = []
    docs = []
    for _, u_row in df_u.iterrows():
        if len(u_row["tvs"]) == 0:
            skipped.append(u_row["legis_id"])
            continue
        doc = Document(
            page_content=u_row["tvs"][0]["tv_txt"],
            metadata={
                "tv_id": u_row["tvs"][0]["tv_id"],
                "legis_version": u_row["tvs"][0]["legis_version"],
                "legis_class": u_row["tvs"][0]["legis_class"],
                "legis_id": u_row["legis_id"],
                "congress_num": u_row["congress_num"],
                "legis_type": u_row["legis_type"],
                "legis_num": u_row["legis_num"],
                "text_date": u_row["tvs"][0]["bs_tv"]["date"],
                "title": u_row["bs_json"]["title"],
                "sponsor_bioguide_id": u_row["bs_json"]["sponsor"]["bioguide_id"],
                "sponsor_full_name": u_row["bs_json"]["sponsor"]["full_name"],
                "sponsor_party": u_row["bs_json"]["sponsor"]["party"],
                "sponsor_state": u_row["bs_json"]["sponsor"]["state"],
                "introduced_date": u_row["bs_json"]["introduced_date"],
                "policy_area": u_row["bs_json"]["policy_area"],
            },
        )
        docs.append(doc)
    rich.print(f"skipped {len(skipped)} rows with no text versions")
    return docs


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


def _find_word_boundary_backwards(text: str, target_pos: int) -> int:
    """
    Find the nearest word boundary by searching backwards from target position.
    A word boundary is defined as whitespace (including newlines) or beginning/end of string.
    
    Args:
        text: The text to search in
        target_pos: The target position to search backwards from
    
    Returns:
        Position where chunk should end (exclusive for slicing)
    """
    if target_pos >= len(text):
        return len(text)
    
    if target_pos <= 0:
        return 0
    
    # Search backwards for whitespace (includes spaces, tabs, newlines, etc.)
    for i in range(target_pos - 1, -1, -1):
        if text[i].isspace():
            # Found whitespace, return position after it (end of previous word)
            return i + 1
    
    # If no whitespace found, return beginning of text
    return 0


def chunk_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Simple sliding window chunker that respects word boundaries.
    
    Chunks are created by sliding a window of approximately chunk_size characters.
    When the window would end in the middle of a word, we search backwards to find
    the nearest whitespace and end the chunk there. The final chunk always extends
    to the end of the text.
    
    Args:
        docs: List of Document objects to chunk
        chunk_size: Target size of each chunk in characters  
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        List of chunked Document objects
    """
    chunked_docs = []
    
    for doc in docs:
        text = doc.page_content
        text_length = len(text)
        
        # If text is shorter than chunk_size, keep as single chunk
        if text_length <= chunk_size:
            new_metadata = doc.metadata.copy()
            new_metadata['chunk_start'] = 0
            new_metadata['chunk_end'] = text_length
            new_metadata['prefix_length'] = 0
            new_doc = Document(
                page_content=text,
                metadata=new_metadata
            )
            chunked_docs.append(new_doc)
            continue
        
        # Create chunks with sliding window
        start = 0
        
        while start < text_length:
            # Calculate target end position
            target_end = start + chunk_size
            
            # If target extends beyond text, use end of text
            if target_end >= text_length:
                end = text_length
            else:
                # Find word boundary by searching backwards from target
                end = _find_word_boundary_backwards(text, target_end)
                
                # If we ended up too close to start (no good word boundary found),
                # just use the target position
                if end <= start:
                    end = target_end
            
            # Create chunk
            chunk_text = text[start:end]
            new_metadata = doc.metadata.copy()
            new_metadata['chunk_start'] = start
            new_metadata['chunk_end'] = end
            new_metadata['prefix_length'] = 0
            
            new_doc = Document(
                page_content=chunk_text,
                metadata=new_metadata
            )
            chunked_docs.append(new_doc)
            
            # If we've reached the end, break
            if end >= text_length:
                break
            
            # Calculate next start position
            next_start = start + chunk_size - chunk_overlap
            
            # Ensure final chunk adjustment: if remaining text is less than chunk_size,
            # adjust start so final chunk extends to end of text
            remaining = text_length - next_start
            if remaining < chunk_size and remaining > 0:
                next_start = max(text_length - chunk_size, start + 1)
            
            start = next_start
    
    return chunked_docs


def write_local(
    congress_hf_path: Union[str, Path],
    congress_num: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):

    rich.print("CHUNKING (write local)")
    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")

    splitter = SentenceTransformersTokenTextSplitter(model_name=model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    u_fpath = (
        congress_hf_path / "usc-unified" / "data" / f"usc-{congress_num}-unified.parquet"
    )
    rich.print(u_fpath)
    df_u = pd.read_parquet(u_fpath)

    docs = get_langchain_docs_from_unified(df_u)
    sys.exit(0)
    chunked_docs = chunk_docs(docs, chunk_size, chunk_overlap)

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

    chunk_tag = f"chunks-{CHUNKING_VERSION}-s{chunk_size}-o{chunk_overlap}"
    file_tag = f"usc-{congress_num}-{chunk_tag}"
    # reorder columns
    cols = ["chunk_id", "tv_id", "legis_id", "text", "metadata"]
    df_c = df_c[cols]
    out_path = congress_hf_path / f"usc-{chunk_tag}" / "data"
    out_path.mkdir(parents=True, exist_ok=True)
    fout = out_path / f"{file_tag}.parquet"
    rich.print(f"{fout=}")
    print()
    df_c.to_parquet(fout)


def upload_dataset(congress_hf_path, chunk_size, chunk_overlap):
    chunk_tag = f"chunks-{CHUNKING_VERSION}-s{chunk_size}-o{chunk_overlap}"
    ds_name = f"usc-{chunk_tag}"
    repo_id = f"hyperdemocracy/{ds_name}"
    rich.print(f"{repo_id=}")

    upload_folder = congress_hf_path / ds_name

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
    model_name = "BAAI/bge-small-en-v1.5"

    #chunks = [(8192, 512), (4096, 512), (2048, 256), (1024, 256)]
    chunks = [(1024, 256)]
    for chunk_size, chunk_overlap in chunks:
        congress_nums = [119, 118, 117, 116, 115, 114, 113]
        for congress_num in congress_nums:
            write_local(congress_hf_path, congress_num, chunk_size, chunk_overlap, model_name)
        upload_dataset(congress_hf_path, chunk_size, chunk_overlap)

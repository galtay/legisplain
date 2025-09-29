"""
Create persistant ChromaDB files
"""

import shutil
from pathlib import Path
import chromadb
from chromadb.config import Settings
import pandas as pd
import rich
from huggingface_hub import HfApi, create_repo


from legisplain.config import Config


def create_chroma_client(config: Config):
    """Create and return a ChromaDB client with persistence"""
    persist_directory = config.get_chroma_persist_directory()
    persist_directory.mkdir(parents=True, exist_ok=True)    
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )
    return client


def check_collection_exists(client, collection_name="usc"):
    """Check if the ChromaDB collection exists and has data"""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        return count > 0, count
    except Exception:
        return False, 0


def load_dataset_to_chroma(client, vec_paths: list[Path], collection_name="usc", n_lim: int | None = None):
    """Load the HuggingFace dataset into ChromaDB"""
    
    # Create or get collection
    try:
        collection = client.get_collection(collection_name)
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection doesn't exist yet
        
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "US Congressional legislation chunks with embeddings"}
    )
    
    # Process each congress split
    total_docs = 0
    batch_size = 1000
    
    
    for vec_path in vec_paths:
        df_vecs = pd.read_parquet(vec_path)
        if n_lim is not None:
            df_vecs = df_vecs.head(n_lim)
        
        # Process in batches
        num_batches = (len(df_vecs) + batch_size - 1) // batch_size
        
        for i in range(0, len(df_vecs), batch_size):
            batch_end = min(i + batch_size, len(df_vecs))
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for idx in range(i, batch_end):
                item = df_vecs.iloc[idx]
                metadata = dict(item['metadata'])
                # Ensure all metadata values are JSON-serializable
                for key, value in metadata.items():
                    if value is None:
                        metadata[key] = ""
                    elif isinstance(value, (int, float, str, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                
                ids.append(item['chunk_id'])
                embeddings.append(item['vec'])
                documents.append(item['text'])
                metadatas.append(metadata)
            
            # Add batch to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            total_docs += len(ids)
            
            # Update progress
            batch_num = (i // batch_size) + 1
            rich.print(f"vec_path: {vec_path}, Batch {batch_num}/{num_batches} ({total_docs} docs loaded)")
    

    rich.print(f"Successfully loaded {total_docs} documents into ChromaDB")
    
    return collection


def get_collection_info(client, collection_name="usc_legislation"):
    """Get information about the ChromaDB collection"""
    try:
        collection = client.get_collection(collection_name)
        return {
            "exists": True,
            "count": collection.count(),
            "name": collection_name
        }
    except Exception:
        return {
            "exists": False,
            "count": 0,
            "name": collection_name
        }


def upload_chroma_to_hf(local_chroma_dir: Path, hf_org: str = "hyperdemocracy"):
    """
    Upload ChromaDB directory to HuggingFace dataset following naming convention.
    
    Args:
        local_chroma_dir: Local path to ChromaDB directory (e.g., .../congress-hf/usc-chroma-.../chromadb)
        hf_org: HuggingFace organization name (default: "hyperdemocracy")
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the parent directory name for the dataset name
        chroma_parent_dir = local_chroma_dir.parent
        dataset_name = chroma_parent_dir.name  # e.g., "usc-chroma-vecs-v1-chunks-v1-s8192-o512-sentence-transformers-static-retrieval-mrl-en-v1"
        repo_id = f"{hf_org}/{dataset_name}"
        
        if not local_chroma_dir.exists():
            rich.print(f"[red]Error: Local ChromaDB directory does not exist: {local_chroma_dir}[/red]")
            return False
        
        # Initialize HF API
        api = HfApi()
        
        rich.print(f"[blue]Uploading ChromaDB to HuggingFace dataset: {repo_id}[/blue]")
        
        # Create the dataset repository
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            rich.print(f"[green]Created/verified HF dataset repository: {repo_id}[/green]")
        except Exception as e:
            rich.print(f"[red]Error creating HF repository: {e}[/red]")
            return False
        
        # Create a simple README for the dataset
        readme_content = f"""---
license: cc0-1.0
tags:
- chromadb
- legal
- congress
- legislation
- embeddings
size_categories:
- 100K<n<1M
---

# {dataset_name}

This dataset contains a pre-built ChromaDB database with US Congressional legislation embeddings.

## Dataset Structure

This dataset contains the ChromaDB files for legislation chunks with embeddings. The database can be loaded directly using ChromaDB's PersistentClient.

## Usage

```python
import chromadb
from huggingface_hub import snapshot_download

# Download the dataset
local_dir = snapshot_download(repo_id="{repo_id}", repo_type="dataset")

# Load the ChromaDB
client = chromadb.PersistentClient(path=local_dir + "/chromadb")
collection = client.get_collection("usc")

# Query the database
results = collection.query(
    query_texts=["your query here"],
    n_results=10
)
```

## Configuration

- Chunk size and overlap settings are encoded in the dataset name
- Embedding model information is included in the dataset name
- This follows the hyperdemocracy dataset naming convention
"""
        
        try:
            # Upload README
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset"
            )
            rich.print("[green]Uploaded README.md[/green]")
        except Exception as e:
            rich.print(f"[yellow]Warning: Could not upload README: {e}[/yellow]")
        
        # Upload the entire ChromaDB directory
        try:
            api.upload_folder(
                folder_path=chroma_parent_dir,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload ChromaDB for {dataset_name}"
            )
            rich.print(f"[green]Successfully uploaded ChromaDB to HuggingFace![/green]")
            rich.print(f"[green]Dataset URL: https://huggingface.co/datasets/{repo_id}[/green]")
            return True
            
        except Exception as e:
            rich.print(f"[red]Error uploading ChromaDB directory: {e}[/red]")
            return False
            
    except Exception as e:
        rich.print(f"[red]Unexpected error uploading to HuggingFace: {e}[/red]")
        return False





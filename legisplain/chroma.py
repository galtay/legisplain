"""
Create persistant ChromaDB files
"""

import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import rich
from chromadb.config import Settings, APIVersion
from dotenv import load_dotenv
import time
import random


from legisplain.config import Config


def retry_with_backoff(func, max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Retry a function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                rich.print(f"[red]Failed after {max_retries} retries: {e}[/red]")
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            rich.print(f"[yellow]Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...[/yellow]")
            time.sleep(delay)


def create_chroma_client(config: Config):
    """Create and return a ChromaDB client"""
    load_dotenv()
    host = os.getenv("CHROMA_PROXY_BASE")
    token = os.getenv("CHROMA_AUTH_TOKEN")
    if not host or not token:
        raise ValueError("Set CHROMA_PROXY_BASE and CHROMA_AUTH_TOKEN")

    headers = {"Authorization": f"Bearer {token}"}
    settings = Settings(
        chroma_server_api_default_path=APIVersion.V2, anonymized_telemetry=False
    )
    client = chromadb.HttpClient(host=host, headers=headers, settings=settings)

    return client


def check_collection_exists(client, collection_name="usc"):
    """Check if the ChromaDB collection exists and has data"""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        return count > 0, count
    except Exception:
        return False, 0


def populate(config: Config, vec_paths: list[Path], collection_name="usc", n_lim: int | None = None):
    """Load the HuggingFace dataset into ChromaDB with proper embedding function"""
    
    # Create embedding function that matches the model used
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model_name
    )
    rich.print(embedding_function)
    client = create_chroma_client(config)

    # Get existing collection or create new one
    try:
        collection = client.get_collection(collection_name)
    except chromadb.errors.NotFoundError:
        # Collection doesn't exist, create it
        rich.print(f"Collection '{collection_name}' does not exist, creating new one")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={
                "description": "US Congressional legislation chunks with embeddings",
                "embedding_model": config.embedding_model_name,
                "embedding_version": config.embedding_version,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "chunking_version": config.chunking_version,
            }
        )
    
    # Check document count and delete if needed
    doc_count = collection.count()
    rich.print(f"Collection '{collection_name}' has {doc_count} documents")
    
    if doc_count > 0:
        rich.print("Collection has documents, deleting and recreating...")
        def delete_collection():
            return client.delete_collection(collection_name)
        
        retry_with_backoff(delete_collection, max_retries=2, base_delay=5.0, max_delay=120.0)
        rich.print(f"Successfully deleted collection '{collection_name}'")
        
        # Recreate the collection
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={
                "description": "US Congressional legislation chunks with embeddings",
                "embedding_model": config.embedding_model_name,
                "embedding_version": config.embedding_version,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "chunking_version": config.chunking_version,
            }
        )
        rich.print("Recreated empty collection")
    else:
        rich.print("Collection is empty, ready to populate")
    
    # Process each congress split
    total_docs = 0
    batch_size = 50
    
    
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
            
            # Add batch to collection with retry logic
            def add_batch():
                return collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            
            retry_with_backoff(add_batch)
            
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








import chromadb

persist_path = "congress-hf/usc-chroma-vecs-v1-chunks-v1-s8192-o512-sentence-transformers-static-retrieval-mrl-en-v1/chromadb"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_collection("usc")

print(collection.count())

# show one item in the collection
#item_id = "119-hconres-2-ih-dtd-0"
item_id = "119-sres-408-is-dtd-1"
item = collection.get(ids=[item_id], include=["embeddings", "documents", "metadatas"])
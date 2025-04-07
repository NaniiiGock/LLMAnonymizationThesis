import os
import uuid
from typing import List, Optional

import chromadb
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChromaDBManager:
    def __init__(
        self,
        collection_name: str = "rag_collection",
        use_cloud: bool = False,
        persist_path: str = "./chroma_db1",
        host: str = "localhost",
        port: int = 8000,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        openai_model: str = "text-embedding-3-small"
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_model = openai_model

        # Init OpenAI client
        self.openai_client = OpenAI()

        # Init Chroma client (local or cloud)
        if use_cloud:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            self.client = chromadb.PersistentClient(path=persist_path)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Use OpenAI API to embed a list of texts."""
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.openai_model
        )
        return [r.embedding for r in response.data]

    def store_file(self, file_path: str, metadata: Optional[dict] = None):
        """Reads a file, splits it into chunks, embeds, and stores in Chroma."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self.splitter.split_text(text)
        embeddings = self._embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]

        default_metadata = metadata or {"source": file_path}
        metadatas = [default_metadata for _ in chunks]


        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ Stored {len(chunks)} chunks from '{file_path}' into '{self.collection_name}'.")

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieves top-k relevant chunks for a query."""
        query_embedding = self._embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results["documents"][0] if results["documents"] else []

    def remove_by_ids(self, ids: List[str]):
        """Deletes documents by their IDs."""
        self.collection.delete(ids=ids)
        print(f"🗑️ Removed {len(ids)} documents from '{self.collection_name}'.")

    def reset_collection(self):
        """Deletes and recreates the current collection (clear all data)."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"♻️ Collection '{self.collection_name}' has been reset.")

    def count_documents(self) -> int:
        return self.collection.count()



db = ChromaDBManager()

# Store content
db.store_file("processing/rag_data/file1.txt")

# Retrieve
query = "Who discovered gravity?"
results = db.retrieve(query)
print(results)

# Count
print("Stored chunks:", db.count_documents())

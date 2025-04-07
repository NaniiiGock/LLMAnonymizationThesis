import base64
import numpy as np
from cryptography.fernet import Fernet
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity


class SecureEmbeddingStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", db_path="./secure-chroma", collection_name="secure_embeddings", key=None):
        self.model = SentenceTransformer(model_name)
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.client = chromadb.PersistentClient(persist_directory=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def _encrypt_vector(self, vec: np.ndarray) -> str:
        byte_vec = vec.astype(np.float32).tobytes()
        return base64.b64encode(self.cipher.encrypt(byte_vec)).decode("utf-8")

    def _decrypt_vector(self, encoded: str) -> np.ndarray:
        byte_vec = self.cipher.decrypt(base64.b64decode(encoded))
        return np.frombuffer(byte_vec, dtype=np.float32)

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def add_document(self, doc_id: str, masked_text: str):
        embedding = self._embed(masked_text)
        encrypted_embedding = self._encrypt_vector(np.array(embedding))

        self.collection.add(
            ids=[doc_id],
            documents=[masked_text],
            metadatas=[{"embedding_encrypted": encrypted_embedding}]
        )

    def search(self, query_text: str, top_k=3):
        query_embedding = self._embed(query_text).reshape(1, -1)

        results = self.collection.get(include=["documents", "metadatas"])
        docs = results["documents"]
        metadatas = results["metadatas"]
        ids = results["ids"]

        decrypted_vecs = [self._decrypt_vector(meta["embedding_encrypted"]) for meta in metadatas]
        decrypted_vecs = np.vstack(decrypted_vecs)

        similarities = cosine_similarity(query_embedding, decrypted_vecs)[0]
        sorted_indices = similarities.argsort()[::-1][:top_k]

        return [
            {
                "id": ids[i],
                "document": docs[i],
                "similarity": round(float(similarities[i]), 4)
            }
            for i in sorted_indices
        ]

    def list_documents(self):
        return self.collection.get(include=["documents"])["documents"]

    def get_encryption_key(self):
        return self.key.decode("utf-8") if isinstance(self.key, bytes) else self.key


if __name__ == "__main__":
    store = SecureEmbeddingStore()

    # Add a masked document
    store.add_document(
        doc_id="doc1",
        masked_text="Hi, I'm [PER_1] from [LOC_1]. My email is [EMAIL_1]."
    )

    # Search with query
    results = store.search("Who is the person from New York?", top_k=2)

    print("🔍 Top Matches:")
    for r in results:
        print(f"• {r['id']} → {r['document']} (score={r['similarity']})")

    print("🔑 Encryption Key (save this!):", store.get_encryption_key())


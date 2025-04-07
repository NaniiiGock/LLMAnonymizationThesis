import tenseal as ts
from sentence_transformers import SentenceTransformer
import pickle
import os

# Initialize SentenceTransformer for multilingual embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ----------------------
# Homomorphic Encryption Setup
# ----------------------
def init_encryption_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 21, 40]
    )
    context.generate_galois_keys()
    context.global_scale = 2**21
    return context

# ----------------------
# Encrypt & Decrypt Utils
# ----------------------
def encrypt_vector(context, embedding):
    return ts.ckks_vector(context, embedding)

def decrypt_vector(encrypted_vector):
    return encrypted_vector.decrypt()

# ----------------------
# Simulated Secure Vector Store
# ----------------------
class EncryptedVectorStore:
    def __init__(self):
        self.store = []  # list of (id, encrypted_vector)

    def add(self, doc_id, encrypted_vector):
        self.store.append((doc_id, encrypted_vector))

    def search(self, query_encrypted, top_k=3):
        scored = []
        for doc_id, enc_vec in self.store:
            score = query_encrypted.dot(enc_vec).decrypt()[0]
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# ----------------------
# Demo
# ----------------------
if __name__ == "__main__":
    context = init_encryption_context()

    # Sample documents
    docs = {
        "doc1": "Устимко сидів біля вогнища",
        "doc2": "Марічка та Тарас їхали до Житомира",
        "doc3": "По воду я ходив давно"
    }

    store = EncryptedVectorStore()

    # Encode and encrypt documents
    for doc_id, text in docs.items():
        vec = model.encode(text)
        encrypted_vec = encrypt_vector(context, vec)
        store.add(doc_id, encrypted_vec)

    # Encode and encrypt query
    query = "Устимком співав пісню"
    query_vec = model.encode(query)
    query_enc = encrypt_vector(context, query_vec)

    # Secure search
    results = store.search(query_enc, top_k=2)

    print("🔍 Top Encrypted Matches:")
    for doc_id, score in results:
        print(f"{doc_id}: score = {score:.4f}")
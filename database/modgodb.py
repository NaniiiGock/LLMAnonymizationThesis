import uuid
import os
import certifi
import tenseal as ts
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

class MongoHEVectorStore:
    def __init__(
        self,
        mongo_uri="mongodb+srv://lilianahotsko:5PFKBqvr7k6GgdCO@cluster0.f8qmsji.mongodb.net/?retryWrites=true&w=majority",
        db_name="encrypted_rag",
        collection_name="documents",
        chunk_size=500,
        chunk_overlap=50,
        model_name="all-MiniLM-L6-v2",
    ):
        self.client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.model = SentenceTransformer(model_name)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()
    
    def _encrypt_embedding(self, embedding):
        return ts.ckks_vector(self.context, embedding).serialize()

    def _decrypt_dot_product(self, enc_vec1, enc_vec2):
        vec1 = ts.ckks_vector_from(self.context, enc_vec1)
        vec2 = ts.ckks_vector_from(self.context, enc_vec2)
        return vec1.dot(vec2).decrypt()[0]
    
    def _embed_and_encrypt(self, texts):
        embeddings = self.model.encode(texts)
        return [self._encrypt_embedding(e) for e in embeddings]
    
    def load_files(self, folder_path):
        docs = []
        for fname in os.listdir(folder_path):
            path = os.path.join(folder_path, fname)
            if fname.endswith(".txt"):
                docs += TextLoader(path).load()
            elif fname.endswith(".pdf"):
                docs += PyPDFLoader(path).load()
        text = " ".join([doc.page_content.replace("\n", " ") for doc in docs])
        return text

    def store_file(self, folder_path, metadata=None):
        text = self.load_files(folder_path)
        chunks = self.splitter.split_text(text)
        encrypted_embeddings = self._embed_and_encrypt(chunks)

        for chunk, enc_vec in zip(chunks, encrypted_embeddings):
            doc_id = str(uuid.uuid4())
            doc = {
                "_id": doc_id,
                "text": chunk,
                "embedding": enc_vec, 
                "metadata": metadata or {"source": folder_path},
            }
            self.collection.insert_one(doc)
        print(f"Stored {len(chunks)} encrypted chunks in collection '{self.collection.name}'.")

    def retrieve(self, query, k=5):
        query_embedding = self.model.encode([query])[0]
        encrypted_query = self._encrypt_embedding(query_embedding)
        results = []

        for doc in self.collection.find():
            try:
                score = self._decrypt_dot_product(encrypted_query, doc["embedding"])
                results.append((score, doc["text"]))
            except Exception as e:
                print(f"Error scoring document {doc.get('_id', 'N/A')}: {e}")
        
        top_k = sorted(results, key=lambda x: x[0], reverse=True)[:k]
        return [text for _, text in top_k]

    def reset_collection(self):
        self.collection.drop()
        print(f"MongoDB collection '{self.collection.name}' has been reset.")

    def count_documents(self):
        return self.collection.count_documents({})
    
    def run_retriever(self, query, folder_path):
        self.store_file(folder_path)
        return self.retrieve(query)


import time

start = time.time()
store = MongoHEVectorStore()
print("Initialized")
print("Time used: ", time.time() - start)

start = time.time()
store.store_file("uploaded_files")
print("Stored")
print("Time used: ", time.time() - start)

start = time.time()
results = store.retrieve("Хто був за кермом?")
print(results)
print("Time Used: ", time.time() - start)


# import certifi
# from pymongo import MongoClient

# uri = "mongodb+srv://lilianahotsko:5PFKBqvr7k6GgdCO@cluster0.f8qmsji.mongodb.net/?retryWrites=true&w=majority"
# client = MongoClient(uri, tlsCAFile=certifi.where())
# db = client["my_db"]
# print(db.list_collection_names())

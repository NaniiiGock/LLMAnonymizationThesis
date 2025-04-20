import os
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

class ChromaDBManager:
    # TODO: add a choice of local(sentence_transformer) or remote embeddings (openai)
    
    def __init__(
        self,
        collection_name="rag_collection",
        persist_path="./chroma_db1",
        chunk_size=500,
        chunk_overlap=50,
        model_name="all-MiniLM-L6-v2", 
        reset=True 
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_path)
        if reset:
            self.reset_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

    def _embed_texts(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings

    def load_files(self, folder_path):   
        uploaded_file_list = os.listdir(folder_path)
        file_paths = []
        if uploaded_file_list:
            for file_name in uploaded_file_list:
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)
        docs = []
        for file_path in file_paths:
            file_type = file_path.split(".")[-1]
            if file_type == "txt":
                loader = TextLoader(file_path)
                docs += loader.load()
            elif file_type == "pdf":
                loader = PyPDFLoader(file_path)
                docs += loader.load()

        document_texts = " ".join([doc.page_content for doc in docs])
        document_texts = document_texts.replace("\n", " ")
        print(document_texts)

        return document_texts
    
    def store_file(self, folder_path, metadata=None):
        text = self.load_files(folder_path)
        chunks = self.splitter.split_text(text)
        embeddings = self._embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]

        default_metadata = metadata or {"source": folder_path}
        metadatas = [default_metadata for _ in chunks]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Stored {len(chunks)} chunks from '{folder_path}' into '{self.collection_name}'.")

    def retrieve(self, query, k=5):
        query_embedding = self._embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results["documents"][0] if results["documents"] else []

    def remove_by_ids(self, ids):
        self.collection.delete(ids=ids)
        print(f"Removed {len(ids)} documents from '{self.collection_name}'.")

    def reset_collection(self):
        existing_collections = self.client.list_collections()
        print(f"Existing collections: {existing_collections}")

        if self.collection_name in existing_collections:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted.")
        else:
            print(f"Collection '{self.collection_name}' does not exist. Skipping deletion.")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' has been reset.")


    def count_documents(self) -> int:
        return self.collection.count()
    
    def run_retriever(self, query, file_path=None):
        if file_path:
            self.store_file(file_path)
        results = self.retrieve(query)
        return results
    
    def get_stored_chunks(self):
        results = self.collection.get(include=["documents"])
        stored_chunks = list(zip(results["documents"]))
        return stored_chunks


# db = ChromaDBManager()
# results = db.run_retriever("Хто був за кермом", "uploaded_files")
# print(results)
# print("Stored chunks:", db.count_documents())

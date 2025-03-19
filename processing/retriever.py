from langchain_chroma import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import os

load_dotenv()

class Retriever:
    def __init__(self, config=None):
        self.config = config
        pass

    def load_file(self, folder_path):
        
        uploaded_file_list = os.listdir(folder_path)
        file_paths = []
        if uploaded_file_list:
            for file_name in uploaded_file_list:
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)

        docs = []
        for file_path in file_paths:
            # if file_path.starts_with("http"):
            #     docs += self.load_link(file_path)
            file_type = file_path.split(".")[-1]
            if file_type == "txt":
                loader = TextLoader(file_path)
                docs += loader.load()
            elif file_type == "pdf":
                loader = PyPDFLoader(file_path)
                docs += loader.load()
        return docs

    def load_link(self, link):
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=(link,),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        return docs

    def split(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)

        print(f"Split blog post into {len(all_splits)} sub-documents.")

        return all_splits

    def store(self, all_splits):
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        return vectorstore
    
    def retrieve(self, vectorstore, query):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke(query)
        return retrieved_docs
    
    def run_retriever(self, user_input, task, upload_folder):
        docs = self.load_file(upload_folder)
        all_splits = self.split(docs)
        vectorstore = self.store(all_splits)
        retrieved_docs = self.retrieve(vectorstore, task)
        retrieved_docs = [doc.page_content for doc in retrieved_docs]
        return retrieved_docs

async def main():
    retriever = Retriever()
    user_input = "This is the book on WW2"
    task = "What happened opposite the defensive French installations of the Maginot line"
    upload_folder = "/Users/lilianahotsko/Desktop/University/UCU/LLMAnonymizationUV/uploaded_files"
    retrieved_docs = retriever.run_retriever(user_input, task, upload_folder)
    
    print(retrieved_docs)

    import openai 
    from dotenv import load_dotenv

    load_dotenv()

    class OpenAIProvider:
        def __init__(self, config):

            self.model = config["model"]
            # self.system_prompt = config["system_prompt"]
            self.temperature = config["temperature"]
            self.max_tokens = config["max_tokens"]
            
        async def query_llm(self, prompt):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        # {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error querying LLM: {str(e)}"
    # from ..providers.openai_provider import OpenAIProvider

    openai_config = {"model":"gpt-4o", "temperature":0.7, "max_tokens": 1000}
    openai_model = OpenAIProvider(openai_config)
    prompt = f"User Input: {task} \n\nRetrieved docs: {retrieved_docs}"
    response =  await openai_model.query_llm(prompt)
    print(response)


if __name__=="__main__":
    import asyncio
    asyncio.run(main())
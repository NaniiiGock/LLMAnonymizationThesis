from processing.retriever import Retriever
from providers.openai_provider import OpenAIProvider
import asyncio
from ragas import SingleTurnSample
from ragas.metrics import BleuScore
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from dotenv import load_dotenv
load_dotenv()

async def main():
    config = {"chunk_size": 200, "chunk_overlap": 20, "retrieval_type": "similarity", "k": 5}

    retriever = Retriever(config)
    upload_folder = "processing/rag_data/"
    openai_config = {"model":"gpt-4o", "temperature":0.7, "max_tokens": 1000}
    openai = OpenAIProvider(openai_config)

    llm = ChatOpenAI(model="gpt-4o")

    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?"
    ]

    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]

    dataset = []

    docs = retriever.load_file(upload_folder)
    all_splits = retriever.split(docs)
    vectorstore = retriever.store(all_splits)
    
    for query,reference in zip(sample_queries,expected_responses):
        retrieved_docs = retriever.retrieve(vectorstore, query)
        retrieved_docs = [doc.page_content for doc in retrieved_docs]
        relevant_docs = [doc.replace("\n", "") for doc in retrieved_docs]
        prompt = f"User Input: {query} \n\nRetrieved docs: {relevant_docs}"
        response = await openai.query_llm(prompt)
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":[doc for doc in relevant_docs],
                "response":response,
                "reference":reference
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        BleuScore()
    ],
    llm=evaluator_llm
)

    print(result)

if __name__=="__main__":
    asyncio.run(main())

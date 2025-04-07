import os
import openai
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


import ollama

class OllamaProvider:
    def __init__(self):
        pass
    def query_llm(self, prompt, task):
        try:
            response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt + task
            }
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
        
llama = OllamaProvider()
openai.api_key = os.getenv("OPENAI_API_KEY")
loader = TextLoader("/Users/lilianahotsko/Desktop/LLMAnonymizationThesis/uploaded_files/ww2.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding)

def generate_questions_from_text(text, n=3):
    prompt = f"Generate {n} factual, answerable questions based on this text:\n\n{text}"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return [q.strip("- ").strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]

def generate_ground_truth(question, context):
    prompt = f"Using the context below, answer the question concisely and accurately.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_llama_answer(question, context):
    response = llama.query_llm(context, question)
    return response

dataset = []

for doc in documents[:1]:  # limit for testing
    questions = generate_questions_from_text(doc.page_content)
    for q in questions:
        context_docs = vectorstore.similarity_search(q, k=3)
        contexts = [d.page_content for d in context_docs]
        context_str = "\n".join(contexts)

        ground_truth = generate_ground_truth(q, context_str)
        llama_answer = generate_llama_answer(q, context_str)

        dataset.append({
            "question": q,
            "contexts": contexts,
            "answer": llama_answer,
            "ground_truth": ground_truth
        })

df = pd.DataFrame(dataset)
import ast 

def safe_parse_contexts(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return [val] if isinstance(val, str) else []

df["contexts"] = df["contexts"].apply(safe_parse_contexts)

import pandas as pd
import ast
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


ragas_dataset = Dataset.from_pandas(df)


# Evaluate
results = evaluate(
    ragas_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

# Print scores
print("📊 RAG Evaluation Results:")
# for metric, score in results.scores:
#     print(f"{metric}: {score:.4f}")
print(results.scores)
df.to_csv("ragas_dataset.csv", index=False)
print("✅ Saved dataset to ragas_dataset.csv")

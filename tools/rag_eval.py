import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# --- Load from CSV and convert context string to list ---
df = pd.read_csv("ragas_dataset.csv")

# Convert stringified list if needed
if isinstance(df.contexts.iloc[0], str):
    df["contexts"] = df["contexts"].apply(eval)
    df['response'] = df['answer'].apply(eval)
# Convert to HuggingFace Dataset
ragas_dataset = Dataset.from_pandas(df)



from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
result = evaluate(dataset=ragas_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
print(result)
# --- Run Evaluation ---
# results = evaluate(
#     ragas_dataset,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#         context_recall
#     ]
# )

# print("📊 Evaluation Results:")
# print(results)

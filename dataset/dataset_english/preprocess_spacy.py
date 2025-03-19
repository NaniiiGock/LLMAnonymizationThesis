import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import ast

df = pd.read_csv('PII43k.csv')

nlp = spacy.blank("en")

def create_doc_bin(dataframe, output_path):
    doc_bin = DocBin()
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        text = row['Filled Template']
        tokens = ast.literal_eval(row['Tokenised Filled Template'])
        labels = ast.literal_eval(row['Tokens'])
        
        doc = nlp.make_doc(text)
        
        ents = []
        current_position = 0
        for token, label in zip(tokens, labels):
            token_start = text.find(token, current_position)
            token_end = token_start + len(token)
            current_position = token_end
            if label != 'O': 
                span = doc.char_span(token_start, token_end, label=label[2:])  # Remove 'B-' or 'I-' prefix
                if span is not None:
                    ents.append(span)
        
        filtered_ents = spacy.util.filter_spans(ents)
    
        doc.ents = filtered_ents
    
        doc_bin.add(doc)
    
    doc_bin.to_disk(output_path)


from sklearn.model_selection import train_test_split

train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)
create_doc_bin(train_df, "data/train.spacy")
create_doc_bin(dev_df, "data/dev.spacy")

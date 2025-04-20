import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'dataset/dataset_ukr/data/ner-uk/v2.0/iob/train.iob'

words = []
pos_tags = []
ner_tags = []
sentences = []

with open(data_path, 'r', encoding='ISO-8859-1') as file:
    sentence = []
    sentence_number = 1 
    
    for line in file:
        line = line.strip()
        
        if not line:
            continue
        
        parts = line.split()
        
        if len(parts) == 2:  
            word, pos = parts
            tag = 'O' 
        elif len(parts) == 3: 
            word, pos, tag = parts
        else:
            print(f"Skipping line due to incorrect format: {line}")
            continue 
        
        sentence.append((word, pos, tag)) 
        
        if tag == 'O' and sentence[-1][1] == 'O':  
            for word, pos, tag in sentence:
                words.append(word)
                pos_tags.append(pos)
                ner_tags.append(tag)
                sentences.append(sentence_number)
            sentence = [] 
            sentence_number += 1

df = pd.DataFrame({
    'Word': words,
    'POS': pos_tags,
    'Tag': ner_tags,
    'Sentence': sentences
})

print("Basic Overview of the Data:")
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Data Types:\n{df.dtypes}\n")
print("First 5 Rows:")
print(df.head())

print("\nNER Tag Distribution:")
ner_tag_counts = df['Tag'].value_counts()
print(ner_tag_counts)

plt.figure(figsize=(10, 6))
sns.countplot(y='Tag', data=df, order=ner_tag_counts.index, palette='Set2')
plt.title('Distribution of NER Tags')
plt.xlabel('Count')
plt.ylabel('NER Tag')
plt.show()

print("\nPOS Tag Distribution:")
pos_tag_counts = df['POS'].value_counts()
print(pos_tag_counts)

plt.figure(figsize=(10, 6))
sns.countplot(y='POS', data=df, order=pos_tag_counts.index, palette='Set3')
plt.title('Distribution of POS Tags')
plt.xlabel('Count')
plt.ylabel('POS Tag')
plt.show()

print("\nMissing Data Analysis:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

sentence_ner_tag_counts = df.groupby('Sentence')['Tag'].value_counts().unstack(fill_value=0)
print("\nNER Tag Counts per Sentence:")
print(sentence_ner_tag_counts.head())

with open("ukrainian_data_exploration_output.txt", "w") as f:
    f.write(f"Basic Overview of the Data:\n{df.shape}\n")
    f.write(f"Columns: {df.columns}\n")
    f.write(f"Data Types:\n{df.dtypes}\n")
    f.write("NER Tag Distribution:\n")
    f.write(str(ner_tag_counts))
    f.write("\nPOS Tag Distribution:\n")
    f.write(str(pos_tag_counts))
    f.write("\nMissing Data Analysis:\n")
    f.write(str(missing_data[missing_data > 0]))

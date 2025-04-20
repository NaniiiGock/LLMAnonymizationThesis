import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'models/NER_models/bert/ner_datasetreference.csv'  # Adjust this to the correct file path
df = pd.read_csv(data_path, encoding='ISO-8859-1')  # Try using ISO-8859-1 encoding

# 1. Basic Overview of the Data
print("Basic Overview of the Data:")
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Data Types:\n{df.dtypes}\n")
print("First 5 Rows:")
print(df.head())

# 2. Summary Statistics for Numerical and Categorical Features
print("\nSummary Statistics:")
print(df.describe(include='all'))

# 3. Class Distribution of NER Tags (Target Variable)
print("\nNER Tag Distribution:")
ner_tag_counts = df['Tag'].value_counts()
print(ner_tag_counts)

# Plot the distribution of NER tags
plt.figure(figsize=(10, 6))
sns.countplot(y='Tag', data=df, order=ner_tag_counts.index, palette='Set2')
plt.title('Distribution of NER Tags')
plt.xlabel('Count')
plt.ylabel('NER Tag')
plt.show()

# 4. Check for Missing Data
print("\nMissing Data Analysis:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# Plot missing data if any
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# 5. Sample Data
print("\nSample Data (First 20 rows):")
print(df.head(20))


# 6. NER Tag Counts for Each Sentence (group by # Sentence)

df['Sentence #'] = df['Sentence #'].fillna(method='ffill')

# Now we can group by 'Sentence #' and count the NER tags
sentence_ner_tag_counts = df.groupby('Sentence #')['Tag'].value_counts().unstack(fill_value=0)

# Display the result
print("NER Tag Counts per Sentence:")
print(sentence_ner_tag_counts.head())

# sentence_ner_tag_counts = df.groupby('# Sentence')['Tag'].value_counts().unstack(fill_value=0)
# print("\nNER Tag Counts per Sentence:")
# print(sentence_ner_tag_counts.head())

# Optional: Save the outputs to a text file
with open("data_exploration_output.txt", "w") as f:
    f.write(f"Basic Overview of the Data:\n{df.shape}\n")
    f.write(f"Columns: {df.columns}\n")
    f.write(f"Data Types:\n{df.dtypes}\n")
    f.write("Summary Statistics:\n")
    f.write(str(df.describe(include='all')))
    f.write("\nNER Tag Distribution:\n")
    f.write(str(ner_tag_counts))
    f.write("\nMissing Data Analysis:\n")
    f.write(str(missing_data[missing_data > 0]))

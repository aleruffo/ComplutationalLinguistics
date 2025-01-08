import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re

# Preprocessing function for text
def preprocess_text(df, text_column):
    df[text_column] = df[text_column].str.lower().str.strip()
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^a-zA-Z0-9\sàèéìòùäöüßâæçêëîïôœûÿãõáíóúâêôçéèëïíóöüñ.,!?;:()\'"-]', '', x))
    return df

# Handle class imbalance
def balance_data(df, label_column):
    majority = df[df[label_column] == 0]
    minority = df[df[label_column] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    return pd.concat([majority, minority_upsampled])

# Split dataset into train, validation, and test
def split_data(df, test_size=0.2, val_size=0.1):
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train, test_size=val_size / (1 - test_size), random_state=42)
    return train, val, test

# --- Misogyny Dataset ---
print("Processing Misogyny Dataset")

# Combine English and Italian datasets
train_data = pd.concat([
    pd.read_csv('datasets/raw/en_training_anon.tsv', sep='\t'),
    pd.read_csv('datasets/raw/it_training_anon.tsv', sep='\t')
])
test_data = pd.concat([
    pd.read_csv('datasets/raw/en_testing_anon.tsv', sep='\t'),
    pd.read_csv('datasets/raw/it_testing_anon.tsv', sep='\t')
])

train_data = preprocess_text(train_data, 'text')
test_data = preprocess_text(test_data, 'text')
train_data = balance_data(train_data, 'misogynous')
train_data, val_data, test_data = split_data(train_data)

train_data[['text', 'misogynous']].to_csv('datasets/processed/misogyny_train.csv', index=False)
val_data[['text', 'misogynous']].to_csv('datasets/processed/misogyny_val.csv', index=False)
test_data[['text', 'misogynous']].to_csv('datasets/processed/misogyny_test.csv', index=False)

# --- Irony Dataset ---
print("Processing Irony Dataset")

irony_df = pd.read_csv('datasets/raw/MultiPICo_Anonymized.csv')
irony_df = irony_df[~irony_df['language'].isin(['ar', 'hi'])]
irony_df['text'] = irony_df['post'] + " " + irony_df['reply']
irony_df = preprocess_text(irony_df, 'text')
train_irony, val_irony, test_irony = split_data(irony_df)

train_irony[['text', 'label']].to_csv('datasets/processed/irony_train.csv', index=False)
val_irony[['text', 'label']].to_csv('datasets/processed/irony_val.csv', index=False)
test_irony[['text', 'label']].to_csv('datasets/processed/irony_test.csv', index=False)

# --- Stance Dataset ---
print("Processing Stance Dataset")

# Function to load JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Process all languages together
stance_train = load_jsonl('datasets/raw/train.jsonl')
stance_valid = load_jsonl('datasets/raw/valid.jsonl')
stance_test = load_jsonl('datasets/raw/test.jsonl')

for df in [stance_train, stance_valid, stance_test]:
    df['text'] = df['question'] + " " + df['comment']
    df = preprocess_text(df, 'text')
    df['label'] = df['label'].map({'FAVOR': 1, 'AGAINST': 0})

stance_train[['text', 'label']].to_csv('datasets/processed/stance_train.csv', index=False)
stance_valid[['text', 'label']].to_csv('datasets/processed/stance_val.csv', index=False)
stance_test[['text', 'label']].to_csv('datasets/processed/stance_test.csv', index=False)

print("Preprocessing complete.")

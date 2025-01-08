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

# English
train_en = pd.read_csv('datasets/raw/en_training_anon.tsv', sep='\t')
test_en = pd.read_csv('datasets/raw/en_testing_anon.tsv', sep='\t')
train_en = preprocess_text(train_en, 'text')
test_en = preprocess_text(test_en, 'text')
train_en = balance_data(train_en, 'misogynous')
train_en, val_en, test_en = split_data(train_en)

train_en[['text', 'misogynous']].to_csv('datasets/processed/misogyny_train_en.csv', index=False)
val_en[['text', 'misogynous']].to_csv('datasets/processed/misogyny_val_en.csv', index=False)
test_en[['text', 'misogynous']].to_csv('datasets/processed/misogyny_test_en.csv', index=False)

# Italian
train_it = pd.read_csv('datasets/raw/it_training_anon.tsv', sep='\t')
test_it = pd.read_csv('datasets/raw/it_testing_anon.tsv', sep='\t')
train_it = preprocess_text(train_it, 'text')
test_it = preprocess_text(test_it, 'text')
train_it = balance_data(train_it, 'misogynous')
train_it, val_it, test_it = split_data(train_it)

train_it[['text', 'misogynous']].to_csv('datasets/processed/misogyny_train_it.csv', index=False)
val_it[['text', 'misogynous']].to_csv('datasets/processed/misogyny_val_it.csv', index=False)
test_it[['text', 'misogynous']].to_csv('datasets/processed/misogyny_test_it.csv', index=False)

# --- Irony Dataset ---
print("Processing Irony Dataset")

irony_df = pd.read_csv('datasets/raw/MultiPICo_Anonymized.csv')
# Filter out Arabic and Hindi
irony_df = irony_df[~irony_df['language'].isin(['ar', 'hi'])]

# Process each language separately
languages = irony_df['language'].unique()
for lang in languages:
    irony_lang = irony_df[irony_df['language'] == lang]
    irony_lang['text'] = irony_lang['post'] + " " + irony_lang['reply']
    irony_lang = preprocess_text(irony_lang, 'text')
    irony_lang, val_lang, test_lang = split_data(irony_lang, test_size=0.2, val_size=0.1)

    irony_lang[['text', 'label']].to_csv(f'datasets/processed/irony_train_{lang}.csv', index=False)
    val_lang[['text', 'label']].to_csv(f'datasets/processed/irony_val_{lang}.csv', index=False)
    test_lang[['text', 'label']].to_csv(f'datasets/processed/irony_test_{lang}.csv', index=False)

# --- Stance Dataset ---
print("Processing Stance Dataset")

# Function to load JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load and process training dataset for each language
stance_train = load_jsonl('datasets/raw/train.jsonl')
for lang in ['de', 'fr']:
    stance_lang = stance_train[stance_train['language'] == lang]
    stance_lang['text'] = stance_lang['question'] + " " + stance_lang['comment']
    stance_lang = preprocess_text(stance_lang, 'text')
    stance_lang['label'] = stance_lang['label'].map({'FAVOR': 1, 'AGAINST': 0})
    stance_lang[['text', 'label']].to_csv(f'datasets/processed/stance_train_{lang}.csv', index=False)

# Validation
stance_valid = load_jsonl('datasets/raw/valid.jsonl')
for lang in ['de', 'fr']:
    stance_lang = stance_valid[stance_valid['language'] == lang]
    stance_lang['text'] = stance_lang['question'] + " " + stance_lang['comment']
    stance_lang = preprocess_text(stance_lang, 'text')
    stance_lang['label'] = stance_lang['label'].map({'FAVOR': 1, 'AGAINST': 0})
    stance_lang[['text', 'label']].to_csv(f'datasets/processed/stance_valid_{lang}.csv', index=False)

# Test
stance_test = load_jsonl('datasets/raw/test.jsonl')
for lang in ['de', 'fr', 'it']:
    stance_lang = stance_test[stance_test['language'] == lang]
    stance_lang['text'] = stance_lang['question'] + " " + stance_lang['comment']
    stance_lang = preprocess_text(stance_lang, 'text')
    stance_lang['label'] = stance_lang['label'].map({'FAVOR': 1, 'AGAINST': 0})
    stance_lang[['text', 'label']].to_csv(f'datasets/processed/stance_test_{lang}.csv', index=False)

print("Preprocessing complete.")

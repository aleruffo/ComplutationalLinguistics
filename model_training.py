import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import os

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'label' in df.columns:
        return df['text'].tolist(), df['label'].tolist()
    elif 'misogynous' in df.columns:
        return df['text'].tolist(), df['misogynous'].tolist()
    else:
        raise ValueError("The dataset does not contain 'label' or 'misogynous' columns.")

# Prepare data for model
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Preprocess data
def preprocess_data(tokenizer, texts, labels, max_length=128):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length', # Force padding
        max_length=max_length,
        return_tensors='pt' # Ensure tensors are returned
    )
    return CustomDataset(encodings, labels)

# Train and Evaluate Model
def train_and_evaluate(model_name, train_file, val_file, test_file, output_dir, epochs=3, batch_size=8):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Ensure padding token is set properly
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Load data
    train_texts, train_labels = load_data(train_file)
    val_texts, val_labels = load_data(val_file)
    test_texts, test_labels = load_data(test_file)

    # Tokenize data
    train_data = preprocess_data(tokenizer, train_texts, train_labels)
    val_data = preprocess_data(tokenizer, val_texts, val_labels)
    test_data = preprocess_data(tokenizer, test_texts, test_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5, 
        logging_dir='./logs', # add logging directory for debugging
        report_to="tensorboard", # add tensorboard for monitoring
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train and Evaluate
    trainer.train()
    results = trainer.evaluate(eval_dataset=test_data)
    print("Test Results:", results)

# Model Configuration
models = {
    'xlm_roberta_base': 'xlm-roberta-base'
}

# Categories
# Categories
categories = ['misogyny', 'irony', 'stance']

# Train and Evaluate for Each Model and Category
for model_name, model_path in models.items():
    for category in categories:
        print(f"Training and Evaluating Model: {model_name} | Category: {category}")
        
        train_file = f'datasets/processed/{category}_train.csv'
        val_file = f'datasets/processed/{category}_val.csv'
        test_file = f'datasets/processed/{category}_test.csv'
        output_dir = f'datasets/{model_name}_{category}'
        
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            train_and_evaluate(model_path, train_file, val_file, test_file, output_dir)
        else:
            print(f"Files for {category} not found. Skipping.")

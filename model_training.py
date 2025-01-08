import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from peft import LoraConfig, get_peft_model
import numpy as np
import os
import warnings

# Suppress Warnings
warnings.filterwarnings('ignore')

# Jo start
lora_config = LoraConfig(
    r=8, #Rank for LoRA
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
# Jo end

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
def train_and_evaluate(model_name, train_file, val_file, test_file, output_dir, epochs=1, batch_size=8):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Jo start
    from transformers import BitsAndBytesConfig 

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # Enable 4-bit quantization
        bnb_4bit_compute_dtype=torch.bfloat16 # Use BF16 instead of FP16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=bnb_config, # apply quantization
        low_cpu_mem_usage=True, # optimize CPU memory during loading
        use_cache=False  # Disable cache to resolve the conflict
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    # Jo end

    # Ensure padding token is set properly
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    model.config.pad_token_id = tokenizer.pad_token_id

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
        max_grad_norm=0.5,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5, 
        bf16=True, # Use BF16 for compatibility
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit", # add optimized 8-bit optimizer
        logging_dir='./logs', # add logging directory for debugging
        report_to="tensorboard", # add tensorboard for monitoring
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,  # Updated deprecated argument
        compute_metrics=compute_metrics
    )

    # Train and Evaluate
    trainer.train()
    results = trainer.evaluate(eval_dataset=test_data)
    print("Test Results:", results)

# Model Configuration
models = {
    'mistral_nemo': 'mistralai/Mistral-Nemo-Instruct-2407'
}

# Categories and Languages
categories = ['misogyny', 'irony', 'stance']
languages = {
    'misogyny': ['en', 'it'],
    'irony': ['en', 'it', 'de', 'es', 'fr', 'nl', 'pt'],
    'stance': ['de', 'fr', 'it']
}

# Train and Evaluate for Each Model, Category, and Language
for model_name, model_path in models.items():
    for category in categories:
        for lang in languages[category]:
            print(f"Training and Evaluating Model: {model_name} | Category: {category} | Language: {lang}")
            
            train_file = f'datasets/processed/{category}_train_{lang}.csv'
            val_file = f'datasets/processed/{category}_val_{lang}.csv'
            test_file = f'datasets/processed/{category}_test_{lang}.csv'
            output_dir = f'datasets/{model_name}_{category}_{lang}'
            
            if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
                train_and_evaluate(model_path, train_file, val_file, test_file, output_dir)
            else:
                print(f"Files for {category} in {lang} not found. Skipping.")

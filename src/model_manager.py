import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from scipy.special import softmax

class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", max_length=128, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
    def load(self, path):
        """Loads model from a local directory."""
        if os.path.exists(path):
            print(f"Loading model from {path}")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self.device)
        else:
            print(f"Path {path} does not exist. Keeping current model.")

    def save(self, path):
        """Saves model and tokenizer to a local directory."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=False)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def train(self, start_model_path, train_data, val_data=None, output_dir="data/checkpoints", epochs=1):
        """
        Fine-tunes the model.
        train_data: list of dicts {'text': ..., 'label': ...} or HF Dataset
        """
        if start_model_path and os.path.exists(start_model_path):
             self.load(start_model_path)
             
        # Prepare datasets
        if isinstance(train_data, list):
            hf_train = Dataset.from_list(train_data)
        else:
            hf_train = train_data
            
        tokenized_train = hf_train.map(self.tokenize_function, batched=True)
        
        tokenized_val = None
        if val_data:
            if isinstance(val_data, list):
                hf_val = Dataset.from_list(val_data)
            else:
                hf_val = val_data
            tokenized_val = hf_val.map(self.tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            eval_strategy="epoch" if tokenized_val else "no",
            save_strategy="no",
            logging_dir=f"{output_dir}/logs",
            learning_rate=2e-5,
            use_cpu=self.device == "cpu"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )

        trainer.train()
        return trainer

    def predict(self, texts, batch_size=32, enable_dropout=False):
        """
        Returns probabilities for the positive class (label 1).
        texts: list of strings
        enable_dropout: if True, runs in train mode (for MC dropout)
        """
        self.model.eval()
        if enable_dropout:
            self.model.train() # Enable dropout
        
        all_probs = []
        
        # Batching
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # Keep only probability of positive class (index 1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                
        return np.array(all_probs)

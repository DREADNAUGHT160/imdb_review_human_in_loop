import os
import pandas as pd
import hashlib
from datasets import load_dataset
from datetime import datetime

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.labels_path = os.path.join(data_dir, "human_labels.csv")
        self.cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # In-memory storage
        self.train_dataset = None
        self.test_dataset = None
        self.unlabeled_pool = None
        
    def _get_hash(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_imdb(self, train_subset_size=2000):
        """
        Loads IMDb dataset.
        Splits train into:
          - initial_train (train_subset_size)
          - unlabeled_pool (remainder of train)
        Test set is kept as is.
        """
        print(f"Loading IMDb dataset...")
        # Load from HF
        dataset = load_dataset("imdb", cache_dir=self.cache_dir)
        
        # Shuffle train
        full_train = dataset["train"].shuffle(seed=42)
        
        # Split
        self.train_dataset = full_train.select(range(train_subset_size))
        self.unlabeled_pool = full_train.select(range(train_subset_size, len(full_train)))
        self.test_dataset = dataset["test"].shuffle(seed=42)
        
        # Add hash IDs if not present (IMDb is just text/label)
        # We'll compute hashes on the fly when needed or wrap in a helper if performance allows.
        # For now, we just store datasets as is.
        
        return self.train_dataset, self.test_dataset, self.unlabeled_pool

    def load_human_labels(self):
        if os.path.exists(self.labels_path):
            return pd.read_csv(self.labels_path)
        else:
            return pd.DataFrame(columns=[
                "id", "text", "label", "timestamp", "model_prob_pos", "strategy_used"
            ])

    def save_human_label(self, item, label, prob_pos, strategy):
        """
        Saves a single human label to disk.
        item: dict containing 'text'
        label: 0 (NEG) or 1 (POS)
        """
        df = self.load_human_labels()
        
        new_row = {
            "id": self._get_hash(item['text']),
            "text": item['text'],
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "model_prob_pos": prob_pos,
            "strategy_used": strategy
        }
        
        # Create DataFrame for the new row
        new_df = pd.DataFrame([new_row])
        
        # avoid appending if duplicate ID exists (though UI should prevent this)
        if not df.empty and new_row['id'] in df['id'].values:
           print(f"Duplicate label for id {new_row['id']}, skipping save.")
           return
           
        # Append and save
        updated_df = pd.concat([df, new_df], ignore_index=True)
        updated_df.to_csv(self.labels_path, index=False)
        return updated_df

    def get_labeled_ids(self):
        df = self.load_human_labels()
        if df.empty:
            return set()
        return set(df['id'].values)

    def get_combined_train_data(self):
        """
        Merges initial train dataset with human labels.
        Returns a HF dataset or list of dicts suitable for training.
        """
        # Convert initial train to list of dicts
        train_data = []
        if self.train_dataset:
            for item in self.train_dataset:
                train_data.append({
                    "text": item['text'], 
                    "label": item['label']
                })
                
        # Add human labels
        df = self.load_human_labels()
        if not df.empty:
            for _, row in df.iterrows():
                train_data.append({
                    "text": row['text'],
                    "label": int(row['label'])
                })
                
        return train_data

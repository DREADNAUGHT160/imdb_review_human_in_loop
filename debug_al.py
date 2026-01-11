import pandas as pd
import numpy as np
from datasets import Dataset

# Mock Model
class MockModel:
    def predict(self, texts, enable_dropout=False):
        return np.random.rand(len(texts))

# Import strategies
from src.active_learning import uncertainty_sampling, entropy_sampling

# Mock Data
data = Dataset.from_dict({'text': ['a', 'b', 'c', 'd', 'e'] * 10})
model = MockModel()

print("Testing Uncertainty Sampling...")
df_unc = uncertainty_sampling(model, data, k=5)
if df_unc is not None:
    print("Columns:", df_unc.columns.tolist())
    print(df_unc.head(2))

print("\nTesting Entropy Sampling...")
df_ent = entropy_sampling(model, data, k=5)
if df_ent is not None:
    print("Columns:", df_ent.columns.tolist())
    print(df_ent.head(2))

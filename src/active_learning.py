import numpy as np
import torch
from scipy.stats import entropy

def random_sampling(data, k=10):
    """Randomly selects k examples."""
    if len(data) <= k:
        return data.to_pandas()
    return data.shuffle().select(range(k)).to_pandas()

def uncertainty_sampling(model, data, k=10, low=0.4, high=0.6):
    """Selects samples where P(pos) is within [low, high]."""
    texts = data["text"]
    probs = model.predict(texts)
    
    # Filter by range
    indices = np.where((probs >= low) & (probs <= high))[0]
    
    if len(indices) == 0:
        return None
        
    selected_indices = indices[:k] if len(indices) > k else indices
    return data.select(selected_indices).to_pandas().assign(model_prob=probs[selected_indices])

def entropy_sampling(model, data, k=10):
    """
    Selects samples with highest entropy (closest to 0.5).
    Entropy H(p) = -p log p - (1-p) log (1-p)
    Maximized when p = 0.5.
    """
    texts = data["text"]
    probs = model.predict(texts)
    
    # Entropy calculation for binary classification
    # Avoid log(0)
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    entropies = -probs_clipped * np.log(probs_clipped) - (1 - probs_clipped) * np.log(1 - probs_clipped)
    
    # Get top K indices with highest entropy
    top_k_indices = np.argsort(entropies)[-k:][::-1]
    
    return data.select(top_k_indices).to_pandas().assign(model_prob=probs[top_k_indices], entropy=entropies[top_k_indices])

def disagreement_sampling(model, data, k=10, n_iter=5):
    """
    Uses MC Dropout to find samples with high disagreement (variance) in predictions.
    """
    texts = data["text"]
    
    # Run multiple forward passes with dropout enabled
    # We need to manually batch this inside the strategy or rely on model.predict handling batching.
    # To keep it simple, we loop n times.
    
    predictions = []
    print(f"Running MC Dropout disagreement sampling ({n_iter} iterations)...")
    
    for _ in range(n_iter):
        probs = model.predict(texts, enable_dropout=True)
        predictions.append(probs)
        
    predictions = np.array(predictions) # Shape: (n_iter, n_samples)
    
    # Calculate variance across iterations for each sample
    variances = np.var(predictions, axis=0)
    
    # Top K variance
    top_k_indices = np.argsort(variances)[-k:][::-1]
    
    # Get mean probability for display
    mean_probs = np.mean(predictions, axis=0)
    
    return data.select(top_k_indices).to_pandas().assign(
        model_prob=mean_probs[top_k_indices], 
        disagreement_var=variances[top_k_indices]
    )

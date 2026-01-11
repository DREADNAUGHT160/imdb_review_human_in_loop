import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import namedtuple

# Mock EvalPrediction object
EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])

def compute_metrics(pred):
    labels = pred.label_ids
    # Mock logits where argmax gives the prediction
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Case 1: Perfect predictions
print("Test Case 1: Perfect Predictions")
labels1 = np.array([0, 1, 0, 1])
# Logits that yield [0, 1, 0, 1]
logits1 = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]) 
metrics1 = compute_metrics(EvalPrediction(predictions=logits1, label_ids=labels1))
print(metrics1)

# Case 2: All wrong
print("\nTest Case 2: All Wrong")
labels2 = np.array([0, 1, 0, 1])
# Logits that yield [1, 0, 1, 0]
logits2 = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
metrics2 = compute_metrics(EvalPrediction(predictions=logits2, label_ids=labels2))
print(metrics2)

# Case 3: Zero Division (No positive labels in ground truth)
print("\nTest Case 3: No positive labels")
labels3 = np.array([0, 0, 0, 0])
# Predictions mix of 0 and 1
logits3 = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]) # Preds: [0, 0, 1, 0]
metrics3 = compute_metrics(EvalPrediction(predictions=logits3, label_ids=labels3))
print(metrics3)

# Case 4: No positive predictions (Recall/Precision undefined?)
print("\nTest Case 4: No positive predictions")
labels4 = np.array([1, 1, 1, 1])
logits4 = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]) # Preds: [0, 0, 0, 0]
metrics4 = compute_metrics(EvalPrediction(predictions=logits4, label_ids=labels4))
print(metrics4)

# Model Architecture & Specifications

## 1. Model Architecture
We use **DistilBERT**, a smaller, faster, cheaper, and lighter version of BERT.

*   **Model Name**: `distilbert-base-uncased-finetuned-sst-2-english`
*   **Source**: Hugging Face Hub (originally trained by Microsoft/Hugging Face).
*   **Base Architecture**: Transformer (Encoder-only).
*   **Parameters**: ~66 Million (compared to BERT-Base's 110M).
*   **Tokenizer**: WordPiece tokenizer (uncased).
*   **Input**: Raw text (truncated/padded to `max_length`).
*   **Output**: 2 classes (POSITIVE, NEGATIVE).

## 2. Hyperparameters
These settings control *how* the model learns. They are defined in `src/model_manager.py`.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `2e-5` (0.00002) | Very small steps. We don't want to break the pre-trained knowledge. |
| **Batch Size** | `16` | Number of reviews processed at once. Fits well on standard GPUs. |
| **Epochs** | `1` (Default) | User-adjustable (1-10) via UI. One pass through the entire dataset. |
| **Max Sequence Length** | `128` | Longer text is cut off. 128 tokens captures most IMDB reviews. |
| **Optimizer** | `AdamW` | Standard optimizer for Transformers (Adaptive Moment Estimation with Weight Decay). |
| **Loss Function** | `CrossEntropyLoss` | Standard loss for classification. |

## 3. Training Process
### A. Baseline Training
1.  **Input**: Initial `labeled_train` dataset (~2,000 examples).
2.  **Process**: Fine-tuning the pre-trained DistilBERT weights.
3.  **Goal**: adapt the generic English model to the specific style of IMDb movie reviews.

### B. Active Learning Retraining
1.  **Input**: `labeled_train` + `human_labels` (Accumulated).
2.  **Process**: **Incremental Fine-tuning**. We load the saved weights from `data/model` and train for another 1-10 epochs.
3.  **Goal**: Fix specific "blind spots" discovered by the Active Learning sampling.

## 4. Hardware Acceleration
*   **Device**: Automatically detects **CUDA** (NVIDIA), **MPS** (Mac), or **CPU**.
*   **Precision**: Uses `float32` (standard).
*   **Speed**: On an NVIDIA RTX 4050, training 1 epoch of 2,000 samples takes ~10-15 seconds.

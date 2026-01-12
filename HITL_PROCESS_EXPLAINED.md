# Human-in-the-Loop (HITL) Sentiment Analysis Process

This document explains the end-to-end workflow of the HITL system, how data is selected for human review, and how the model training works.

## 1. The Big Picture
The goal of this system is to train a high-quality Sentiment Analysis model (Positive vs. Negative) using a **Human-in-the-Loop** approach. Instead of randomly labeling thousands of examples, we use **Active Learning** to intelligently select the *most confusing* or *informative* examples for you to label. This makes the model learn faster with fewer human labels.

---

## 2. Step-by-Step User Guide

This section explains how to use the web interface, organized like a presentation for clarity.

### Phase 1: Setup & Initialization
*   **Action**: Look at the **Sidebar** on the left.
*   **1. Load Data**:
    *   *What to do*: Set "Initial Train Size" (default 2000) and click **"Load/Reset Data"**.
    *   *Why*: This splits the IMDb dataset. 2,000 reviews become your "Labeled Training Set" (what the model learns from first). The remaining ~23,000 become the "Unlabeled Pool" (what the model will query you about later).
*   **2. Initialize Model**:
    *   *What to do*: Set "Max Sequence Length" (default 128) and click **"Initialize Model"**.
    *   *Why*: This loads the `DistilBERT` model into memory and moves it to the GPU (if available). It prepares the brain of the system.

### Phase 2: Establish Baseline
*   **3. Train Baseline**:
    *   *What to do*: Under "Training" in the sidebar, click **"Train Baseline"**.
    *   *Why*: The model needs a starting point. It trains *only* on the initial 2,000 labeled examples.
    *   *Outcome*: You will see an initial Accuracy score (e.g., ~82%) in the top metrics panel.

### Phase 3: The Active Learning Loop (The Core Task)
*   **4. Select Strategy**:
    *   *What to do*: Under "Active Learning" in the sidebar, choose a strategy (e.g., **"Uncertainty"**) and set "Queue Size" (e.g., 10).
    *   *Why*: This decides *how* the model finds confusing data. "Uncertainty" looks for reviews where it is 50/50 confused. "Entropy" looks for maximum information.
*   **5. Build Queue**:
    *   *What to do*: Click **"Build Queue"**.
    *   *Why*: The model scans the 23,000 unlabeled reviews, scores them, and picks the top 10 hardest ones. It moves them to your **Review Queue**.

### Phase 4: Human Labeling
*   **6. Label the Queue**:
    *   *What to do*: Look at the main center panel. Read the review text.
    *   *Insight*: Check "Model P(POS)". If it says 0.50, the model has NO idea if this is positive or negative.
    *   *Action*: Click **"POSITIVE (1)"** or **"NEGATIVE (0)"** based on your judgment.
    *   *Why*: You are teaching the model. Every label you provide resolves a confusion the model had.

### Phase 5: Improvement
*   **7. Retrain**:
    *   *What to do*: After labeling the queue (10 items), go to the sidebar and click **"Retrain with Human Labels"**.
    *   *Why*: The model updates its internal weights. It learns from its original data *PLUS* the 10 difficult cases you just solved.
    *   *Outcome*: Watch the **Loss Curve** (in the "Visualizations" tab) go down and the **Accuracy** go up. The model is getting smarter!

---

## 3. Technical Execution Flow

For advanced users or developers, here is what happens under the hood when you click those buttons:

### A. Initialization (`src/model_manager.py`)
1.  **Device Selection**: The system checks for CUDA (NVIDIA), MPS (Mac), or CPU.
2.  **Model Loading**:
    *   It looks for a local checkpoint in `data/model`.
    *   If not found, it downloads `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face.
    *   **Tokenizer**: Loads the matching AutoTokenizer.

### B. Training Loop (`SentinmentModel.train`)
When you click **"Train"** or **"Retrain"**:
1.  **Data Preparation**:
    *   Converts text to tokens (input_ids, attention_mask) using `max_length` (padding/truncation).
    *   Creates a PyTorch `Dataset`.
2.  **Trainer Setup**:
    *   Uses Hugging Face `Trainer`.
    *   **Batch Size**: 16 (default).
    *   **Learning Rate**: 2e-5.
    *   **Optimizer**: AdamW.
3.  **Execution**:
    *   Runs for $N$ epochs.
    *   Computes Loss (CrossEntropy) and updates weights via Backpropagation.
4.  **Serialization**: Saves the fine-tuned model to `data/model` (safetensors format).

### C. Inference Loop (`SentimentModel.predict`)
When the model scores the **Unlabeled Pool** for Active Learning:
1.  **Batching**: It processes 23,000 items in batches of 32 to avoid Memory (OOM) errors.
2.  **Forward Pass**: 
    *   `no_grad()` context is used (faster, no training).
    *   **Logits**: The raw output scores from the final layer.
3.  **Probability**:
    *   Apply `Softmax(logits)` to get probabilities [0.0, 1.0].
    *   We extract the probability of Class 1 (Positive) for scoring.

---

## 4. Training vs. Testing
*   **Training**: Happens on the `labeled_train` + `human_labels`. The model aligns its weights to minimize error on these examples.
*   **Testing**: Happens on the `test_set` (validation). This data is **never** shown to the model during training. It acts as an unbiased exam.
    *   *Example*: If training accuracy is 99% but test accuracy is 80%, the model is "overfitting" (memorizing).
    *   *Goal*: We want the **Test Accuracy** (shown in the dashboard) to go up.

---

## 4. Work Done & Improvements
Here is a summary of the technical enhancements made to this project:

### Phase 1: Foundation
*   **Model Pipeline**: Built the class `SentimentModel` using Hugging Face Transformers (`distilbert`).
*   **App UI**: Created the Streamlit interface for labeling and interaction.

### Phase 2: Performance
*   **GPU Support**: Enabled **NVIDIA RTX 4050 (CUDA)** support. Training time dropped from minutes to seconds.
*   **Optimization**: Fixed `numpy` compatibility issues and tokenizer deadlocks on Windows.

### Phase 3: Analytics
*   **Detailed Metrics**: Added tracking for Accuracy, Precision, Recall, and F1-Score.
*   **Visualizations**:
    *   **Metrics History**: Line charts showing how your labeling impacts performance over time.
    *   **Confusion Matrix**: A heatmap showing exactly where the model confuses Positive vs. Negative.

### Phase 4: Deployment
*   **GitHub**: The entire codebase is version-controlled and pushed to your repository.

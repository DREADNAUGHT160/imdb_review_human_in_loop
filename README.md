# HITL Sentiment Analysis System

A human-in-the-loop sentiment analysis system for IMDb reviews using BERT and Active Learning.

## Features
- **Fine-tuned BERT**: Starts from `distilbert-base-uncased-finetuned-sst-2-english`.
- **Active Learning**: Strategies including Uncertainty, Entropy, and Disagreement (MC Dropout).
- **Interactive UI**: Streamlit-based interface for labeling and model management.
- **Efficient**: Caching for dataset/models, GPU support.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Workflow
1. **Load Data**: Adjust the training subset size in the sidebar and click "Load/Reset Data".
2. **Train Baseline**: Click "Train Baseline" to fine-tune the model on the initial subset.
3. **Build Queue**: Select a sampling strategy (e.g., Uncertainty) and click "Build Queue".
4. **Label**: Review the text and predicted probabilities. Click Positive/Negative/Skip.
5. **Retrain**: Once you've labeled some examples, click "Retrain with Human Labels" to update the model.

## Common Failure Modes & Fixes

- **OutOfMemory (OOM)**:
    - Reduce `batch_size` in `src/model_manager.py` (default 16).
    - Reduce `Max Sequence Length` in sidebar.
- **Slow Training (CPU)**:
    - Reduce `Initial Train Size`.
    - Install PyTorch with CUDA support if you have an NVIDIA GPU.
- **Tokenizer Warnings**:
    - The model truncates long reviews to 512 tokens. This is normal for BERT.

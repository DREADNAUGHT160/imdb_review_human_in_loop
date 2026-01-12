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

## Deployment (Docker)

You can containerize this application for easy deployment.

1.  **Build the Image**:
    ```bash
    docker build -t hitl-sentiment .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8501:8501 hitl-sentiment
    ```

    *   Access the app at `http://localhost:8501`.
    *   **Note**: The container runs on CPU by default. To persist data or models, mount the `/app/data` volume:
        ```bash
        docker run -p 8501:8501 -v $(pwd)/data:/app/data hitl-sentiment
        ```

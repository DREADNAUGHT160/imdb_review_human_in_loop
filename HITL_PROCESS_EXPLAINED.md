# Human-in-the-Loop (HITL) Sentiment Analysis Process

This document explains the end-to-end workflow of the HITL system, how data is selected for human review, and how the model training works.

## 1. The Big Picture
The goal of this system is to train a high-quality Sentiment Analysis model (Positive vs. Negative) using a **Human-in-the-Loop** approach. Instead of randomly labeling thousands of examples, we use **Active Learning** to intelligently select the *most confusing* or *informative* examples for you to label. This makes the model learn faster with fewer human labels.

---

## 2. The Workflow Steps

### Step 1: Initial Data Loading
*   **Source**: We load the **IMDb** dataset (movie reviews).
*   **Split**:
    *   **labeled_train**: A small initial set (e.g., 2,000 examples) that already has labels (0 or 1).
    *   **unlabeled_pool**: A large pool of reviews (~20,000+) treated as "unknown" (we hide the labels).
    *   **test_set**: A separate set of examples used ONLY for testing accuracy. The model never trains on this.

### Step 2: Baseline Training
*   **Action**: We train the `DistilBERT` model on the small `labeled_train` set.
*   **Outcome**: The model learns basic sentiment patterns but will likely be uncertain about complex or subtle reviews.
*   **Metrics**: We measure Accuracy, F1-Score, and show a Confusion Matrix on the `test_set` to see how well it started.

### 3. Build Queue (Active Learning Sampling)
When you click **"Build Queue"**:
1.  **Sampling**: The system looks at the **Unlabeled Pool** (~23,000 items).
2.  **Scoring**: It uses the selected strategy (e.g., *Uncertainty*) to score these items.
    *   *Uncertainty*: Finds items where the model is roughly 50/50 (0.4 to 0.6 probability).
    *   *Entropy*: Finds items with the highest information content (most confusing).
    *   *Disagreement*: Run the model multiple times with dropout and pick items where predictions fluctuate.
3.  **Filtration**: It checks if you've already labeled these items to avoid duplicates.
4.  **Selection**: It picks the top $K$ items (e.g., 10) and puts them in your **Review Queue**.
5.  **Display**: The "Queue Overview" table shows you the text and the specific score (e.g., `model_prob` or `entropy`) so you know *why* it was picked.
This is the core "Human-in-the-Loop" part. We don't just pick random reviews. We use **Strategies** to find what the model *doesn't* know.

#### How Selection Works:
1.  **Sampling**: We take a chunk of the `unlabeled_pool`.
2.  **Prediction**: The model predicts the sentiment for these unlabeled examples.
3.  **Ranking**: We score each example based on the chosen strategy:

    *   **Uncertainty Sampling**: We pick examples where the model's confidence is lowest (e.g., 51% Positive, 49% Negative). These are the "hardest" questions for the model.
    *   **Entropy Sampling**: Similar to uncertainty, but mathematically measures "chaos" or information density. High entropy = high confusion.
    *   **Disagreement**: We use "Monte Carlo Dropout". We run the model multiple times on the same text with slight random noise. If the predictions vary wildly (e.g., Run 1 says Positive, Run 2 says Negative), the model is unstable and needs help.
    *   **Random**: Purely random selection (used as a control baseline).

4.  **Queue**: The top 10 (or $K$) most "valuable" examples are put into your **Review Queue**.

### Step 4: Human Labeling
*   **Action**: You, the human, see the text of these difficult examples.
*   **Insight**: The UI shows *why* it was picked (e.g., "Model Confidence: 50.1%").
*   **Labeling**: You click **Positive** or **Negative**.
*   **Result**: The example is moved from the `unlabeled_pool` to the `human_labels` list.

### 5. Retraining (Refining the Model)
**Crucial Concept**: We do **NOT** delete the old model and start over. 
*   **Continuous Learning**: We load the **existing** model (the one that already knows some sentiment) and continue training it.
*   **Data**: We feed it the original data + your new 10 (or more) hard examples.
*   **Result**: The model gently adjusts its internal "weights" to fix its mistakes on the hard examples while keeping what it already knew. It gets smarter, step by step.

---

## 3. Training vs. Testing
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

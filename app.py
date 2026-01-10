import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

# Disable tokenizer parallelism to prevent deadlocks on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data_manager import DataManager
from src.model_manager import SentimentModel
from src.active_learning import random_sampling, uncertainty_sampling, entropy_sampling, disagreement_sampling

# --- Page Config ---
st.set_page_config(page_title="HITL Sentiment Lab", layout="wide")

# --- Session State Initialization ---
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'model' not in st.session_state:
    st.session_state.model = None # Lazy load
if 'review_queue' not in st.session_state:
    st.session_state.review_queue = [] # List of dicts
if 'queue_index' not in st.session_state:
    st.session_state.queue_index = 0
if 'train_metrics' not in st.session_state:
    st.session_state.train_metrics = {}

# --- Helper Functions ---
def load_datasets(train_size):
    with st.spinner("Loading Dataset..."):
        train, test, pool = st.session_state.data_manager.load_imdb(train_subset_size=train_size)
        st.session_state.train_dataset = train
        st.session_state.test_dataset = test
        st.session_state.unlabeled_pool = pool
        st.success("Datasets loaded!")

def init_model(max_length):
    if st.session_state.model is None or st.session_state.model.max_length != max_length:
        with st.spinner("Initializing Model..."):
            st.session_state.model = SentimentModel(max_length=max_length)
            # Try to load existing checkpoint if available
            if os.path.exists("data/model"):
                 st.session_state.model.load("data/model")
            st.success("Model initialized.")

def train_baseline():
    if not st.session_state.data_manager.train_dataset:
         st.error("Load data first!")
         return
    
    with st.spinner("Training Baseline..."):
        # Initial training on just the subset
        trainer = st.session_state.model.train(
             start_model_path=None,
             train_data=st.session_state.train_dataset,
             val_data=st.session_state.test_dataset.select(range(500)), # tiny val set for speed
             output_dir="data/checkpoints/baseline"
        )
        st.session_state.model.save("data/model")
        
        # Eval
        eval_metrics = trainer.evaluate()
        st.session_state.train_metrics = eval_metrics
        st.success(f"Baseline Training Complete! Accuracy: {eval_metrics.get('eval_accuracy', 'N/A')}")
        
        # Confusion Matrix
        preds = st.session_state.model.predict(st.session_state.test_dataset['text'])
        probs = preds
        y_pred = (probs > 0.5).astype(int)
        y_true = st.session_state.test_dataset['label']
        cm = confusion_matrix(y_true, y_pred)
        st.session_state.confusion_matrix = cm

def retrain():
    with st.spinner("Retraining with Human Labels..."):
        # Combine data
        combined_data = st.session_state.data_manager.get_combined_train_data()
        
        trainer = st.session_state.model.train(
             start_model_path="data/model",
             train_data=combined_data,
             val_data=st.session_state.test_dataset.select(range(500)),
             output_dir="data/checkpoints/retrain"
        )
        st.session_state.model.save("data/model")
        
        eval_metrics = trainer.evaluate()
        st.session_state.train_metrics = eval_metrics
        st.success(f"Retraining Complete! Accuracy: {eval_metrics.get('eval_accuracy', 'N/A')}")

def build_queue(strategy, k, low, high):
    if not st.session_state.model:
        st.error("Model not initialized.")
        return
    if not st.session_state.unlabeled_pool:
        st.error("No unlabeled pool.")
        return

    # Filter out already labeled IDs
    labeled_ids = st.session_state.data_manager.get_labeled_ids()
    # In a real app, we'd filter the HF dataset more efficiently. 
    # Here we just blindly sample from pool and check IDs, or filter pool first if small enough.
    # Since pool is ~20k, we can't filter all of them every time easily without iterating.
    # We will sample 2*k candidates using the strategy, filter duplicates, and take top k.
    
    # We'll work with a subset of pool to keep it fast
    pool_subset = st.session_state.unlabeled_pool.shuffle().select(range(min(2000, len(st.session_state.unlabeled_pool))))
    
    with st.spinner(f"Sampling with {strategy}..."):
        if strategy == "Random":
            samples = random_sampling(pool_subset, k=k*2)
        elif strategy == "Uncertainty":
            samples = uncertainty_sampling(st.session_state.model, pool_subset, k=k*2, low=low, high=high)
        elif strategy == "Entropy":
            samples = entropy_sampling(st.session_state.model, pool_subset, k=k*2)
        elif strategy == "Disagreement":
            samples = disagreement_sampling(st.session_state.model, pool_subset, k=k*2)
            
        if samples is None or samples.empty:
            st.warning("No samples found matching criteria.")
            return

        # Filter duplicates
        samples['hash'] = samples['text'].apply(st.session_state.data_manager._get_hash)
        samples = samples[~samples['hash'].isin(labeled_ids)]
        
        if samples.empty:
             st.warning("All sampled items were already labeled.")
             return
             
        # Take top k
        final_samples = samples.head(k)
        
        # Convert to list of dicts for queue
        st.session_state.review_queue = final_samples.to_dict('records')
        st.session_state.queue_index = 0
        st.session_state.current_strategy = strategy
        st.success(f"Queue built with {len(st.session_state.review_queue)} items.")


# --- Sidebar ---
st.sidebar.title("Configuration")
train_size = st.sidebar.slider("Initial Train Size", 2000, 25000, 2000, step=1000)
st.sidebar.button("Load/Reset Data", on_click=load_datasets, args=(train_size,))

max_length = st.sidebar.slider("Max Sequence Length", 128, 512, 128)
if st.sidebar.button("Initialize Model"):
    init_model(max_length)
    
st.sidebar.divider()
st.sidebar.subheader("Active Learning")
strategy = st.sidebar.selectbox("Strategy", ["Random", "Uncertainty", "Entropy", "Disagreement"])
queue_size = st.sidebar.slider("Queue Size", 5, 50, 10)

if strategy == "Uncertainty":
    col1, col2 = st.sidebar.columns(2)
    unc_low = col1.slider("Low Prob", 0.0, 0.5, 0.4)
    unc_high = col2.slider("High Prob", 0.5, 1.0, 0.6)
else:
    unc_low, unc_high = 0.4, 0.6

st.sidebar.button("Build Queue", on_click=build_queue, args=(strategy, queue_size, unc_low, unc_high))

st.sidebar.divider()
st.sidebar.subheader("Training")
st.sidebar.button("Train Baseline", on_click=train_baseline)
st.sidebar.button("Retrain with Human Labels", on_click=retrain)


# --- Main Area ---
st.title("Human-in-the-Loop Sentiment Labeling")

# Metrics
st.markdown("### Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acc", f"{st.session_state.train_metrics.get('eval_accuracy', 0):.2%}")
col2.metric("F1", f"{st.session_state.train_metrics.get('eval_f1', 0):.4f}")
col3.metric("Prec", f"{st.session_state.train_metrics.get('eval_precision', 0):.4f}")
col4.metric("Rec", f"{st.session_state.train_metrics.get('eval_recall', 0):.4f}")

c1, c2 = st.columns(2)
c1.metric("Human Labels", len(st.session_state.data_manager.load_human_labels()))
c2.metric("Queue Remaining", len(st.session_state.review_queue) - st.session_state.queue_index if st.session_state.review_queue else 0)

if 'confusion_matrix' in st.session_state and st.session_state.confusion_matrix is not None:
    st.markdown("### Confusion Matrix")
    cm_df = pd.DataFrame(st.session_state.confusion_matrix, index=["Actual Neg", "Actual Pos"], columns=["Pred Neg", "Pred Pos"])
    st.table(cm_df)

# Labeling Panel
if st.session_state.review_queue and st.session_state.queue_index < len(st.session_state.review_queue):
    item = st.session_state.review_queue[st.session_state.queue_index]
    
    st.markdown("### Review Text")
    st.info(item['text'])
    
    # Model info
    prob_pos = item.get('model_prob', 0.5)
    # If prob isn't in item (e.g. random sampling), predict it now
    if 'model_prob' not in item and st.session_state.model:
         prob_pos = float(st.session_state.model.predict([item['text']])[0])
         
    # Ensure float and range [0, 1] for st.progress
    prob_pos = float(prob_pos)
    prob_pos = max(0.0, min(1.0, prob_pos))
         
    entropy_val = item.get('entropy', - (prob_pos * np.log(prob_pos+1e-9) + (1-prob_pos)*np.log(1-prob_pos+1e-9))) # approx
    
    # Visualization
    c1, c2, c3 = st.columns(3)
    c1.metric("Model P(POS)", f"{prob_pos:.4f}")
    c2.metric("Entropy", f"{entropy_val:.4f}")
    if 'disagreement_var' in item:
        c3.metric("Disagreement (Var)", f"{item['disagreement_var']:.4f}")

    with st.expander("Why uncertain?", expanded=True):
        st.write("The model is uncertain because the probability is close to 0.5 or there is high disagreement between dropout heads.")
        # Simple highlight of sentiment words could go here (omitted for brevity/stability)
        if prob_pos > 0.5:
             st.progress(prob_pos, text="Positive Confidence")
        else:
             st.progress(1 - prob_pos, text="Negative Confidence")

    # Buttons
    b1, b2, b3 = st.columns([1, 1, 1])
    
    def save_and_next(label):
        st.session_state.data_manager.save_human_label(
            item, 
            label, 
            prob_pos, 
            st.session_state.current_strategy
        )
        st.session_state.queue_index += 1

    def skip():
        st.session_state.queue_index += 1

    b1.button("NEGATIVE (0)", type="primary", use_container_width=True, on_click=save_and_next, args=(0,))
    b2.button("POSITIVE (1)", type="primary", use_container_width=True, on_click=save_and_next, args=(1,))
    b3.button("Skip", use_container_width=True, on_click=skip)

elif st.session_state.review_queue:
    st.success("Queue completed! Build a new one.")
else:
    st.info("Queue is empty. Use the sidebar to Build Queue.")

# Recent Labels Table
st.divider()
st.subheader("Recent Human Labels")
df_labels = st.session_state.data_manager.load_human_labels()
if not df_labels.empty:
    st.dataframe(df_labels.tail(20).sort_index(ascending=False), use_container_width=True)

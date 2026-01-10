
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'hitl_sentiment'))

# Disable tokenizer parallelism to prevent deadlocks on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data_manager import DataManager
from src.model_manager import SentimentModel
from src.active_learning import uncertainty_sampling

def run_simulation():
    print("--- 1. Initialize Components ---")
    data_mgr = DataManager(data_dir="hitl_sentiment/data")
    train, test, pool = data_mgr.load_imdb(train_subset_size=50) # Tiny subset for speed
    print(f"Data loaded: Train={len(train)}, Pool={len(pool)}")

    model = SentimentModel(device="cpu") # Force CPU for simple verification
    
    print("\n--- 2. Train Baseline (Simulated) ---")
    model.train(None, train, val_data=test.select(range(10)), epochs=1) 
    # train on 50, eval on 10, 1 epoch
    
    print("\n--- 3. Active Learning Sampling ---")
    # Uncertainty sampling
    samples = uncertainty_sampling(model, pool.select(range(100)), k=5)
    if samples is not None:
        print(f"Sampled {len(samples)} items.")
        print(samples[['text', 'model_prob']].head(2))
    else:
        print("No samples found in range (might happen with small pool/untrained model).")

    print("\n--- 4. Labeling ---")
    if samples is not None and not samples.empty:
        item = samples.iloc[0].to_dict()
        data_mgr.save_human_label(item, 1, 0.9, "simulation")
        print("Saved 1 human label.")
        
    print("\n--- 5. Retraining ---")
    combined = data_mgr.get_combined_train_data()
    print(f"Combined data size: {len(combined)}")
    model.train("hitl_sentiment/data/model", combined, output_dir="hitl_sentiment/data/checkpoints/retrain", epochs=1)

    print("\n--- SUCCESS: Simulation Completed ---")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        with open("error.log", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()

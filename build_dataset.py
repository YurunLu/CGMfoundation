import argparse
import os
from pathlib import Path
import glob
import yaml
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from multiprocessing import cpu_count
from utils.data_cleaning import align_cgm_dataframe_columns, clean_cgm_array, clean_cgm_dataframe

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fill_nan_locf(x):
    s = pd.Series(x)
    s = s.ffill().fillna(5.5)
    return s.to_numpy(dtype=np.float32)

def process_patient(example):
    """ 
    Clean and impute full sequences; extract patches on the fly during training.
    """
    seq_raw = clean_cgm_array(example['seq_raw'])
    orig_mask = (~np.isnan(seq_raw)).astype(np.float32)
    
    seq_filled = fill_nan_locf(seq_raw)
    
    return {
        "seq_filled": seq_filled,
        "orig_mask": orig_mask
    }

def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    csv_dir = config['data']['csv_dir']
    hf_dir = config['data']['hf_dataset_dir']
    
    print(f"Starting sequence preprocessing...")
    
    all_csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    max_len = config['model']['max_seq_len']
    
    all_data_list = []
    
    print("1. Reading CSV files...")
    for file_path in all_csv_files:
        df = pd.read_csv(file_path, index_col=0)
        df = align_cgm_dataframe_columns(df, max_len)
        df = clean_cgm_dataframe(df)
        
        # Discard patient sequences with fewer than 80% valid observations.
        valid_df = df[df.notna().sum(axis=1) >= max_len * 0.8]
        
        for row in valid_df.to_numpy().astype(np.float32):
            all_data_list.append({"seq_raw": row})
            
    total_patients = len(all_data_list)
    print(f"Reading complete. Valid patient sequences: {total_patients}")

    print("2. Shuffling and splitting the dataset...")
    np.random.seed(config['training'].get('seed', 42))
    np.random.shuffle(all_data_list)
    
    train_end = int(config['data']['train_ratio'] * total_patients)
    val_end = train_end + int(config['data']['val_ratio'] * total_patients)
    
    raw_datasets = DatasetDict({
        "train": Dataset.from_list(all_data_list[:train_end]),
        "validation": Dataset.from_list(all_data_list[train_end:val_end]),
        "test": Dataset.from_list(all_data_list[val_end:])
    })

    num_proc = max(1, cpu_count() - 2)
    print(f"3. Running parallel CPU imputation ({num_proc} processes)")
    
    processed_datasets = DatasetDict()
    empty_features = Features({
        "seq_filled": Sequence(Value("float32")),
        "orig_mask": Sequence(Value("float32")),
    })
    for split, dataset in raw_datasets.items():
        if len(dataset) == 0:
            processed_datasets[split] = Dataset.from_dict(
                {"seq_filled": [], "orig_mask": []}, features=empty_features
            )
        else:
            processed_datasets[split] = dataset.map(
                process_patient,
                num_proc=num_proc,
                remove_columns=["seq_raw"],
                desc="Cleaning sequences and generating masks..."
            )

    print(f"4. Saving preprocessed sequences to disk ({hf_dir})...")
    Path(hf_dir).parent.mkdir(parents=True, exist_ok=True)
    processed_datasets.save_to_disk(hf_dir)
    print("Sequence preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the preprocessed CGM dataset")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to the dataset configuration YAML"
    )
    args = parser.parse_args()
    main(args.config)

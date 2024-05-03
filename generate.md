## Steps to generate synthetic data

1. Create a CSV file of source audio samples to be used using `source_data/create_csv.py`
2. Use that to generate data.
    ```sh
    python scripts/generate_dual_resonance.py --csv ./source_data/v0.3.20240503.csv --version v8.0 --save_both --num_samples 5000
    ```
3. Create splits with the generated data.
    ```sh
    python scripts/create_splits.py --version v8.0 --source_data ./source_data/v0.3.20240503.csv 
    ```
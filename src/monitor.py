import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from data_ingestion import fetch_historical_data
from features import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

BASELINE_PATH = 'models/baseline.json'

def create_baseline(tickers, start_date=None, end_date=None, save_path=BASELINE_PATH):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    logger.info(f"Creating baseline from {start_date} to {end_date} for {tickers}")

    data = fetch_historical_data(tickers, start_date, end_date)
    baseline = {}

    for ticker in tickers:
        if ticker in data:
            df_features = prepare_features(data[ticker])
            feature_stats = {}

            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                           'future_return_1d', 'target_1d']

            # Filter to numeric columns only
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_cols if col not in exclude_cols]

            for feature in features:
                feature_data = df_features[feature].dropna()
                if len(feature_data) > 0:
                    feature_stats[feature] = {
                        'mean': float(feature_data.mean()),
                        'std': float(feature_data.std()),
                        'min': float(feature_data.min()),
                        'max': float(feature_data.max()),
                        'median': float(feature_data.median()),
                        'q1': float(feature_data.quantile(0.25)),
                        'q3': float(feature_data.quantile(0.75)),
                    }

            baseline[ticker] = {
                'feature_stats': feature_stats,
                'created_at': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'n_samples': len(df_features)
            }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(baseline, f, indent=4)

    logger.info(f"Baseline saved to {save_path}")
    return baseline

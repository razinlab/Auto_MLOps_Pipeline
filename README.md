# Automated ML Trading Pipeline with Champion/Challenger A/B Testing

End-to-end MLOps system for algorithmic trading with automated model training, versioning, deployment, and performance monitoring.

## Key Features
- **Automated data pipeline** for multi-stock ingestion and feature engineering (70+ technical indicators)
- **Rolling window backtesting** on 4+ years of historical data with XGBoost classifier
- **MLflow experiment tracking** with alias-based model registry (migrated from deprecated stages)
- **Champion/Challenger A/B testing** framework for production model validation
- **Docker containerization** with multi-service orchestration
- **Real-time monitoring dashboard** (Streamlit) for trade analytics and model performance

## Results
- **Accuracy**: 50.4% | **Precision**: 54.0% | **Recall**: 64.2%
- **804 trading days** backtested | **222 trades** executed

  Note: The model specs and results aren't the main focus here, more so creating the platform to deploy any model. As a result (if you check backtest.py) you will notice the model parameters are intentionally subpar for quicker training and debugging.

## Tech Stack
**ML/Data**: Python, XGBoost, NumPy, Pandas, scikit-learn  

**MLOps**: MLflow (experiment tracking, model registry), Docker, docker-compose  

**Monitoring**: Streamlit dashboard, SQLite trade logging

**APIs**: Alpaca Trading API

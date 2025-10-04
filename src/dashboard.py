import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

DB_PATH = 'trade_logs.db'

def load_trade_data():

    if not os.path.exists(DB_PATH):
        st.error(f"Database file {DB_PATH} not found. Please run the trading script first.")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)

    trades_query = """
    SELECT * FROM trades
    ORDER BY timestamp DESC
    """
    trades_df = pd.read_sql_query(trades_query, conn)

    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    conn.close()
    
    return trades_df

def load_performance_data():
    if not os.path.exists(DB_PATH):
        st.error(f"Database file {DB_PATH} not found. Please run the trading script first.")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)

    performance_query = """
    SELECT * FROM performance
    ORDER BY date ASC
    """
    performance_df = pd.read_sql_query(performance_query, conn)

    if not performance_df.empty:
        performance_df['date'] = pd.to_datetime(performance_df['date'])
    
    conn.close()
    
    return performance_df

def calculate_metrics(trades_df):
    if trades_df.empty:
        return {}

    metrics = {}
    
    for model_type in trades_df['model_type'].unique():
        model_trades = trades_df[trades_df['model_type'] == model_type]

        metrics[model_type] = {
            'total_trades': len(model_trades),
            'buy_trades': len(model_trades[model_trades['order_side'] == 'buy']),
            'sell_trades': len(model_trades[model_trades['order_side'] == 'sell']),
            'avg_quantity': model_trades['quantity'].mean(),
            'total_quantity': model_trades['quantity'].sum(),
            'last_trade': model_trades['timestamp'].max() if not model_trades.empty else None,
            'tickers_traded': model_trades['ticker'].nunique(),
            'unique_tickers': model_trades['ticker'].unique().tolist()
        }
    
    return metrics

def plot_trades_by_day(trades_df):
    if trades_df.empty:
        return None

    trades_df['date'] = trades_df['timestamp'].dt.date

    trades_by_day = trades_df.groupby(['date', 'model_type']).size().reset_index(name='count')

    fig = px.bar(
        trades_by_day,
        x='date',
        y='count',
        color='model_type',
        barmode='group',
        title='Number of Trades by Day',
        labels={'count': 'Number of Trades', 'date': 'Date', 'model_type': 'Model Type'}
    )
    
    return fig

def plot_cumulative_returns(performance_df):
    if performance_df.empty:
        return None

    fig = px.line(
        performance_df,
        x='date',
        y='cumulative_return',
        color='model_type',
        title='Cumulative Returns',
        labels={'cumulative_return': 'Cumulative Return', 'date': 'Date', 'model_type': 'Model Type'}
    )

    fig.add_shape(
        type='line',
        x0=performance_df['date'].min(),
        y0=0,
        x1=performance_df['date'].max(),
        y1=0,
        line=dict(color='gray', dash='dash')
    )
    
    return fig

def plot_daily_pnl(performance_df):
    if performance_df.empty:
        return None

    fig = px.bar(
        performance_df,
        x='date',
        y='pnl',
        color='model_type',
        barmode='group',
        title='Daily P&L',
        labels={'pnl': 'P&L ($)', 'date': 'Date', 'model_type': 'Model Type'}
    )

    fig.add_shape(
        type='line',
        x0=performance_df['date'].min(),
        y0=0,
        x1=performance_df['date'].max(),
        y1=0,
        line=dict(color='gray', dash='dash')
    )
    
    return fig

def plot_ticker_distribution(trades_df):

    if trades_df.empty:
        return None

    ticker_counts = trades_df.groupby(['ticker', 'model_type']).size().reset_index(name='count')

    fig = px.bar(
        ticker_counts,
        x='ticker',
        y='count',
        color='model_type',
        barmode='group',
        title='Trades by Ticker',
        labels={'count': 'Number of Trades', 'ticker': 'Ticker', 'model_type': 'Model Type'}
    )
    
    return fig

def main():

    st.set_page_config(
        page_title="Trading Strategy Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Trading Strategy Dashboard")
    st.write("Monitor the performance of Champion and Challenger models")

    trades_df = load_trade_data()
    performance_df = load_performance_data()
    
    if trades_df.empty:
        st.warning("No trade data available. Please run the trading script first.")
        return

    metrics = calculate_metrics(trades_df)

    st.header("Performance Metrics")
    
    if metrics:
        cols = st.columns(len(metrics))
        
        for i, (model_type, model_metrics) in enumerate(metrics.items()):
            with cols[i]:
                st.subheader(f"{model_type} Model")
                st.metric("Total Trades", model_metrics['total_trades'])
                st.metric("Buy Trades", model_metrics['buy_trades'])
                st.metric("Sell Trades", model_metrics['sell_trades'])
                st.metric("Tickers Traded", model_metrics['tickers_traded'])
                st.write(f"**Tickers:** {', '.join(model_metrics['unique_tickers'])}")
                
                if model_metrics['last_trade']:
                    st.write(f"**Last Trade:** {model_metrics['last_trade']}")

    st.header("Trading Activity")
    trades_by_day_fig = plot_trades_by_day(trades_df)
    if trades_by_day_fig:
        st.plotly_chart(trades_by_day_fig, use_container_width=True)

    ticker_dist_fig = plot_ticker_distribution(trades_df)
    if ticker_dist_fig:
        st.plotly_chart(ticker_dist_fig, use_container_width=True)

    if not performance_df.empty:
        st.header("Performance")

        cum_returns_fig = plot_cumulative_returns(performance_df)
        if cum_returns_fig:
            st.plotly_chart(cum_returns_fig, use_container_width=True)

        daily_pnl_fig = plot_daily_pnl(performance_df)
        if daily_pnl_fig:
            st.plotly_chart(daily_pnl_fig, use_container_width=True)

    st.header("Recent Trades")
    if not trades_df.empty:
        trades_df['formatted_time'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        display_cols = ['formatted_time', 'ticker', 'model_type', 'order_side', 'quantity', 'price', 'prediction']

        st.dataframe(trades_df[display_cols].head(20), use_container_width=True)

    if st.button("Refresh Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
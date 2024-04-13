import gradio as gr
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.cla import CLA
import cvxpy as cp
from main1 import *
import ast

# --- Gradio Interface ---
def gradio_interface(
    tickers: str = "MSFT, AMZN, KO, MA, COST, LUV, XOM, PFE, JPM, UNH, ACN, DIS, GILD, F, TSLA",
    period: str = "max",
    frequency: int = 252,
    weight_bounds_str: str = "None, None",
    sector_mapper_str: str = '{"MSFT": "Tech", "AMZN": "Tech", "KO": "Consumer Staples", "MA": "Financial Services", "COST": "Consumer Staples", "LUV": "Aerospace", "XOM": "Energy", "PFE": "Healthcare", "JPM": "Financial Services", "UNH": "Healthcare", "ACN": "Tech", "DIS": "Media", "GILD": "Healthcare", "F": "Auto", "TSLA": "Auto"}',
    sector_lower_str: str = '{"Consumer Staples": 0.1, "Tech": 0.05}',
    sector_upper_str: str = '{"Tech": 0.2, "Aerospace": 0.1, "Energy": 0.1, "Auto": 0.15}',
    objective_function: str = "max_sharpe",
    gamma: float = 0.0,
    target_volatility: float = 0.0,
    target_return: float = 0.0,
    market_neutral: bool = False,
    total_portfolio_value: int = 20000,
    short_ratio: float = 0.3,
    target_cvar: float = -0.025,
    max_big_tech_weight: float = 0.3,
):
    # --- Input Validation and Processing ---
    try:
        tickers = [t.strip() for t in tickers.upper().split(",")]  # Remove spaces
        weight_bounds = tuple(float(w) if w.strip().lower() != "none" else None for w in weight_bounds_str.split(","))
        sector_mapper = ast.literal_eval(sector_mapper_str)
        sector_lower = ast.literal_eval(sector_lower_str)
        sector_upper = ast.literal_eval(sector_upper_str)

        # Fetch historical prices
        prices = fetch_historical_prices(tickers, period=period)

        # Calculate expected returns and covariance matrix
        mu = calculate_expected_returns(prices)
        cov_matrix = calculate_sample_covariance(prices, frequency=frequency)

        # Portfolio optimization
        weights, performance = optimize_portfolio(prices, mu, cov_matrix, 
            weight_bounds=weight_bounds,
            sector_mapper=sector_mapper,
            sector_lower=sector_lower,
            sector_upper=sector_upper,
            objective_function=objective_function,
            gamma=gamma,
            target_volatility=target_volatility,
            target_return=target_return,
            market_neutral=market_neutral
        )

        # Discrete allocation
        latest_prices = prices.iloc[-1]
        alloc = allocate_portfolio(weights, latest_prices, total_portfolio_value=total_portfolio_value, short_ratio=short_ratio)

        # Sector allocation
        sector_allocation = map_sectors_to_tickers(tickers, sector_mapper, weights)

        # For compute_cvar
        if prices is None or mu is None:
            cvar_performance = ("Unable to compute CVaR: prices or expected returns are missing", None)
        else:
            returns = expected_returns.returns_from_prices(prices).dropna()
            cvar_performance = compute_cvar(prices, mu, returns, target_cvar=target_cvar)

        # For perform_cla_optimization
        if mu is None or cov_matrix is None:
            cla_performance = ("Unable to perform CLA optimization: expected returns or covariance matrix is missing", None)
        else:
            cla_performance = perform_cla_optimization(mu, cov_matrix)

        # Constraint optimization
        big_tech_indices = [tickers.index(ticker) for ticker in {"MSFT", "AMZN", "TSLA"}]
        if not tickers or not big_tech_indices:
            constraint_performance = ("Unable to perform constraint optimization: tickers or big tech indices are missing", None)
        else:
            constraint_performance = perform_constraint_optimization(mu, cov_matrix, tickers, big_tech_indices, max_big_tech_weight=max_big_tech_weight)

        # Efficient Frontier plot
        if constraint_performance is None:
            efficient_frontier_plot = None
        else:
            fig, ax = plt.subplots()
            plot_efficient_frontier_with_random_portfolios(constraint_performance, ax=ax)
            plt.close(fig)  
            efficient_frontier_plot = fig.get_children()[0]  # Access the plot

        # Formatting output
        performance_text = performance[0]
        cvar_performance_text = cvar_performance[0]
        cla_performance_text = cla_performance[0]
        constraint_performance_text = constraint_performance[0]

    except Exception as e:
        return f"Error processing inputs: {e}", None, None, None, None, None, None

    return [value if value is not None else "Error" for value in [performance_text, alloc.to_frame(), str(sector_allocation), cvar_performance_text, cla_performance_text, constraint_performance_text, efficient_frontier_plot]]


# --- Gradio Interface Definition ---
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Tickers (comma-separated)", value="MSFT, AMZN, KO, MA, COST, LUV, XOM, PFE, JPM, UNH, ACN, DIS, GILD, F, TSLA"),
        gr.Dropdown(label="Period", choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], value="max"),
        gr.Slider(minimum=1, maximum=365, value=252, step=1, label="Frequency"),
        gr.Textbox(label="Weight Bounds (comma-separated)", value="0.0, 0.3"),
        gr.Textbox(label="Sector Mapper (JSON format)", value='{"MSFT": "Tech", "AMZN": "Tech", "KO": "Consumer Staples", "MA": "Financial Services", "COST": "Consumer Staples", "LUV": "Aerospace", "XOM": "Energy", "PFE": "Healthcare", "JPM": "Financial Services", "UNH": "Healthcare", "ACN": "Tech", "DIS": "Media", "GILD": "Healthcare", "F": "Auto", "TSLA": "Auto"}'),
        gr.Textbox(label="Sector Lower Bounds (JSON format)", value='{"Consumer Staples": 0.1, "Tech": 0.05}'),
        gr.Textbox(label="Sector Upper Bounds (JSON format)", value='{"Tech": 0.2, "Aerospace": 0.1, "Energy": 0.1, "Auto": 0.15}'),
        gr.Dropdown(label="Objective Function", choices=["max_sharpe", "min_volatility", "efficient_risk", "efficient_return"], value="max_sharpe"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Gamma (L2 regularization)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Target Volatility"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Target Return"),
        gr.Checkbox(label="Market Neutral", value=False),
        gr.Number(label="Total Portfolio Value", value=20000),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Short Ratio"),
        gr.Slider(minimum=-0.1, maximum=0, value=-0.025, step=0.001, label="Target CVaR"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Max Big Tech Weight"),
    ],
    outputs=[
        gr.Textbox(label="Portfolio Performance"),
        gr.Dataframe(label="Discrete Allocation"),
        gr.Textbox(label="Sector Allocation"),
        gr.Textbox(label="CVaR Performance"),
        gr.Textbox(label="CLA Performance"),
        gr.Textbox(label="Constraint Optimization"),
        gr.Plot(label="Efficient Frontier")
    ],
    title="Portfolio Optimization App",
    description="This Gradio application allows you to optimize a portfolio using various techniques and constraints.",
)
iface.launch(share=True)




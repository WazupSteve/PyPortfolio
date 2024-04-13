import gradio as gr
import matplotlib.pyplot as plt
from main1 import *
import json


def gradio_interface(
    tickers: str,
    period: str,
    frequency: int,
    weight_bounds_str: str,
    sector_mapper_str: str,
    sector_lower_str: str,
    sector_upper_str: str,
    objective_function: str,
    gamma: float,
    target_volatility: float,
    target_return: float,
    market_neutral: bool,
    total_portfolio_value: int,
    short_ratio: float,
    target_cvar: float,
    max_big_tech_weight: float,
):
    tickers = [t.strip()
               for t in tickers.upper().split(",")]
    weight_bounds = tuple(float(w) if w.strip().lower(
    ) != "none" else None for w in weight_bounds_str.split(","))
    sector_mapper = json.loads(sector_mapper_str)
    sector_lower = json.loads(sector_lower_str)
    sector_upper = json.loads(sector_upper_str)

    prices = fetch_historical_prices(tickers, period=period)

    mu = calculate_expected_returns(prices)
    cov_matrix = calculate_sample_covariance(prices, frequency=frequency)

    weights, performance = optimize_portfolio(mu, cov_matrix,
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

    latest_prices = prices.iloc[-1]
    alloc = allocate_portfolio(
        weights, latest_prices, total_portfolio_value=total_portfolio_value, short_ratio=short_ratio)

    sector_allocation = str(map_sectors_to_tickers(
        tickers, sector_mapper, weights))

    if prices is None or mu is None:
        cvar_performance = (
            "Unable to compute CVaR: prices or expected returns are missing", None)
    else:
        cvar_performance = compute_cvar(
            prices, mu, target_cvar=target_cvar)

    if mu is None or cov_matrix is None:
        cla_performance = (
            "Unable to perform CLA optimization: expected returns or covariance matrix is missing", None)
    else:
        cla_performance = perform_cla_optimization(mu, cov_matrix)

    big_tech_indices = [tickers.index(ticker)
                        for ticker in {"MSFT", "AMZN", "TSLA"}]
    if not tickers or not big_tech_indices:
        constraint_performance = (
            "Unable to perform constraint optimization: tickers or big tech indices are missing", None)
    else:
        constraint_performance = perform_constraint_optimization(
            mu, cov_matrix, big_tech_indices, max_big_tech_weight=max_big_tech_weight)

    if constraint_performance is None:
        efficient_frontier_plot = None
    else:
        fig, ax = plt.subplots()
        plot_efficient_frontier_with_random_portfolios(
            constraint_performance, ax=ax)
        efficient_frontier_plot = fig

    performance_text = str(performance[0])
    cvar_performance_text = str(cvar_performance[0])
    cla_performance_text = str(cla_performance[0])
    constraint_performance_text = str(
        constraint_performance.portfolio_performance(verbose=True)[0])
    return performance_text, alloc.to_dict(), sector_allocation, cvar_performance_text, cla_performance_text, constraint_performance_text, efficient_frontier_plot


iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Tickers (comma-separated)",
                   value="MSFT, AMZN, KO, MA, COST, LUV, XOM, PFE, JPM, UNH, ACN, DIS, GILD, F, TSLA"),
        gr.Dropdown(label="Period", choices=[
                    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], value="max"),
        gr.Slider(minimum=1, maximum=365, value=252,
                  step=1, label="Frequency"),
        gr.Textbox(label="Weight Bounds (comma-separated)", value="0.0, 0.3"),
        gr.Textbox(label="Sector Mapper (JSON format)",
                   value='{"MSFT": "Tech", "AMZN": "Tech", "KO": "Consumer Staples", "MA": "Financial Services", "COST": "Consumer Staples", "LUV": "Aerospace", "XOM": "Energy", "PFE": "Healthcare", "JPM": "Financial Services", "UNH": "Healthcare", "ACN": "Tech", "DIS": "Media", "GILD": "Healthcare", "F": "Auto", "TSLA": "Auto"}'),
        gr.Textbox(label="Sector Lower Bounds (JSON format)",
                   value='{"Consumer Staples": 0.1, "Tech": 0.05}'),
        gr.Textbox(label="Sector Upper Bounds (JSON format)",
                   value='{"Tech": 0.2, "Aerospace": 0.1, "Energy": 0.1, "Auto": 0.15}'),
        gr.Dropdown(label="Objective Function", choices=[
                    "max_sharpe", "min_volatility", "efficient_risk", "efficient_return"], value="max_sharpe"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0,
                  step=0.01, label="Gamma (L2 regularization)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0,
                  step=0.01, label="Target Volatility"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.0,
                  step=0.01, label="Target Return"),
        gr.Checkbox(label="Market Neutral", value=False),
        gr.Number(label="Total Portfolio Value", value=20000),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.3,
                  step=0.01, label="Short Ratio"),
        gr.Slider(minimum=0, maximum=0.1, value=0.025,
                  step=0.001, label="Target CVaR"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.3,
                  step=0.01, label="Max Big Tech Weight"),
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
iface.launch(share=False)

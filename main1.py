import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier, DiscreteAllocation, objective_functions, plotting, EfficientCVaR
from pypfopt.cla import CLA
import cvxpy as cp
from pypfopt.discrete_allocation import DiscreteAllocation


def fetch_historical_prices(tickers, period="max"):
    """Fetch historical stock prices from Yahoo Finance."""
    ohlc = yf.download(tickers, period=period)
    prices = ohlc["Adj Close"].dropna(how="all")
    return prices


def calculate_sample_covariance(prices, frequency=252):
    """Calculate sample covariance matrix."""
    return risk_models.sample_cov(prices, frequency=frequency)


def calculate_expected_returns(prices):
    """Calculate expected returns using CAPM model."""
    return expected_returns.capm_return(prices)


def optimize_portfolio(mu, cov_matrix, weight_bounds=(None, None), sector_mapper=None, sector_lower=None, sector_upper=None, objective_function="default", gamma=None, target_volatility=None, target_return=None, market_neutral=False):
    """Optimize portfolio."""
    ef = EfficientFrontier(mu, cov_matrix, weight_bounds=weight_bounds)

    if sector_mapper:
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    match objective_function:
        case "default":
            pass
        case "max_sharpe":
            ef.max_sharpe()
        case "min_volatility":
            ef.min_volatility()
        case "efficient_risk":
            ef.efficient_risk(target_volatility=target_volatility)
        case "efficient_return":
            ef.efficient_return(target_return=target_return,
                                market_neutral=market_neutral)
        case _:
            raise ValueError(f'Objective cannot be {objective_function}')

    if gamma:
        ef.add_objective(objective_functions.L2_reg, gamma=gamma)

    weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True)
    return weights, performance


def allocate_portfolio(weights, latest_prices, total_portfolio_value, short_ratio=0.3):
    # Get the intersection of tickers between weights and latest_prices
    common_tickers = list(set(weights.keys()) &
                          set(latest_prices.dropna().index))

    filtered_weights = {ticker: weights[ticker] for ticker in common_tickers}
    filtered_prices = latest_prices[common_tickers]

    da = DiscreteAllocation(filtered_weights, filtered_prices,
                            total_portfolio_value=total_portfolio_value, short_ratio=short_ratio)
    alloc, leftover = da.lp_portfolio()
    print(f"Discrete allocation performed with ${leftover:.2f} leftover")

    alloc_series = pd.Series(alloc)

    alloc_series = alloc_series.reindex(weights.keys(), fill_value=0)

    return alloc_series


def map_sectors_to_tickers(tickers, sector_mapper, weights):
    """Map tickers to sectors."""
    sector_allocation = {}
    for ticker in tickers:
        sector = sector_mapper.get(ticker, "Unknown")
        if sector not in sector_allocation:
            sector_allocation[sector] = 0
        sector_allocation[sector] += weights[ticker]
    return sector_allocation


def compute_cvar(prices, mu, target_cvar=0.025):
    """Compute Conditional Value at Risk (CVaR)."""
    semicov = risk_models.semicovariance(prices, benchmark=0)
    ec = EfficientCVaR(mu, semicov)
    ec.efficient_risk(target_cvar=target_cvar)
    return ec.portfolio_performance(verbose=True)


def perform_cla_optimization(mu, cov_matrix):
    """Perform Critical Line Algorithm (CLA) optimization."""
    cla = CLA(mu, cov_matrix)
    cla.max_sharpe()
    return cla.portfolio_performance(verbose=True)


def perform_constraint_optimization(mu, cov_matrix, big_tech_indices, max_big_tech_weight=0.3):
    """Perform portfolio optimization with constraints."""
    ef_sharpe = EfficientFrontier(mu, cov_matrix)
    ef_sharpe.max_sharpe()

    ef = EfficientFrontier(mu, cov_matrix)

    def big_tech_constraint(w): return cp.sum(
        [w[idx] for idx in big_tech_indices]) <= max_big_tech_weight

    ef.add_constraint(big_tech_constraint)
    ef.max_sharpe()
    ef.clean_weights()
    return ef


def _ef_default_returns_range(ef, n_samples=10000):
    """Generate a range of expected returns for the efficient frontier."""
    mus = []
    for _ in range(n_samples):
        w = np.random.dirichlet(np.ones(len(ef.expected_returns)), 1)
        mus.append(w.dot(ef.expected_returns)[0])
    return np.linspace(min(mus), max(mus), n_samples), mus


def plot_efficient_frontier_with_random_portfolios(ef, ax=None, n_samples=10000, show_assets=True):
    if ax is None:
        ax = plt.gca()

    # Ensure these are methods and not overwritten
    ef_param_range = _ef_default_returns_range(ef, n_samples)

    # Correctly calling methods with parentheses
    ax.plot(ef_param_range[0], ef_param_range[1], label="Efficient Frontier")

    if show_assets:
        # Ensure these are methods and not overwritten
        ax.scatter(
            np.sqrt(np.diag(ef.cov_matrix)),
            ef.expected_returns,
            s=30,
            color="k",
            label="Assets",
        )

    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    return ax


def get_efficient_frontier_data(ef, n_samples=10000, show_assets=True):
    # Ensure these are methods and not overwritten
    ef_param_range = _ef_default_returns_range(ef, n_samples)

    ef_data = {
        'returns': ef_param_range[0],
        'volatility': ef_param_range[1]
    }

    assets_data = None
    if show_assets:
        assets_data = {
            'volatility': np.sqrt(np.diag(ef.cov_matrix)),
            'expected_returns': ef.expected_returns
        }

    return ef_data, assets_data


def main():
    tickers = ["MSFT", "AMZN", "KO", "MA", "COST", "LUV", "XOM",
               "PFE", "JPM", "UNH", "ACN", "DIS", "GILD", "F", "TSLA"]

    prices = fetch_historical_prices(tickers)
    sample_cov = calculate_sample_covariance(prices)
    mu = calculate_expected_returns(prices)

    weights, _ = optimize_portfolio(
        mu, sample_cov, objective_function="max_sharpe")

    sector_mapper = {
        "MSFT": "Tech",
        "AMZN": "Consumer Discretionary",
        "KO": "Consumer Staples",
        "MA": "Financial Services",
        "COST": "Consumer Staples",
        "LUV": "Aerospace",
        "XOM": "Energy",
        "PFE": "Healthcare",
        "JPM": "Financial Services",
        "UNH": "Healthcare",
        "ACN": "Misc",
        "DIS": "Media",
        "GILD": "Healthcare",
        "F": "Auto",
        "TSLA": "Auto"
    }

    sector_allocation = map_sectors_to_tickers(tickers, sector_mapper, weights)
    print("Sector allocation:", sector_allocation)

    big_tech_indices = [tickers.index(ticker)
                        for ticker in {"MSFT", "AMZN", "TSLA"}]
    ef = perform_constraint_optimization(
        mu, sample_cov, big_tech_indices)

    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt(np.diag(w @ sample_cov @ w.T))
    sharpes = rets / stds
    _, ax = plt.subplots()
    plot_efficient_frontier_with_random_portfolios(
        ef, ax=ax, n_samples=n_samples)
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

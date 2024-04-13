# Portfolio Optimization App

This Gradio application allows you to optimize a portfolio using various techniques and constraints.

## Introduction

Portfolio optimization is a crucial aspect of investment management, aiming to construct an investment portfolio that maximizes returns while minimizing risks. This application provides a user-friendly interface to perform portfolio optimization with customizable parameters.

## Features

- **Customizable Parameters:** Users can specify various parameters such as stock tickers, historical period, objective function, target volatility, sector allocations, and more.
- **Multiple Optimization Techniques:** The application supports different portfolio optimization techniques including maximizing Sharpe ratio, minimizing volatility, efficient risk, and efficient return.
- **Constraint Optimization:** Users can impose constraints on portfolio weights, sector allocations, market neutrality, and maximum weights for specific stocks or sectors.
- **Visualization:** The application generates visualizations such as efficient frontier plots, sector allocations, and discrete allocation of stocks in the optimized portfolio.
- **Example Usage:** Detailed examples are provided to guide users on how to utilize the application effectively.

## Input Parameters

- **Tickers:** A comma-separated list of stock tickers.
- **Period:** The historical period over which to fetch stock prices.
- **Frequency:** The frequency at which to sample stock prices.
- **Weight Bounds:** The lower and upper bounds on portfolio weights.
- **Sector Mapper:** A JSON object mapping tickers to sectors.
- **Objective Function:** The portfolio optimization objective function.
- **Gamma:** The regularization parameter for L2 regularization.
- **Target Volatility:** The target portfolio volatility.
- **Target Return:** The target portfolio return.
- **Market Neutral:** Whether to optimize for a market-neutral portfolio.
- **Total Portfolio Value:** The total value of the portfolio to be optimized.
- **Short Ratio:** The maximum ratio of short positions to long positions.
- **Target CVaR:** The target Conditional Value at Risk (CVaR).
- **Max Big Tech Weight:** The maximum weight allowed for big tech stocks (MSFT, AMZN, TSLA).

## Output

- **Portfolio Performance:** The performance metrics of the optimized portfolio, including expected return, volatility, and Sharpe ratio.
- **Discrete Allocation:** The discrete allocation of the portfolio, including the number of shares of each stock and the total investment amount.
- **Sector Allocation:** The allocation of the portfolio to different sectors.
- **CVaR Performance:** The performance metrics of the optimized portfolio under the CVaR objective.
- **CLA Performance:** The performance metrics of the optimized portfolio using the Critical Line Algorithm (CLA).
- **Constraint Optimization:** The performance metrics of the optimized portfolio with the specified constraints.
- **Efficient Frontier:** A plot of the efficient frontier for the given portfolio.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/WazupSteve/PyPortfolio.git
    ```

2. Navigate to the project directory

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    python app.py
    ```
    
5. Access the application in your web browser at [http://localhost:7860](http://localhost:7860).

## Usage

1. Enter the desired input parameters in the corresponding fields.
2. Click the "Run" button to optimize the portfolio.
3. The output will be displayed in the corresponding fields.

Click the "Run" button to optimize the portfolio. The output will include the portfolio performance, discrete allocation, sector allocation, and efficient frontier plot.

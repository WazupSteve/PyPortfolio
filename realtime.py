import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf

# Alpha Vantage API configuration
API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
BASE_URL = 'https://www.alphavantage.co/query'


# Step 1: Define Questionnaire
questionnaire_options = {
    'Investment Horizon': {
        'Very Short-Term (<= 2 years)': 1,
        'Short-Term (2-5 years)': 2,
        'Medium-Term (5-10 years)': 3,
        'Long-Term (10-15 years)': 4,
        'Very Long-Term (> 15 years)': 5
    },
    'Risk Attitude': {
        'Extremely Conservative': 1,
        'Conservative': 2,
        'Moderately Conservative': 3,
        'Moderately Aggressive': 4,
        'Aggressive': 5,
        'Very Aggressive': 6
    },
    'Financial Knowledge': {
        'Novice': 1,
        'Intermediate': 2,
        'Advanced': 3
    },
    'Investment Experience': {
        'None': 1,
        'Limited': 2,
        'Extensive': 3
    },
    'Age': {
        '18-25': 1,
        '26-35': 2,
        '36-45': 3,
        '46-55': 4,
        '56-65': 5,
        '66+': 6
    },
    'Income Level': {
        'Less than $25,000': 1,
        '$25,000 - $49,999': 2,
        '$50,000 - $74,999': 3,
        '$75,000 - $99,999': 4,
        '$100,000 - $149,999': 5,
        '$150,000 or more': 6
    }
}

# Step 2: Compute Risk Score
def compute_risk_score(responses):
    score = 0
    for question, value in responses.items():
        if question in ['Investment Horizon', 'Risk Attitude', 'Age']:
            score += value * 2
        elif question == 'Income Level':
            score += value * 1.5
        else:
            score += value
    return score

def prepare_data(responses_df, ratios_df):
    categorical_features = ['Investment Horizon', 'Risk Attitude', 'Financial Knowledge', 'Investment Experience', 'Age', 'Income Level']
    X = responses_df[categorical_features]
    encoder = OrdinalEncoder()
    X = encoder.fit_transform(X)
    y = ratios_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, encoder

# Step 5: Model Training
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def suggest_investment_ratio(responses, model, encoder):
    categorical_features = ['Investment Horizon', 'Risk Attitude', 'Financial Knowledge', 'Investment Experience', 'Age', 'Income Level']
    X = np.array([responses[feature] for feature in categorical_features]).reshape(1, -1)
    X = encoder.transform(X)
    predicted_ratios = model.predict(X)
    ratios = predicted_ratios.flatten()
    return ratios / ratios.sum()

def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def update_expected_returns():
    investment_amount = float(investment_amount_entry.get())
    debt_ratio = debt_scale.get() / 100
    equity_ratio = equity_scale.get() / 100
    hybrid_ratio = hybrid_scale.get() / 100
    total_ratio = debt_ratio + equity_ratio + hybrid_ratio
    if total_ratio > 1:
        debt_ratio /= total_ratio
        equity_ratio /= total_ratio
        hybrid_ratio /= total_ratio
        debt_scale.set(debt_ratio * 100)
        equity_scale.set(equity_ratio * 100)
        hybrid_scale.set(hybrid_ratio * 100)
    expected_returns = investment_amount * np.array([debt_ratio, equity_ratio, hybrid_ratio])
    debt_returns_label.config(text=f"Debt: {expected_returns[0]:.2f}")
    equity_returns_label.config(text=f"Equity: {expected_returns[1]:.2f}")
    hybrid_returns_label.config(text=f"Hybrid: {expected_returns[2]:.2f}")

def submit_questionnaire():
    user_responses = {}
    for question, options in questionnaire_options.items():
        user_responses[question] = int(selected_options[question].get())
    risk_score = compute_risk_score(user_responses)
    risk_score_label.config(text=f"Your Risk Score: {risk_score}")
    update_results(user_responses)

def update_results(user_responses):
    filtered_responses_df = responses_df[responses_df.apply(compute_risk_score, axis=1) == compute_risk_score(user_responses)]
    filtered_ratios_df = ratios_df.iloc[filtered_responses_df.index]
    if not filtered_responses_df.empty:
        X_train, X_test, y_train, y_test, encoder = prepare_data(filtered_responses_df, filtered_ratios_df)
        model = train_model(X_train, y_train)
        suggested_ratios = suggest_investment_ratio(user_responses, model, encoder)
    else:
        suggested_ratios = ratios_df.mean().values
    debt_scale.set(suggested_ratios[0] * 100)
    equity_scale.set(suggested_ratios[1] * 100)
    hybrid_scale.set(suggested_ratios[2] * 100)
    update_expected_returns()

import matplotlib.pyplot as plt

def plot_stock_data(stock_data, stock_frame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for stock, data in stock_data.items():
        ax.plot(data.index, data['Close'], label=stock)
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Stock Prices')
    ax.legend()
    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=stock_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def fetch_and_plot_stock_data(stock_entry, start_date_entry, end_date_entry, stock_frame):
    stock_symbols = stock_entry.get().split(',')
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    stock_data = {}
    for symbol in stock_symbols:
        symbol = symbol.strip()
        data = fetch_stock_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            stock_data[symbol] = data
    if stock_data:
        plot_stock_data(stock_data, stock_frame)
    else:
        tk.messagebox.showinfo("Error", "No valid stock data found.")

def main():
    global debt_scale, equity_scale, hybrid_scale, investment_amount_entry, debt_returns_label, equity_returns_label, hybrid_returns_label
    global responses_df, ratios_df, selected_options, risk_score_label

    # Create the main window with better initial size
    window = tk.Tk()
    window.title("Investment Suggestion System")
    window.geometry("800x600")  # Adjust dimensions as needed

    # Use Notebook for organized sections
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)

    # Questionnaire Tab
    questionnaire_frame = ttk.Frame(notebook)
    selected_options = {}
    row = 0
    for question, options in questionnaire_options.items():
        question_label = ttk.Label(questionnaire_frame, text=question)
        question_label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        selected_option = tk.StringVar(value=list(options.values())[0])
        selected_options[question] = selected_option
        for key, value in options.items():
            radio_button = ttk.Radiobutton(questionnaire_frame, text=key, value=value, variable=selected_option)
            radio_button.grid(row=row, column=1, padx=5, pady=2, sticky="w")
            row += 1
        row += 1
    submit_button = ttk.Button(questionnaire_frame, text="Submit", command=submit_questionnaire)
    submit_button.grid(row=row, columnspan=2, padx=5, pady=10)
    risk_score_label = ttk.Label(questionnaire_frame, text="Your Risk Score: ")
    risk_score_label.grid(row=row+1, columnspan=2, padx=5, pady=10)
    notebook.add(questionnaire_frame, text="Questionnaire")

    # Results Tab
    results_frame = ttk.Frame(notebook)
    # Create labels and scroll bars for investment ratios
    debt_label = ttk.Label(results_frame, text="Debt Ratio:")
    debt_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    debt_scale = tk.Scale(results_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=lambda _: update_expected_returns())
    debt_scale.grid(row=0, column=1, padx=5, pady=5)
    equity_label = ttk.Label(results_frame, text="Equity Ratio:")
    equity_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    equity_scale = tk.Scale(results_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=lambda _: update_expected_returns())
    equity_scale.grid(row=1, column=1, padx=5, pady=5)
    hybrid_label = ttk.Label(results_frame, text="Hybrid Ratio:")
    hybrid_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    hybrid_scale = tk.Scale(results_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=lambda _: update_expected_returns())
    hybrid_scale.grid(row=2, column=1, padx=5, pady=5)
    # Create and pack the investment amount frame
    investment_amount_label = ttk.Label(results_frame, text="Investment Amount:")
    investment_amount_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
    investment_amount_entry = ttk.Entry(results_frame)
    investment_amount_entry.grid(row=3, column=1, padx=5, pady=5)
    # Create and pack the expected returns frame
    expected_returns_label = ttk.Label(results_frame, text="Expected Returns:")
    expected_returns_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
    debt_returns_label = ttk.Label(results_frame, text="Debt: 0.00")
    debt_returns_label.grid(row=5, column=0, padx=5, pady=2, sticky="w")
    equity_returns_label = ttk.Label(results_frame, text="Equity: 0.00")
    equity_returns_label.grid(row=6, column=0, padx=5, pady=2, sticky="w")
    hybrid_returns_label = ttk.Label(results_frame, text="Hybrid: 0.00")
    hybrid_returns_label.grid(row=7, column=0, padx=5, pady=2, sticky="w")
    notebook.add(results_frame, text="Results & Adjustments")

    # Stock Data Tab
    stock_frame = ttk.Frame(notebook)
    # Create an entry field for stock ticker names
    stock_label = ttk.Label(stock_frame, text="Enter stock ticker names (comma-separated):")
    stock_label.pack(pady=5)
    stock_entry = ttk.Entry(stock_frame)
    stock_entry.pack(pady=5)
    # Create entry fields for start and end dates
    start_date_label = ttk.Label(stock_frame, text="Start Date (YYYY-MM-DD):")
    start_date_label.pack(pady=5)
    start_date_entry = ttk.Entry(stock_frame)
    start_date_entry.pack(pady=5)
    end_date_label = ttk.Label(stock_frame, text="End Date (YYYY-MM-DD):")
    end_date_label.pack(pady=5)
    end_date_entry = ttk.Entry(stock_frame)
    end_date_entry.pack(pady=5)
    # Create a button to fetch and plot stock data
    fetch_button = ttk.Button(stock_frame, text="Fetch Stock Data", command=lambda: fetch_and_plot_stock_data(stock_entry, start_date_entry, end_date_entry, stock_frame))
    fetch_button.pack(pady=5)
    notebook.add(stock_frame, text="Stock Data")

    # Run the main event loop
    window.mainloop()

if __name__ == "__main__":
    main()

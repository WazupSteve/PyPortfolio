import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import yfinance as yf
from typing import Dict

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


def get_user_questionnaire() -> Dict[str, int]:
    responses = {}
    for question, options in questionnaire_options.items():
        print(f"\n** {question} **")
        for key, value in options.items():
            print(f"{value}. {key}")
        response = input("Enter your choice: ").strip()
        while response not in map(str, options.values()):
            print("Invalid choice. Please select from the options above.")
            response = input("Enter your choice: ").strip()
        responses[question] = int(response)
    return responses

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

# Step 3: Fetch Historical Returns Data


def get_historical_returns(tickers, start_date, end_date):
    historical_returns = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            historical_returns[ticker] = data['Adj Close'].pct_change(
            ).dropna()
        except (yf.DownloadError, ValueError) as e:
            print(f"Error downloading data for {ticker}: {e}")
            historical_returns[ticker] = pd.DataFrame()
    historical_returns_df = pd.DataFrame(historical_returns)
    return historical_returns_df


def prepare_data(responses_df, ratios_df):
    categorical_features = ['Investment Horizon', 'Risk Attitude',
                            'Financial Knowledge', 'Investment Experience', 'Age', 'Income Level']
    X = responses_df[categorical_features]
    encoder = OrdinalEncoder()
    X = encoder.fit_transform(X)
    y = ratios_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
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

# Step 7: Investment Ratio Suggestion


def suggest_investment_ratio(responses, model, encoder):
    categorical_features = ['Investment Horizon', 'Risk Attitude',
                            'Financial Knowledge', 'Investment Experience', 'Age', 'Income Level']
    X = np.array([responses[feature]
                 for feature in categorical_features]).reshape(1, -1)
    X = encoder.transform(X)
    predicted_ratios = model.predict(X)
    ratios = predicted_ratios.flatten()
    return ratios / ratios.sum()


def generate_simulated_dataset(num_samples):
    responses = []
    ratios = []
    for _ in range(num_samples):
        response = {
            'Investment Horizon': np.random.randint(1, 6),
            'Risk Attitude': np.random.randint(1, 7),
            'Financial Knowledge': np.random.randint(1, 4),
            'Investment Experience': np.random.randint(1, 4),
            'Age': np.random.randint(1, 7),
            'Income Level': np.random.randint(1, 7),
        }
        responses.append(response)
        # Generate random investment ratios for 3 categories (debt, equity, hybrid)
        ratio = np.random.dirichlet(np.ones(3))
        ratios.append(ratio)
    return pd.DataFrame(responses), pd.DataFrame(ratios, columns=['Debt', 'Equity', 'Hybrid'])


def main():
    # Generate a simulated dataset with 1000 samples
    responses_df, ratios_df = generate_simulated_dataset(1000)

    # Get User Questionnaire Responses
    user_responses = get_user_questionnaire()

    # Compute Risk Score
    risk_score = compute_risk_score(user_responses)
    print(f"\nYour Risk Score: {risk_score}")

    # Filter the simulated dataset based on the user's risk score
    filtered_responses_df = responses_df[responses_df.apply(
        compute_risk_score, axis=1) == risk_score]
    filtered_ratios_df = ratios_df.iloc[filtered_responses_df.index]

    # Calculate the average investment ratios for the filtered dataset
    avg_ratios = filtered_ratios_df.mean()

    # Prepare Data for Training (using the simulated dataset)
    X_train, X_test, y_train, y_test, encoder = prepare_data(
        filtered_responses_df, filtered_ratios_df)

    # Train Model
    model = train_model(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Get Investment Ratio Suggestions for the user
    ratio = suggest_investment_ratio(user_responses, model, encoder)
    print("\nSuggested investment ratio:")
    for category, ratio_value in zip(['Debt', 'Equity', 'Hybrid'], ratio):
        print(f"{category}: {ratio_value:.2f}")

    # Calculate the expected returns based on the suggested investment ratios
    investment_amount = float(input("\nEnter the investment amount: "))
    expected_returns = investment_amount * avg_ratios
    print("\nExpected returns:")
    for category, return_value in zip(['Debt', 'Equity', 'Hybrid'], expected_returns):
        print(f"{category}: {return_value:.2f}")


if __name__ == "__main__":
    main()

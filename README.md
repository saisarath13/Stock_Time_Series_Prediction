# Stock Price Prediction using RNN and LSTM

This project demonstrates how to use RNN and LSTM models for stock price prediction. The pipeline includes data fetching, preprocessing, model building, training, evaluation, and saving fine-tuned models.

## Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Required libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow`

Install the dependencies using pip:

```bash
!pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

## Project Workflow

### 1. Data Preparation

1. **Fetch Historical Data:**
   - Use Yahoo Finance to fetch historical stock data.
   - The `fetch_stock_data` function fetches data for a given ticker and start date:

     ```python
     stock_data = fetch_stock_data('AAPL', '2015-01-01')
     ```

2. **Normalize Data:**
   - Normalize the stock prices using `MinMaxScaler` for better model performance.

3. **Create Sequences:**
   - Convert the normalized data into sequences for time series forecasting.

4. **Train-Test Split:**
   - Split the data into training and testing sets with an 80/20 ratio.

### 2. Model Building

The following models are implemented:

- **Simple RNN Model:**
  - A basic RNN model with 50 units and a Dense layer for output.

- **LSTM Model:**
  - An LSTM model with 50 units for better long-term dependencies.

- **Fine-tuned Models:**
  - Both RNN and LSTM models with additional layers and Dropout for regularization.

### 3. Model Training

- Train the models using the `train_model` function.
- Plot the training and validation loss to monitor performance.
- Example:

  ```python
  rnn_model = create_rnn_model(input_shape=(X_train.shape[1], 1))
  train_model(rnn_model, X_train, y_train, X_test, y_test)
  ```

### 4. Evaluation

- Evaluate the models using Root Mean Squared Error (RMSE).
- Example:

  ```python
  rnn_rmse = calculate_rmse(rnn_predictions, y_test_actual)
  print(f"RNN RMSE: {rnn_rmse}")
  ```

### 5. Saving Models

- Save the fine-tuned models for later use.
- Example:

  ```python
  rnn_model_finetuned.save('fine_tuned_rnn_model.keras')
  lstm_model_finetuned.save('fine_tuned_lstm_model.keras')
  ```

## Directory Structure

```
project_root/
|
|-- data_fetching.py          # Script to fetch historical stock data
|-- model_building.py         # Script for building RNN and LSTM models
|-- model_training.py         # Script for training and evaluating models
|-- fine_tuned_rnn_model.keras  # Fine-tuned RNN model
|-- fine_tuned_lstm_model.keras # Fine-tuned LSTM model
|-- requirements.txt          # Dependencies
|-- .gitignore                # Ignore unnecessary files
```

## Results

- RNN and LSTM models were trained to predict stock prices.
- RMSE values for both models indicate their prediction accuracy.
- Fine-tuned models with Dropout improve generalization.

## Usage

1. Clone the repository and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the scripts in order to fetch data, train models, and evaluate performance.

3. Use the saved models for future predictions.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

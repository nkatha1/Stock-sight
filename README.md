# Stock Predictor App 📈

A **Python-based web app** that predicts the next-day stock price for a given company using **historical stock data** and **machine learning**. Built with **Streamlit**, **PyTorch**, and **scikit-learn**, this app demonstrates practical data science, time series forecasting, and web app deployment.

---

## Features

- Download historical stock data from Yahoo Finance (`yfinance`)  
- Train a neural network model to predict stock prices  
- Predict the next-day closing price for a given stock ticker  
- Interactive web interface with **Streamlit**  
- Data visualization using **Matplotlib**  

---

## Tech Stack

- **Python 3.14**  
- **Streamlit** – Web application framework  
- **PyTorch** – Neural network training  
- **scikit-learn** – Data preprocessing and scaling  
- **yfinance** – Stock market data  
- **Matplotlib & Pandas** – Data visualization and handling  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/stock-predictor.git
cd stock-predictor


2.Create a virtual environment:
python -m venv venv

3.Activate the environment:

Windows (PowerShell):
venv\Scripts\Activate.ps1

Windows (CMD):
venv\Scripts\activate.bat

Mac/Linux:
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Usage
Run the Streamlit app:

streamlit run app.py

Enter the stock ticker (e.g., AAPL)

The app will download historical data, train the model, and display the predicted next-day price

Visualizations of historical stock trends will also be shown

Project Structure
stock-predictor/
│
├─ app.py                  # Streamlit front-end
├─ stock_predictor.py      # Training & prediction scripts
├─ requirements.txt        # Python dependencies
├─ .gitignore              # Files to ignore in Git
└─ README.md               # Project documentation

Notes

Training the model may take a few minutes depending on the dataset size.

Only the closing price is used for prediction in this version.

For real-world trading, this is educational and should not be used for financial decisions.


License

This project is licensed under the MIT License.
Acknowledgements

Yahoo Finance
 for stock data

PyTorch
 for machine learning framework

Streamlit
 for making web app deployment easy







📊 Dealership KPI Dashboard

An interactive Streamlit dashboard for analyzing and forecasting dealership KPIs such as Sales, Revenue, and Profit.
The app combines Python forecasting models (ARIMA / Linear Regression) with interactive visualization tools so users can:

📈 View actual vs. forecasted KPIs

🔥 Explore correlation heatmaps between KPIs

🤔 Run what-if scenario simulations (e.g., “What if Sales increased by 10%?”)

📂 Upload their own CSV files for analysis

🚀 Features

CSV Upload – upload dealership KPI data (account_id, english_name, year, month, monthly_value)

Forecasting – 3-month forecasts with ARIMA or Linear Regression fallback

Correlation Analysis – heatmap of KPI dependencies

What-If Simulation – adjust KPI by a percentage and see ripple effects

Interactive Dashboard – Streamlit interface with charts & tables

🛠 Tech Stack

Python 3.9+

Streamlit – interactive UI

Pandas & NumPy – data wrangling

Matplotlib – visualization

Scikit-Learn – regression models

Statsmodels – ARIMA forecasting

📂 Project Structure
Dealership-kpi-dashboard/
│
├── dashboard.py          # Streamlit dashboard app
├── index.py              # Forecasting + correlations + what-if logic
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── data/                 # (Optional) Sample CSVs for testing
    ├── sample_data.csv
    └── dealership_kpis.csv

▶️ How to Run Locally
1️⃣ Clone the Repo
git clone https://github.com/Janvi-001/Dealership-kpi-dashboard.git
cd Dealership-kpi-dashboard

2️⃣ Create Virtual Environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Dashboard
streamlit run dashboard.py

5️⃣ Open in Browser

Go to 👉 http://localhost:8501

📊 Sample Data

You can test the dashboard with the included sample_data.csv.
The CSV must include these required columns:

account_id

english_name (KPI name)

year

month

monthly_value

🌐 Deployment
Option 1 – Localhost

Run with:

streamlit run dashboard.py

Option 2 – Streamlit Cloud (free)

Push this repo to GitHub

Go to Streamlit Cloud

Create a new app → select your repo → set main file = dashboard.py

Option 3 – Hugging Face Spaces (recommended if Streamlit Cloud fails)

Go to Hugging Face Spaces

Create a Space → SDK = Streamlit → Public

Connect your GitHub repo or upload files

App runs automatically with a public link

📸 Screenshots (Optional)
<img width="1872" height="801" alt="image" src="https://github.com/user-attachments/assets/bc92cc6a-e00c-4cec-abc0-c2752218ca57" />



# 📈 Quantitative Finance Dashboard: Option Pricing & Volatility Analysis

A professional-grade quantitative finance dashboard built with **Python**, **Streamlit**, and **Plotly**. This tool allows traders and researchers to price options using the **Black-Scholes Model**, analyze **Volatility Surfaces**, and visualize real-time **Greeks**.

**Live Demo:** https://option-pricing-volatility-analysis.streamlit.app/

---

## 🚀 Key Features

### 1. 📊 Advanced Option Pricing
- **Black-Scholes Model**: Real-time pricing for Call and Put options.
- **Interactive P&L Visualizer**: A dynamic chart comparing "Value at Expiry" (Intrinsic) vs. "Value Today" to visualize **Theta (Time Decay)**.
- **Greeks Analysis**: 3D Interactive Surfaces for **Delta** (Sensitivity) and **Gamma** (Risk).

### 2. 📉 Real-World Market Data
- **Volatility Smile**: Visualizes the Implied Volatility skew across different strike prices using live data from `yfinance`.
- **Historical Volatility**: Calculates and plots annualized 30-day rolling volatility using 1 year of historical price data.
- **Real-Time Ticker Search**: Fetch live prices for any stock (AAPL, TSLA, NVDA, etc.) or use the "Quick Select" dropdown.

### 3. 🎨 Professional UI/UX
- **Dark Mode "Terminal" Aesthetic**: Custom CSS styling for a professional trading terminal look.
- **P&L Heatmaps**: Interactive Red/Green heatmaps to visualize profit scenarios across Spot Price and Volatility ranges.
- **Advanced Toggle**: Clean interface with a sidebar toggle to hide/show complex quantitative plots.

---

## 🛠️ Tech Stack

- **Core Logic**: `Python`, `NumPy`, `SciPy` (Normal Distribution, PDFs, CDFs)
- **Visualization**: `Plotly` (Interactive Charts), `Seaborn` (Heatmaps), `Matplotlib`
- **Web Framework**: `Streamlit`
- **Data Source**: `yfinance` (Yahoo Finance API)

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/vedpatel-dev/Option-Pricing-Volatility-Analysis.git
cd Option-Pricing-Volatility-Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run option_pricer.py
```

(Note: The main file is named option_pricer.py, not streamlit_app.py)

---

## 📂 Project Structure

```
📦 Option-Pricer/
 ┣ 📄 option_pricer.py    # Main Application Code (Logic + UI)
 ┣ 📄 requirements.txt    # List of Python dependencies
 ┣ 📄 README.md           # Project Documentation
 ┗ 📄 LICENSE             # MIT License
```

---

## 📬 Contact & Credits

**Developed by:** Ved Patel

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/ved-rajeshkumar-patel-vrp/)

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/vedpatel-dev)

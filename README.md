# Smart Inventory and Stock Management System using Machine Learning

A clean, beginner-friendly, and interactive inventory management dashboard built with **Python**, **Streamlit**, **SQLite**, and **Scikit-learn**.

The system automates product tracking, monitors real-time inventory levels, records sales transactions, alerts on low-stock conditions, and utilizes a Machine Learning model to forecast future demand and recommend restocking quantities.

---

## 🌟 Key Features

1. **Product & Stock Management**:
   - Add new products with categories, prices, stock quantities, and minimum alert thresholds.
   - View, search, and filter inventory by category or keyword.
   - Update product details, restock quantities, or remove discontinued items.

2. **Automated Low-Stock Alerts**:
   - Identifies items at or below their safety stock threshold.
   - Visual alerts and deficit calculation to prevent stockouts.

3. **Point of Sale (POS) Transaction Recording**:
   - Record customer purchases with automatic stock validation.
   - Prevents overselling when stock is insufficient.
   - Automatically decrements stock in SQLite in real time.

4. **Interactive Analytics & Charts**:
   - **Daily Revenue Trend**: Historical sales revenue over time.
   - **Category Distribution**: Donut chart breakdown of revenue by category.
   - **Best Sellers**: Top products ranked by total volume sold.
   - **Stock vs. Threshold**: Side-by-side comparison of inventory against minimum safety levels.

5. **Machine Learning Demand Forecasting**:
   - Predicts future sales demand for the next **7, 14, or 30 days**.
   - Choice of algorithms: **Random Forest Regressor** or **Linear Regression**.
   - Feature engineering incorporates historical trend, day-of-week seasonality, and rolling averages.

6. **Smart Restocking Recommendations**:
   - Compares predicted future demand directly against current inventory and safety buffers.
   - Generates actionable advice:
     - 🔴 **Urgent Restock**: Projected demand exceeds current stock.
     - 🟠 **Restock Advised**: Stock is low and will compromise safety buffers.
     - 🟢 **Stock Sufficient**: Adequate stock to fulfill anticipated demand.

---

## 📁 Project Structure

```text
smart_inventory/
├── app.py                  # Main Streamlit web application & UI dashboard
├── database.py             # SQLite database layer (CRUD operations, sales, alerts)
├── ml_model.py             # Scikit-learn demand forecasting & recommendation engine
├── sample_sales_data.csv   # Historical sales dataset for initial seeding & ML training
├── requirements.txt        # Project Python dependencies
└── README.md               # Documentation and setup guide
```

---

## 🛠️ Technology Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Matplotlib](https://matplotlib.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest & Linear Regression)
- **Database**: [SQLite3](https://docs.python.org/3/library/sqlite3.html) (Built-in, zero configuration)

---

## 🚀 How to Run in Visual Studio Code (Step-by-Step)

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File > Open Folder...** (or press `Ctrl + K, Ctrl + O`).
3. Select the `smart_inventory` folder:
   ```
   C:\Users\<YourUser>\.gemini\antigravity\scratch\smart_inventory
   ```

### Step 2: Open the Integrated Terminal
- In VS Code, open the terminal by pressing `` Ctrl + ` `` (Backtick) or go to **Terminal > New Terminal**.

### Step 3: (Recommended) Create & Activate a Virtual Environment
In the VS Code terminal, run:

- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

*(If you get a script execution policy error on PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*.

### Step 4: Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### Step 5: Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```

Streamlit will automatically launch in your default web browser at:
```
http://localhost:8501
```

---

## 🧠 How the Machine Learning Model Works

1. **Data Preprocessing**:
   - Retrieves daily sales for the selected product.
   - Fills non-trading days with 0 sales so time series spacing is continuous.
2. **Feature Engineering**:
   - `day_num`: Sequential timeline index capturing overall sales trend.
   - `day_of_week` & `is_weekend`: Captures weekly fluctuations and weekend demand surges.
   - `day_of_month`: Accounts for monthly cycles.
   - `rolling_7d`: 7-day moving average capturing recent momentum.
   - `lag_1`: Previous day's sales velocity.
3. **Training & Forecasting**:
   - Fits a `RandomForestRegressor` (or `LinearRegression`).
   - Uses iterative multi-step forecasting to predict demand day-by-day across the chosen horizon.
4. **Recommendation Formula**:
   $$\text{Projected Remaining} = \text{Current Stock} - \text{Total Predicted Demand}$$
   - If $\text{Projected Remaining} < 0$, a stockout is projected. Recommended restock quantity is calculated to fulfill demand and restore the safety buffer.

---

## 💾 SQLite Database Details

The application creates an `inventory.db` SQLite database file automatically upon launch.
- `products`: Stores product metadata, unit price, current stock, and minimum safety threshold.
- `sales`: Stores completed sales transactions linked to products with quantity, sale price, and date.
- Initial seed data from `sample_sales_data.csv` is populated on first start so you can explore all features right away!

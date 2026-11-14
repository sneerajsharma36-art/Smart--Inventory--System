"""
Smart Inventory and Stock Management System using Machine Learning
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date
import database
import ml_model

# Page configuration
st.set_page_config(
    page_title="Smart Inventory & Stock Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and seed if empty
database.init_db()

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Low stock alert badge */
    .alert-banner {
        background-color: #fff5f5;
        border-left: 5px solid #e53e3e;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .safe-banner {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    
    /* Recommendation Card */
    .rec-card {
        border-radius: 10px;
        padding: 20px;
        color: #ffffff;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2897/2897785.png", width=70)
st.sidebar.title("Smart Inventory")
st.sidebar.caption("AI-Powered Stock & Demand Manager")

menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard Overview",
        "📦 Manage Products & Stock",
        "🛒 Record Sale",
        "📈 Sales & Stock Analytics",
        "🤖 AI Demand Forecasting"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Switch tabs to manage inventory, track live stock deficits, and predict demand with Scikit-learn.")


# -------------------------------------------------------------
# TAB 1: DASHBOARD OVERVIEW
# -------------------------------------------------------------
if menu == "📊 Dashboard Overview":
    st.title("📊 Smart Inventory Dashboard")
    st.markdown("Real-time stock overview, low-inventory notifications, and key performance metrics.")

    products_df = database.get_products_df()
    sales_df = database.get_sales_df()
    low_stock_df = database.get_low_stock_products()

    # Top KPI Metrics
    total_products = len(products_df)
    total_stock_units = int(products_df["stock"].sum()) if not products_df.empty else 0
    total_sales_rev = float(sales_df["total_amount"].sum()) if not sales_df.empty else 0.0
    low_stock_count = len(low_stock_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Products</div>
            <div class="metric-value">📦 {total_products}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Stock Units</div>
            <div class="metric-value">🏷️ {total_stock_units:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Sales Revenue</div>
            <div class="metric-value">💰 ${total_sales_rev:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        alert_color = "#e53e3e" if low_stock_count > 0 else "#38a169"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Low Stock Alerts</div>
            <div class="metric-value" style="color: {alert_color};">⚠️ {low_stock_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Low Stock Alert Section
    if low_stock_count > 0:
        st.markdown(f"""
        <div class="alert-banner">
            <h4 style="margin: 0; color: #c53030;">⚠️ Action Required: {low_stock_count} Product(s) Below Minimum Stock Level</h4>
            <p style="margin: 5px 0 0 0; color: #742a2a; font-size: 0.95rem;">
                The following items are running critically low. Review and restock to prevent fulfillment delays.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        display_low = low_stock_df[["name", "category", "price", "stock", "min_stock", "deficit"]].copy()
        display_low.columns = ["Product Name", "Category", "Price ($)", "Current Stock", "Min Threshold", "Deficit Units"]
        st.dataframe(display_low, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class="safe-banner">
            <h4 style="margin: 0; color: #276749;">✅ All Stock Levels Healthy</h4>
            <p style="margin: 5px 0 0 0; color: #2f855a; font-size: 0.95rem;">
                No products are currently below their minimum safety thresholds.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Lower Grid: Quick Stock Visual & Recent Sales
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("📦 Inventory Stock Levels")
        if not products_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            colors = ["#e53e3e" if s <= m else "#3182ce" for s, m in zip(products_df["stock"], products_df["min_stock"])]
            bars = ax.barh(products_df["name"], products_df["stock"], color=colors, height=0.55)
            ax.set_xlabel("Units in Stock", fontsize=10, fontweight="bold")
            ax.set_title("Current Stock by Product (Red = Low Stock)", fontsize=11, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            
            # Annotate values
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No products found in inventory.")

    with c2:
        st.subheader("🛒 Recent Transactions")
        if not sales_df.empty:
            recent_sales = sales_df[["product_name", "quantity", "total_amount", "sale_date"]].head(8).copy()
            recent_sales.columns = ["Product", "Qty", "Total ($)", "Date"]
            st.dataframe(recent_sales, use_container_width=True, hide_index=True)
        else:
            st.info("No sales recorded yet.")


# -------------------------------------------------------------
# TAB 2: MANAGE PRODUCTS & STOCK
# -------------------------------------------------------------
elif menu == "📦 Manage Products & Stock":
    st.title("📦 Product & Stock Management")
    st.markdown("View all products, add new inventory items, and adjust stock quantities.")

    tab_view, tab_add, tab_update = st.tabs(["📋 View All Products", "➕ Add New Product", "🔄 Update Existing Product"])

    products_df = database.get_products_df()

    with tab_view:
        st.subheader("Current Inventory Catalog")
        
        # Search & Filter controls
        s1, s2 = st.columns([2, 1])
        with s1:
            search_query = st.text_input("🔍 Search product by name", "")
        with s2:
            categories = ["All"] + sorted(list(products_df["category"].unique())) if not products_df.empty else ["All"]
            cat_filter = st.selectbox("Filter by Category", categories)

        filtered_df = products_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["name"].str.contains(search_query, case=False, na=False)]
        if cat_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == cat_filter]

        if not filtered_df.empty:
            # Highlight low stock
            def highlight_stock(row):
                if row["stock"] <= row["min_stock"]:
                    return ["background-color: #fee2e2; color: #991b1b; font-weight: bold;"] * len(row)
                return [""] * len(row)

            display_cols = filtered_df[["id", "name", "category", "price", "stock", "min_stock"]].copy()
            display_cols.columns = ["ID", "Name", "Category", "Price ($)", "Current Stock", "Min Stock Alert"]
            
            st.dataframe(
                display_cols.style.apply(
                    lambda row: ["background-color: #fee2e2; color: #991b1b;" if row["Current Stock"] <= row["Min Stock Alert"] else "" for _ in row],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
            st.caption("Rows highlighted in red indicate products that have reached or dropped below their minimum stock threshold.")
        else:
            st.info("No products match the search criteria.")

    with tab_add:
        st.subheader("Add a New Product")
        with st.form("add_product_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                new_name = st.text_input("Product Name *", placeholder="e.g. Wireless Mouse Pro")
                new_category = st.selectbox("Category *", ["Electronics", "Accessories", "Furniture", "Office Supplies", "Other"])
                if new_category == "Other":
                    new_category = st.text_input("Specify Category", placeholder="e.g. Storage")
            with col_b:
                new_price = st.number_input("Unit Price ($) *", min_value=0.01, value=29.99, step=1.0)
                new_stock = st.number_input("Initial Stock Quantity *", min_value=0, value=20, step=1)
                new_min_stock = st.number_input("Minimum Stock Alert Level *", min_value=1, value=5, step=1)

            submitted = st.form_submit_button("➕ Add Product to Inventory")
            if submitted:
                if not new_name.strip():
                    st.error("Product name cannot be empty!")
                else:
                    success, msg = database.add_product(new_name, new_category, new_price, new_stock, new_min_stock)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    with tab_update:
        st.subheader("Edit Product Details or Restock")
        if not products_df.empty:
            product_dict = {f"{row['name']} (Current Stock: {row['stock']})": row["id"] for _, row in products_df.iterrows()}
            selected_label = st.selectbox("Select Product to Edit", list(product_dict.keys()))
            selected_id = product_dict[selected_label]
            
            selected_row = products_df[products_df["id"] == selected_id].iloc[0]

            with st.form("update_product_form"):
                u_col1, u_col2 = st.columns(2)
                with u_col1:
                    edit_name = st.text_input("Product Name", value=selected_row["name"])
                    edit_category = st.text_input("Category", value=selected_row["category"])
                    edit_price = st.number_input("Unit Price ($)", min_value=0.01, value=float(selected_row["price"]), step=1.0)
                with u_col2:
                    edit_stock = st.number_input("Stock Quantity", min_value=0, value=int(selected_row["stock"]), step=1)
                    edit_min_stock = st.number_input("Minimum Alert Threshold", min_value=1, value=int(selected_row["min_stock"]), step=1)

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    update_submitted = st.form_submit_button("💾 Save Changes")
                with btn_col2:
                    delete_submitted = st.form_submit_button("🗑️ Delete Product")

                if update_submitted:
                    success, msg = database.update_product(selected_id, edit_name, edit_category, edit_price, edit_stock, edit_min_stock)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

                if delete_submitted:
                    success, msg = database.delete_product(selected_id)
                    if success:
                        st.warning(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            st.info("No products available to update.")


# -------------------------------------------------------------
# TAB 3: RECORD SALE
# -------------------------------------------------------------
elif menu == "🛒 Record Sale":
    st.title("🛒 Record Product Sale")
    st.markdown("Record a customer purchase. Stock quantities are automatically validated and deducted in real time.")

    products_df = database.get_products_df()

    if products_df.empty:
        st.warning("No products in inventory! Please add products first.")
    else:
        # Create user-friendly product options with stock indicator
        prod_options = {}
        for _, row in products_df.iterrows():
            label = f"{row['name']} — ${row['price']:.2f} (Available Stock: {row['stock']})"
            prod_options[label] = row

        col_left, col_right = st.columns([3, 2])

        with col_left:
            with st.form("sale_entry_form"):
                chosen_label = st.selectbox("Select Product to Sell", list(prod_options.keys()))
                chosen_product = prod_options[chosen_label]

                current_avail = int(chosen_product["stock"])
                sale_qty = st.number_input("Quantity to Sell", min_value=1, max_value=max(1, current_avail), value=1, step=1)
                sale_date_input = st.date_input("Sale Date", value=date.today())

                total_cost = sale_qty * float(chosen_product["price"])
                st.markdown(f"**Total Transaction Amount:** `${total_cost:.2f}`")

                record_btn = st.form_submit_button("💳 Confirm & Record Sale")

                if record_btn:
                    if current_avail <= 0:
                        st.error(f"Cannot record sale: '{chosen_product['name']}' is out of stock!")
                    elif sale_qty > current_avail:
                        st.error(f"Cannot record sale: Only {current_avail} units available!")
                    else:
                        success, msg = database.record_sale(
                            chosen_product["id"],
                            sale_qty,
                            sale_date_input.strftime("%Y-%m-%d")
                        )
                        if success:
                            st.success(msg)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(msg)

        with col_right:
            st.subheader("Selected Item Summary")
            st.markdown(f"""
            - **Product:** `{chosen_product['name']}`
            - **Category:** `{chosen_product['category']}`
            - **Unit Price:** `${chosen_product['price']:.2f}`
            - **Available In Stock:** `{chosen_product['stock']} units`
            - **Min Safety Threshold:** `{chosen_product['min_stock']} units`
            """)
            if chosen_product['stock'] <= chosen_product['min_stock']:
                st.warning("⚠️ Warning: This item is already at or below low stock threshold!")
            else:
                st.success("✅ Stock level is sufficient.")


# -------------------------------------------------------------
# TAB 4: SALES & STOCK ANALYTICS
# -------------------------------------------------------------
elif menu == "📈 Sales & Stock Analytics":
    st.title("📈 Sales & Inventory Analytics")
    st.markdown("Visual representations of sales performance, inventory health, and revenue breakdown.")

    sales_df = database.get_sales_df()
    products_df = database.get_products_df()

    if sales_df.empty:
        st.info("No sales data available to chart.")
    else:
        # Convert date to datetime for plotting
        sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
        
        # Row 1: Sales Trend over Time & Category Distribution
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("📅 Daily Revenue Trend")
            daily_rev = sales_df.groupby("sale_date")["total_amount"].sum().reset_index()
            daily_rev = daily_rev.sort_values("sale_date")

            fig1, ax1 = plt.subplots(figsize=(7, 3.8))
            ax1.plot(daily_rev["sale_date"], daily_rev["total_amount"], color="#2b6cb0", linewidth=2, marker="o", markersize=3)
            ax1.set_ylabel("Revenue ($)", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Date", fontsize=10, fontweight="bold")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax1.grid(True, linestyle="--", alpha=0.5)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig1)

        with col_g2:
            st.subheader("🏷️ Sales Revenue by Category")
            cat_sales = sales_df.groupby("category")["total_amount"].sum()

            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            colors = ["#3182ce", "#38a169", "#d69e2e", "#805ad5", "#e53e3e"]
            wedges, texts, autotexts = ax2.pie(
                cat_sales,
                labels=cat_sales.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors[:len(cat_sales)],
                wedgeprops=dict(width=0.45, edgecolor="white")
            )
            for t in texts:
                t.set_fontsize(9)
            for at in autotexts:
                at.set_fontsize(9)
                at.set_weight("bold")
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("---")

        # Row 2: Top Selling Products & Stock vs Threshold
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            st.subheader("🏆 Top Selling Products (Units Sold)")
            top_products = sales_df.groupby("product_name")["quantity"].sum().sort_values(ascending=True)

            fig3, ax3 = plt.subplots(figsize=(7, 4))
            bars = ax3.barh(top_products.index, top_products.values, color="#38a169", height=0.5)
            ax3.set_xlabel("Total Units Sold", fontsize=10, fontweight="bold")
            ax3.grid(axis="x", linestyle="--", alpha=0.5)
            for bar in bars:
                w = bar.get_width()
                ax3.text(w + 1, bar.get_y() + bar.get_height()/2, f"{int(w)}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig3)

        with col_g4:
            st.subheader("⚖️ Current Stock vs Min Threshold")
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            indices = range(len(products_df))
            bar_w = 0.35
            
            ax4.bar([i - bar_w/2 for i in indices], products_df["stock"], width=bar_w, label="Current Stock", color="#3182ce")
            ax4.bar([i + bar_w/2 for i in indices], products_df["min_stock"], width=bar_w, label="Min Threshold", color="#e53e3e")
            ax4.set_xticks(indices)
            ax4.set_xticklabels(products_df["name"], rotation=35, ha="right", fontsize=9)
            ax4.set_ylabel("Units", fontsize=10, fontweight="bold")
            ax4.legend(loc="upper right")
            ax4.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig4)


# -------------------------------------------------------------
# TAB 5: AI DEMAND FORECASTING
# -------------------------------------------------------------
elif menu == "🤖 AI Demand Forecasting":
    st.title("🤖 AI Product Demand Forecasting & Restocking")
    st.markdown("""
    Predict future customer demand using Machine Learning models trained on historical sales patterns.
    The system analyzes sales velocity, day-of-week trends, and recent order volume to recommend optimal restocking quantities.
    """)

    products_df = database.get_products_df()
    sales_df = database.get_sales_df()

    if products_df.empty:
        st.warning("No products available to forecast.")
    else:
        # Configuration Controls
        col_p, col_days, col_mod = st.columns([2, 1, 1])
        
        with col_p:
            prod_names = list(products_df["name"])
            chosen_pname = st.selectbox("Select Product to Forecast", prod_names)
            selected_prod = products_df[products_df["name"] == chosen_pname].iloc[0]

        with col_days:
            forecast_horizon = st.selectbox("Forecast Horizon", [7, 14, 30], index=0, format_func=lambda x: f"Next {x} Days")

        with col_mod:
            model_choice = st.selectbox(
                "ML Algorithm",
                ["random_forest", "linear_regression"],
                format_func=lambda x: "Random Forest Regressor" if x == "random_forest" else "Linear Regression"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Run Forecast Pipeline
        with st.spinner("Training Machine Learning model on historical sales..."):
            daily_sales, forecast_df, metrics, rec = ml_model.run_demand_forecast(
                sales_df,
                selected_prod,
                forecast_days=forecast_horizon,
                model_type=model_choice
            )

        # Recommendation Banner
        bg_col = rec["badge_color"]
        st.markdown(f"""
        <div style="background-color: {bg_col}; padding: 18px 24px; border-radius: 10px; color: white; margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h3 style="margin: 0; color: white; font-weight: 700;">{rec['title']}</h3>
            <p style="margin: 8px 0 0 0; font-size: 1.05rem; line-height: 1.5; color: #ffffff;">
                {rec['message']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current Stock</div>
                <div class="metric-value">📦 {rec['current_stock']}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Predicted {forecast_horizon}-Day Demand</div>
                <div class="metric-value">🔮 {rec['total_predicted_demand']}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            rem_val = rec['projected_remaining']
            rem_color = "#e53e3e" if rem_val < 0 else "#2b6cb0"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Projected Remaining</div>
                <div class="metric-value" style="color: {rem_color};">{rem_val}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            restock_qty = rec['recommended_restock']
            rec_color = "#e53e3e" if restock_qty > 0 else "#38a169"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Recommended Restock</div>
                <div class="metric-value" style="color: {rec_color};">⚡ {restock_qty} units</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Forecast Chart
        st.subheader("📈 Historical Sales vs Predicted Future Demand")
        
        fig, ax = plt.subplots(figsize=(12, 4.8))
        
        if not daily_sales.empty:
            # Historical actual sales (last 30 days for clarity)
            recent_daily = daily_sales.tail(30)
            ax.plot(
                recent_daily["sale_date"],
                recent_daily["quantity"],
                label="Historical Sales (Actual)",
                color="#2b6cb0",
                marker="o",
                linewidth=2,
                markersize=4
            )
            # Connect the last actual point to the first predicted point
            last_actual_date = recent_daily["sale_date"].iloc[-1]
            last_actual_val = recent_daily["quantity"].iloc[-1]
            
            comb_dates = [last_actual_date] + list(forecast_df["sale_date"])
            comb_preds = [last_actual_val] + list(forecast_df["predicted_demand"])
            
            ax.plot(
                comb_dates,
                comb_preds,
                label=f"Predicted Demand ({metrics['model_type']})",
                color="#e53e3e",
                linestyle="--",
                marker="s",
                linewidth=2.2,
                markersize=4
            )
        else:
            ax.plot(
                forecast_df["sale_date"],
                forecast_df["predicted_demand"],
                label="Baseline Projected Demand",
                color="#e53e3e",
                linestyle="--",
                marker="s"
            )

        ax.set_ylabel("Units Demanded / Sold", fontsize=11, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        plt.xticks(rotation=25)
        plt.tight_layout()
        st.pyplot(fig)

        # Model Metadata and Detailed Day-by-Day Forecast
        c_meta, c_table = st.columns([1, 1])

        with c_meta:
            st.subheader("🧠 Model Evaluation & Stats")
            st.markdown(f"""
            - **Algorithm:** `{metrics.get('model_type', 'Random Forest Regressor')}`
            - **Training Set Size:** `{len(daily_sales)} days of sales records`
            - **Mean Absolute Error (MAE):** `{metrics.get('mae', 'N/A')} units`
            - **R² Score:** `{metrics.get('r2', 'N/A')}`
            - **Product Min Buffer Threshold:** `{rec['min_stock']} units`
            
            *The Scikit-learn model incorporates sequential trend indices, day-of-week seasonality, and 7-day rolling moving averages to estimate demand fluctuations.*
            """)

        with c_table:
            st.subheader("📋 Day-by-Day Forecast Breakdown")
            table_display = forecast_df.copy()
            table_display["Date"] = table_display["sale_date"].dt.strftime("%Y-%m-%d (%a)")
            table_display["Forecast Demand (Units)"] = table_display["predicted_demand"].round(1)
            st.dataframe(table_display[["Date", "Forecast Demand (Units)"]], use_container_width=True, hide_index=True)

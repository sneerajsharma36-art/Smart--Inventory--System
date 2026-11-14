"""
Database helper module for Smart Inventory and Stock Management System.
Uses SQLite for persistent storage of products and sales transactions.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.db")
SAMPLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sales_data.csv")


def get_connection():
    """Returns a SQLite connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables and seeds initial data if database is empty."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create Products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL DEFAULT 0,
            min_stock INTEGER NOT NULL DEFAULT 5
        )
    """)

    # Create Sales table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            sale_price REAL NOT NULL,
            sale_date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    conn.commit()

    # Check if empty, seed initial data
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        seed_sample_data()


def seed_sample_data():
    """Seeds initial products and sales records from sample CSV."""
    # Default initial products with initial stock and threshold
    initial_products = [
        {"name": "Wireless Mouse", "category": "Electronics", "price": 25.0, "stock": 42, "min_stock": 15},
        {"name": "Mechanical Keyboard", "category": "Electronics", "price": 75.0, "stock": 8, "min_stock": 10},
        {"name": "USB-C Hub", "category": "Accessories", "price": 30.0, "stock": 25, "min_stock": 12},
        {"name": "Noise Cancelling Headphones", "category": "Electronics", "price": 120.0, "stock": 5, "min_stock": 8},
        {"name": "Ergonomic Office Chair", "category": "Furniture", "price": 199.0, "stock": 14, "min_stock": 5},
        {"name": "Aluminum Laptop Stand", "category": "Accessories", "price": 35.0, "stock": 30, "min_stock": 10},
        {"name": "HD Webcam 1080p", "category": "Electronics", "price": 50.0, "stock": 4, "min_stock": 10}
    ]

    conn = get_connection()
    cursor = conn.cursor()

    prod_id_map = {}
    for p in initial_products:
        cursor.execute("""
            INSERT OR IGNORE INTO products (name, category, price, stock, min_stock)
            VALUES (?, ?, ?, ?, ?)
        """, (p["name"], p["category"], p["price"], p["stock"], p["min_stock"]))
        
        cursor.execute("SELECT id FROM products WHERE name = ?", (p["name"],))
        row = cursor.fetchone()
        if row:
            prod_id_map[p["name"]] = row["id"]

    conn.commit()

    # Load sales from sample_sales_data.csv if available
    if os.path.exists(SAMPLE_CSV):
        try:
            df_sales = pd.read_csv(SAMPLE_CSV)
            for _, row in df_sales.iterrows():
                p_name = row["product_name"]
                p_id = prod_id_map.get(p_name, 1)
                cursor.execute("""
                    INSERT INTO sales (product_id, product_name, quantity, sale_price, sale_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    p_id,
                    p_name,
                    int(row["quantity"]),
                    float(row["price"]),
                    str(row["sale_date"])
                ))
            conn.commit()
        except Exception as e:
            print(f"Error seeding sales data: {e}")

    conn.close()


def get_products_df():
    """Returns a pandas DataFrame of all products."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM products ORDER BY name ASC", conn)
    conn.close()
    return df


def add_product(name, category, price, stock, min_stock):
    """Adds a new product to the inventory."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO products (name, category, price, stock, min_stock)
            VALUES (?, ?, ?, ?, ?)
        """, (name.strip(), category.strip(), float(price), int(stock), int(min_stock)))
        conn.commit()
        return True, f"Product '{name}' added successfully!"
    except sqlite3.IntegrityError:
        return False, f"Product '{name}' already exists!"
    except Exception as e:
        return False, f"Failed to add product: {str(e)}"
    finally:
        conn.close()


def update_product(product_id, name, category, price, stock, min_stock):
    """Updates an existing product's details."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE products
            SET name = ?, category = ?, price = ?, stock = ?, min_stock = ?
            WHERE id = ?
        """, (name.strip(), category.strip(), float(price), int(stock), int(min_stock), int(product_id)))
        conn.commit()
        return True, f"Product updated successfully!"
    except Exception as e:
        return False, f"Failed to update product: {str(e)}"
    finally:
        conn.close()


def update_stock(product_id, new_stock):
    """Updates the stock level of a product."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE products SET stock = ? WHERE id = ?", (int(new_stock), int(product_id)))
        conn.commit()
        return True, "Stock updated successfully!"
    except Exception as e:
        return False, f"Failed to update stock: {str(e)}"
    finally:
        conn.close()


def delete_product(product_id):
    """Deletes a product and its associated sales."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM sales WHERE product_id = ?", (int(product_id),))
        cursor.execute("DELETE FROM products WHERE id = ?", (int(product_id),))
        conn.commit()
        return True, "Product removed successfully!"
    except Exception as e:
        return False, f"Failed to delete product: {str(e)}"
    finally:
        conn.close()


def record_sale(product_id, quantity, sale_date=None):
    """Records a sale, decrements stock, and stores the transaction."""
    if sale_date is None:
        sale_date = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, name, price, stock FROM products WHERE id = ?", (int(product_id),))
        product = cursor.fetchone()

        if not product:
            return False, "Product not found!"

        current_stock = product["stock"]
        if current_stock < quantity:
            return False, f"Not enough stock! Available: {current_stock}, Requested: {quantity}"

        new_stock = current_stock - quantity
        cursor.execute("UPDATE products SET stock = ? WHERE id = ?", (new_stock, int(product_id)))

        cursor.execute("""
            INSERT INTO sales (product_id, product_name, quantity, sale_price, sale_date)
            VALUES (?, ?, ?, ?, ?)
        """, (product["id"], product["name"], int(quantity), float(product["price"]), str(sale_date)))

        conn.commit()
        return True, f"Sale recorded! Sold {quantity} unit(s) of '{product['name']}'. Remaining stock: {new_stock}."
    except Exception as e:
        conn.rollback()
        return False, f"Failed to record sale: {str(e)}"
    finally:
        conn.close()


def get_sales_df():
    """Returns a pandas DataFrame of all sales transactions."""
    conn = get_connection()
    query = """
        SELECT s.id, s.product_id, s.product_name, p.category, s.quantity, s.sale_price,
               ROUND(s.quantity * s.sale_price, 2) AS total_amount, s.sale_date
        FROM sales s
        LEFT JOIN products p ON s.product_id = p.id
        ORDER BY s.sale_date DESC, s.id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_low_stock_products():
    """Returns products where current stock is at or below minimum stock threshold."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, name, category, price, stock, min_stock,
               (min_stock - stock) AS deficit
        FROM products
        WHERE stock <= min_stock
        ORDER BY (min_stock - stock) DESC, stock ASC
    """, conn)
    conn.close()
    return df

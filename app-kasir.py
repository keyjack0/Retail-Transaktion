import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model dan data
model = joblib.load("diskon_transaktion.pkl")
data = pd.read_csv("data_fix.csv")

st.set_page_config(page_title="Kasir Prediktif Diskon", layout="wide")
st.title("🛒 Sistem Kasir Prediktif Diskon")

# Inisialisasi session state
if "cart" not in st.session_state:
    st.session_state.cart = []
if "predict_ready" not in st.session_state:
    st.session_state.predict_ready = False

# Input Produk
st.header("📦 Input Produk")
col1, col2 = st.columns(2)
with col1:
    product_category = st.selectbox("Kategori Produk", ["Books", "Clothing", "Electronics", "Home Decor"])
    quantity = st.number_input("Jumlah", min_value=1, value=1)
    price = st.number_input("Harga Satuan", min_value=0.0, value=100.0)

with col2:
    payment_method = st.radio("Metode Pembayaran", ["Cash", "Credit Card", "Debit Card", "PayPal"])
    now = datetime.now()
    day, month, year, hour = now.day, now.month, now.year, now.hour
    st.markdown(f"🕒 Waktu Transaksi: **{day}-{month}-{year} {hour}:00**")

# Tombol Tambah ke Keranjang
if st.button("➕ Tambah ke Keranjang"):
    st.session_state.cart.append({
        "Kategori": product_category,
        "Harga Satuan": price,
        "Jumlah": quantity,
        "Subtotal": price * quantity,
        "Metode": payment_method,
        "Day": day,
        "Month": month,
        "Year": year,
        "Hour": hour
    })
    st.success("Produk ditambahkan ke keranjang!")

# Tampilkan Keranjang
if st.session_state.cart:
    st.subheader("🧾 Keranjang Belanja")
    df_cart = pd.DataFrame(st.session_state.cart)
    st.table(df_cart)

    # Tombol Proses & Prediksi
    if st.button("🔍 Prediksi Diskon & Hitung Total"):
        total_price = df_cart["Subtotal"].sum()
        last_item = st.session_state.cart[-1]

        payment_features = {
            "PaymentMethod_Cash": 1 if last_item["Metode"] == "Cash" else 0,
            "PaymentMethod_Credit Card": 1 if last_item["Metode"] == "Credit Card" else 0,
            "PaymentMethod_Debit Card": 1 if last_item["Metode"] == "Debit Card" else 0,
            "PaymentMethod_PayPal": 1 if last_item["Metode"] == "PayPal" else 0,
        }

        product_features = {
            "ProductCategory_Books": 1 if last_item["Kategori"] == "Books" else 0,
            "ProductCategory_Clothing": 1 if last_item["Kategori"] == "Clothing" else 0,
            "ProductCategory_Electronics": 1 if last_item["Kategori"] == "Electronics" else 0,
            "ProductCategory_Home Decor": 1 if last_item["Kategori"] == "Home Decor" else 0,
        }

        input_data = pd.DataFrame([{
            "Quantity": last_item["Jumlah"],
            "Price": last_item["Harga Satuan"],
            "TotalAmount": last_item["Subtotal"],
            "Day": last_item["Day"],
            "Month": last_item["Month"],
            "Year": last_item["Year"],
            "Hour": last_item["Hour"],
            **payment_features,
            **product_features
        }])

        prediction = model.predict(input_data)[0]
        discounted_total = total_price * (1 - prediction / 100)

        st.subheader("📣 Hasil Prediksi")
        st.success(f"🎯 Diskon yang diprediksi: {prediction:.2f}%")
        st.metric("💰 Total Sebelum Diskon", f"Rp {total_price:,.2f}".replace(",", "."))
        st.success(f"💸 Total Setelah Diskon: Rp {discounted_total:,.2f}".replace(",", "."))

    # Tombol Reset
    if st.button("🔄 Reset Keranjang"):
        st.session_state.cart = []
        st.session_state.predict_ready = False
        st.success("Keranjang dikosongkan.")
else:
    st.info("Keranjang kosong. Tambahkan produk terlebih dahulu.")

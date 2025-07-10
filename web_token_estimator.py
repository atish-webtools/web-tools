import streamlit as st
import tiktoken
import pandas as pd
import requests

# --- Live Exchange Rate ---
def get_usd_to_inr_rate():
    try:
        response = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR")
        data = response.json()
        return data["rates"]["INR"]
    except Exception:
        return 86  # fallback if API fails

USD_TO_INR = get_usd_to_inr_rate()

# --- Pricing Table (July 2025) ---
MODEL_PRICING = {
    "gpt-4o": {"input": 2.5, "output": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.075},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.5-preview": {"input": 75.0, "output": 150.0},
    "o3": {"input": 2.0, "output": 8.0},
    "o3-pro": {"input": 20.0, "output": 80.0},
    "o4-mini": {"input": 1.10, "output": 4.40},
}

st.set_page_config(page_title="OpenAI Token Estimator", layout="wide")
st.title("🧮 OpenAI Token & Cost Estimator")

st.markdown(f"💱 **Live USD → INR Rate**: ₹{USD_TO_INR:.2f}")

# --- Input Area ---
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("🔤 Input Text (Prompt)", height=150)

with col2:
    output_text = st.text_area("📝 Expected Output Text (Model Reply)", height=150)

model = st.selectbox("🤖 Select Model for Your Estimate", list(MODEL_PRICING.keys()), index=0)

# --- Token Counter ---
def count_tokens(text, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# --- Process Estimate ---
if st.button("Estimate Cost"):
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    total_tokens = input_tokens + output_tokens

    input_price_usd = (input_tokens / 1_000_000) * MODEL_PRICING[model]["input"]
    output_price_usd = (output_tokens / 1_000_000) * MODEL_PRICING[model]["output"]
    total_price_usd = input_price_usd + output_price_usd
    total_price_inr = total_price_usd * USD_TO_INR

    row_data = [{
        "Input Tokens": input_tokens,
        "Output Tokens": output_tokens,
        "Total Tokens": total_tokens,
        "Cost (USD)": f"${total_price_usd:.6f}",
        "Cost (INR)": f"₹{total_price_inr:.4f}"
    }]
    usage_df = pd.DataFrame(row_data)

    st.subheader("📊 Your Token Usage & Cost")
    st.dataframe(usage_df, use_container_width=True)

    # --- Pricing Comparison Table ---
    st.subheader("📈 Model Pricing Comparison (per 1M tokens)")
    pricing_df = pd.DataFrame(MODEL_PRICING).T.reset_index()
    pricing_df.columns = ["Model", "Input Price (USD)", "Output Price (USD)"]
    pricing_df["Total Price (USD)"] = pricing_df["Input Price (USD)"] + pricing_df["Output Price (USD)"]
    pricing_df["Total Price (INR)"] = pricing_df["Total Price (USD)"] * USD_TO_INR

    min_price = pricing_df["Total Price (USD)"].min()
    max_price = pricing_df["Total Price (USD)"].max()

    def highlight_extremes(row):
        if row["Total Price (USD)"] == max_price:
            return ['background-color: #ffcccc'] * len(row)  # 🔴 Max
        elif row["Total Price (USD)"] == min_price:
            return ['background-color: #ccffcc'] * len(row)  # 🟢 Min
        else:
            return [''] * len(row)

    styled_pricing_df = pricing_df.style.apply(highlight_extremes, axis=1).format({
        "Input Price (USD)": "${:.2f}",
        "Output Price (USD)": "${:.2f}",
        "Total Price (USD)": "${:.2f}",
        "Total Price (INR)": "₹{:.2f}"
    })

    st.dataframe(styled_pricing_df, use_container_width=True)

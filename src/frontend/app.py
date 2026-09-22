
import os
import streamlit as st
import requests

st.set_page_config(
    page_title="M5 Sales Forecasting",
    page_icon="📈",
    layout="centered"
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8003/forecast"
)

st.title("📈 M5 Sales Forecasting")
st.write("Predict next-day sales for a selected item and store.")

st.subheader("Forecast Input")

item_id = st.text_input(
    "Item ID",
    placeholder="Example: HOBBIES_1_001"
)

store_id = st.text_input(
    "Store ID",
    placeholder="Example: CA_1"
)

if st.button("Forecast"):

    if not item_id or not store_id:
        st.warning("Please enter both Item ID and Store ID.")

    else:
        payload = {
            "item_id": item_id,
            "store_id": store_id
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Forecast generated successfully!")

                st.write("### Forecast Result")

                st.write("**Item ID:**", result["item_id"])
                st.write("**Store ID:**", result["store_id"])

                st.write(
                    "**Last Available Date:**",
                    result["last_available_date"]
                )

                st.write(
                    "**Forecast Date:**",
                    result["forecast_date"]
                )

                st.metric(
                    "Predicted Sales",
                    f"{result['forecast']:.2f}"
                )

            elif response.status_code == 404:

                st.error(
                    "The requested item/store combination was not found."
                )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Could not connect to the forecasting API: {e}"
            )

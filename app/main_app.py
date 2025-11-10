import pandas as pd
import streamlit as st
import sys
import os
import joblib
#modelpaths
modeldir = os.path.dirname(__file__)
modelpath = os.path.abspath(os.path.join(modeldir,"..","models"))
current_dir = os.path.dirname(__file__)
scripts_path = os.path.abspath(os.path.join(current_dir,"..","scripts"))
sys.path.append(scripts_path)
#call dataset
df = pd.read_csv("data/transformed/transformed_clean_car_dataset.csv")
from visuals import (
    plot_price_correlation,
    reconstruct_fuel_type,
    plot_violin_pricedistr,
    plot_scatter_pricedistr,
    plot_line_pricetrend,
    reconstruct_transmission,
    plot_box_pricestrend,
    plot_countcategories,
    plot_histplot,
    reconstruct_sellertype,
    plot_piechart)
#visuals requiring reconstruction
visuals_charts = {
    "🎻 Price Distribution (Violin)": (plot_violin_pricedistr,"violin_price_distribution.png"),
    "📦 Price Box Plot": (plot_box_pricestrend,"boxplot.png"),
    "🧮 Price Count Plot": (plot_countcategories,"countplot_categories.png"),
    "⭕ Price Pie Chart":(plot_piechart,"piechart.png")
}
# use a fxn to Apply reconstruction functions
def reconstruction_engine(df):
    df = reconstruct_fuel_type(df)
    df = reconstruct_transmission(df)
    df = reconstruct_sellertype(df)
    return df
dashboard_charts = {
    "🔥 Heatmap Price Correlation": (plot_price_correlation,"correlation_heatmap.png"),
    "📍 Price Scatter Plot": (plot_scatter_pricedistr,"scatterplot.png"),
    "📈 Price Line Trend Plot": (plot_line_pricetrend,"lineplot.png"),
    "📊 Price Histogram Plot": (plot_histplot, "histplot.png")
}

#slidebar navigation
st.sidebar.title("🧭 -- Navigation -- ")
tab1 = st.sidebar.selectbox(" ➡️ Choose a section: ",["Select"," 📊 Dashboard", " 🔮 Prediction"])
if tab1 == " 📊 Dashboard":
    st.header("Car Price Analysis Dashboard")
    st.subheader("Summary")
    st.markdown("""
🚗 **Sales Overview**  
The majority of cars are sold at low to medium price points, with only a few reaching higher price
 ranges.

⛽ **Fuel Preferences**  
Customers show a strong preference for petrol vehicles, typically willing to pay between 5–10 units.
 Diesel cars, while less common, attract slightly higher prices — some reaching up to 100 units — 
indicating a niche market willing to invest more.

📈 **Insight Opportunity**  
This trend suggests potential for targeted pricing strategies and inventory planning based on fuel
 type and price sensitivity.
""")
    for title, (chart_fn,filename) in dashboard_charts.items():
        with st.expander(title):
            chart_fn(df)
            filepath = os.path.join("visuals",filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.download_button("Download Chart", f, file_name=filename)
    df_reconstructed = reconstruction_engine(df)
    for titles, (charts_fn,filename) in visuals_charts.items():
        with st.expander(titles):
            charts_fn(df_reconstructed)
            filepath = os.path.join("visuals",filename)
            if os.path.exists(filepath):
                with open(filepath,"rb") as f:
                    st.download_button("Download Chart",f,file_name=filename)
elif tab1 == " 🔮 Prediction":
    st.header("Car Price Prediction")
    model_paths = {
        "Linear Regression":"linear_app.pkl",
        "Decision Tree":"decisiontree_app.pkl",
        "Random Forest":"randomforest_app.pkl",
        "XGB Regressor":"xgb_app.pkl"
    }
    models = {name:joblib.load(os.path.join(modelpath,path)) for name,path in model_paths.items()}
    #prediction logic
    st.markdown("### Enter Car Details")
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2015)
    kms = st.number_input("Kilometers Driven", min_value=0, value=50000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
    expected_columns = [
        'Year', 'Kms_Driven',
        'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
        'Seller_Type_Individual',
        'Transmission_Manual'
    ]
    def prepare_input(df,expected_columns):
        df = df.copy()
        df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'])
        # Add missing columns with 0
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns
        df = df[expected_columns]
        return df
    input_df = pd.DataFrame([{
        "Fuel_Type": fuel,
        "Seller_Type": seller,
        "Transmission": transmission,
        "Year": year,
        "Kms_Driven": kms,
    }])
 
    processed_input = prepare_input(input_df,expected_columns)
    # Create tabs dynamically
    tabs = st.tabs(list(model_paths.keys()))
    # Loop through tabs and models
    for tab, model_name in zip(tabs, model_paths.keys()):
        with tab:
            st.subheader(f"{model_name} Prediction")
            mymodel = models[model_name]
            if st.button(f"Predict with {model_name}", key=model_name):
                expected_columns = mymodel.feature_names_in_.tolist()
                processed_input = prepare_input(input_df, expected_columns)
                prediction = mymodel.predict(processed_input)[0]
                st.success(f"💰 Predicted Price: {prediction:.2f} units")
else:
    st.info("Please select an option from the sidebar to begin.")
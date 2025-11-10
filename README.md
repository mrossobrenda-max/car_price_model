# 🚗 Car Price Predictor

A modular, multi-model app that predicts the resale price of used cars based on user inputs like year, mileage, fuel type, and more. Built for deployment clarity, stakeholder trust, and reproducible insights.

---

## 📦 Features

- Multi-model prediction interface (Linear, Decision Tree, Random Forest, XGBoost)
- Modular preprocessing with safe encoding and optional age calculation
- Streamlit GUI for user-friendly interaction
- Transparent, reproducible pipeline with centralized path logic

---

## 🛠️ Tech Stack

- Python
- pandas, scikit-learn, XGBoost
- Streamlit
- joblib

---

## 🚀 How to Run

1. Clone this repository  
   `git clone https://github.com/your-username/car-price-predictor.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Launch the app  
   `streamlit run app.py`

---

## 🧾 Inputs Required

- Year of manufacture
- Kilometers driven
- Fuel type (`Petrol`, `Diesel`)
- Seller type (`Dealer`, `Individual`)
- Transmission (`Manual`, `Automatic`)

---

## 📊 Output

Predicted resale prices from four models, displayed in a clean dashboard with consistent formatting and modular logic.

---

## 📁 Folder Structure
car-price-model/ │ ├── models/              # Saved model .pkl files 
├── scripts/             # Preprocessing and training scripts ├── app.py        # Streamlit GUI ├── transform.py         # Input transformation logic ├── requirements.txt     # Dependencies └── README.md            # Project overview

---

## 👤 Author

Built by Brenda — strategic IT consultant, dashboard architect, and emerging data scientist based in Dar es Salaam. Passionate about stakeholder-ready analytics, modular design, and empowering users through clean, interpretable solutions.

---

## 📌 Notes

- All models expect encoded inputs aligned with training-time preprocessing.
- `Car_Age` can be optionally calculated as `2025 - Year` for explicit depreciation modeling.
- GUI supports consistent prediction display and future expansion for metrics and logging.

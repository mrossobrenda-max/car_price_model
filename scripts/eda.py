import pandas as pd
#load our dataset for cleaning
data = pd.read_csv("data/raw/car_dataset.csv")
#load dataset info
print("\n🔎 -- DataSet Information:")
data.info()
print("\n📊 -- DataSet Statistical Information:")
print(data.describe())
print("\n🚘 -- Top 5 cars:")
print(data.head(5))
#total car inventory
carcount = data['Car_Name'].value_counts()
print("\n📦 -- Car Current Inventory:")
print(carcount)
print(f"✅ -- We have a total {carcount.shape[0]} unique models")
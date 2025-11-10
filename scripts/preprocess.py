import pandas as pd
#now we are loading our raw dataset that we worked on with EDA
#load dataset
df = pd.read_csv("data/raw/car_dataset.csv")
#lets start by looking through our rows and columns
print(f" 🔃 -- Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
#check dataset for null values 
nullvalues = df.isnull().sum()
#use a boolean mask to only check columns with >0 which will give us our actual null columns
if nullvalues.sum()>0:
    print(" 🔎 Colums with empty values\n")
    print(nullvalues)
    #save as csv to nullcolumns.csv
    nullvalues.to_csv("data/cleaned/null_summary.csv",index=False)
    print(" ✅ -- Final null dataset saved to data/cleaned/null_summary.csv --")
    #drop the nulls
    df = df.dropna()
    print(f" ✅ -- Rows with null values dropped -- remaining are {df.shape[0]} rows ")
else:
    print(" ✅ -- No null values found in dataset -- ")
#we proceed to check for duplicate values
duplicates = df.duplicated().sum()
#we already know our flagged duplicates is due to a NaN issue so they arent really duplicates
#dropping them would mislead our analytics
# Fill NaNs and recheck
data_filled = df.fillna("missing")
if duplicates>0:
    print(f" ⚠️ -- Columns with duplicate values are {duplicates}")
    #write the duplicate columns to csv for audit crosschecks
    data_filled['is_duplicate'] = data_filled.duplicated()
    dupvalue = data_filled[data_filled['is_duplicate']]
    #log them to a csv
    dupvalue.to_csv("data/cleaned/duplicates_log.csv",index=False)
    print(" ✅ -- Final duplicate dataset saved to data/cleaned/duplicates_log.csv --")
    #because we know the dataset has NaN flag issues we are not really dropping the 2 rows
    #however if we were really dealing with duplicates then we would drop them as below:
    #df = df.drop_duplicates()
    df.to_csv("data/cleaned/clean_car_dataset.csv",index=False)
    print(" ✅ -- Final clean dataset saved to data/cleaned/clean_car_dataset.csv --")
else:
    print(" ✅ -- There are no duplicate values in dataset --")

import pandas as pd
#now load our cleaned dataset from clean repository
cleandata = pd.read_csv("data/cleaned/clean_car_dataset.csv")
#we can't visualize some visuals because out variables are mixed with strings and integers
#therefore we have to encode first
encoded = pd.get_dummies(cleandata,columns=['Fuel_Type','Seller_Type','Transmission'],drop_first=True)
#assigning x still looks at the entire df so we need to omit the variables that we do not want for our modelling
#here we noticed that the int columns get duplicated therefore to eliminate the issue we explicitly drop them first before
#concat function can do the join and it worked!
columns_to_drop = ['Car_Name','Selling_Price','Owner','Year','Kms_Driven','Present_Price']
x = encoded.drop(columns  = columns_to_drop)  #this just includes my string set i still need the int set as part of
#my independent variables before prediction and visualization 
#now we are pulling the int columns to be part of our x without any duplicate issue
numeric_columns = cleandata[['Year','Kms_Driven']]
#now our final x variables can be joined
final_x_var = pd.concat([x,numeric_columns],axis=1) #we are dealing with columns
#now we deal with our y
y = encoded['Present_Price']
#write our transformed 
transformed_df = pd.concat([final_x_var,y],axis=1)
transformed_df.to_csv("data/transformed/transformed_clean_car_dataset.csv",index=False)
print(" ✅ -- Final transformed dataset saved to data/transformed/transformed_clean_car_dataset.csv --")
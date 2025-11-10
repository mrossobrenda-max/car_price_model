import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os
#fxn to handle heatmap correlations
def plot_price_correlation(df, target='Present_Price', save=False):
    matrix = df.corr()
    price_corr = matrix[target].sort_values(ascending=False)
    corr_df = price_corr.reset_index()
    corr_df.columns = ['Feature', 'Correlation']
    print(f"Correlation with {target}:\n", corr_df)
    #create figure and axis
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm')
    ax.set_title(f"Correlation with {target}")
    plt.xticks(rotation=45)
    #save figure
    if save:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        visual_dr = os.path.join(BASE_DIR, "..", "visuals")
        os.makedirs(visual_dr,exist_ok=True)
        save_path=os.path.join(visual_dr,"correlation_heatmap.png")
        fig.savefig(save_path,bbox_inches='tight')
        print(f" ✅ -- Heatmap has been saved to {save_path} --")
    #plt.show() we need to change this since it is not interactive with streamlit
    #display in streamlit
    st.pyplot(fig)
#reconstruct encoded columns that were for modelling to be able to visualize
def reconstruct_fuel_type(df):
    fuel_cols = ['Fuel_Type_Diesel','Fuel_Type_Petrol']
    #check if all expected columns exist
    missing = [col for col in fuel_cols if col not in df.columns]
    if missing:
        print(f" ⚠️ -- Missing columns for reconstruction : {missing}")
        return df
    # Initialize with 'CNG'
    df['Fuel_Type'] = 'CNG'
    # Overwrite where one-hot columns are active
    df.loc[df['Fuel_Type_Diesel'] == 1, 'Fuel_Type'] = 'Diesel'
    df.loc[df['Fuel_Type_Petrol'] == 1, 'Fuel_Type'] = 'Petrol'
    return df
#fxn to visualize a violinplot to show price distributions
def plot_violin_pricedistr(df,targets = ['Fuel_Type','Present_Price'],save=False):
    x_col,y_col = targets
    #create fig and axis
    fig, ax =plt.subplots(figsize=(10,6))
    sns.violinplot(x=x_col,y=y_col,data=df,order=['Petrol','Diesel','CNG'])
    ax.set_title(f"{y_col} Distribution by {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    #save figure
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualdir,exist_ok=True)
        saveplot = os.path.join(visualdir,"violin_price_distribution.png")
        fig.savefig(saveplot)
        print(f" ✅ -- Violin plot has been successfully saved to {saveplot} --")
    #plt.show()
    #render in streamlit
    st.pyplot(fig)
#fxn to visualize a scatterplot to see how kmsdriven relate with prices
def plot_scatter_pricedistr(df,targets = ['Kms_Driven','Present_Price'], save=False):
    xcol,ycol = targets
    #create fig and axis
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(x=xcol,y=ycol,data=df)
    ax.set_title(f"How {xcol} affect {ycol} Distribution")
    ax.set_xlabel(f"Total {xcol}")
    ax.set_ylabel(f"Current {ycol}")
    #save figure
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualsdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualsdir,exist_ok=True)
        savepath = os.path.join(visualsdir,"scatterplot.png")
        fig.savefig(savepath)
        print(f" ✅ -- Scatterplot has been successfully saved to {savepath} -- ")
    #plt.show()
    st.pyplot(fig)
#fxn to a line plot to show trend btn year and prices
#grouping data with lineplots give us control on aggregation without it seaborn would still group them perfectly 
#however when you need alternate stats like median,mode then the yearavg becomes very useful
#it also helps eliminate hidden assumptions like what if seaborn used count'mode'
# which is default in pivoting to draw the visual thats why explicit instruction below is useful
def plot_line_pricetrend(df,target = ['Year','Present_Price'],save=False):
    xcol,ycol=target
    #create fig and axis 
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(x=xcol,y=ycol,data=df,marker="o",linewidth=2,errorbar=('ci',0))
    ax.set_xlabel("Year of Manufacture")
    ax.set_ylabel("Present Car Price")
    ax.set_title("Price performance trend by Manufacture Year")
    #save fig
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualdir,exist_ok=True)
        savepath = os.path.join(visualdir,"lineplot.png")
        fig.savefig(savepath)
        print(f" ✅ -- You have successfully saved your line chart to {savepath} --")
    #plt.show()
    #rendering streamlit
    st.pyplot(fig)
#reconstruct encoded columns that were for modelling to be able to visualize
def reconstruct_transmission(df):
    transmission_col = ['Transmission_Manual']
    #check if all expected columns exist
    missing_col = [col for col in transmission_col if col not in df.columns]
    if missing_col:
        print(f" ⚠️ -- Missing columns for reconstruction : {missing_col}")
        return df
    #default to 'automatic'
    df['Transmission'] = 'Automatic'
    df.loc[df['Transmission_Manual']==1,'Transmission'] = 'Manual'
    return df
#boxplot to show rxnshp btn transmission type and prices
def plot_box_pricestrend(df,target=['Transmission','Present_Price'],save=False):
    xcol,ycol=target
    #create fig and axis
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x=xcol,y=ycol,data=df)
    ax.set_title("Transmission Types Variations and their Price Ranges")
    ax.set_xlabel("Transmission Type")
    ax.set_ylabel("Present Prices")
    #save figure
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualsdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualsdir,exist_ok=True)
        savepath = os.path.join(visualsdir,"boxplot.png")
        fig.savefig(savepath)
        print(f" ✅ -- The boxplot visual has been successfully saved to {savepath} --")
    #plt.show()
    st.pyplot(fig)
#countplot to show transmission categories
def plot_countcategories(df, targets=['Transmission'],save=False):
    fig,ax=  plt.subplots(figsize=(10,6))
    sns.countplot(x=targets[0],data=df)
    ax.set_title("Transmission Types Distributions")
    ax.set_xlabel("Transmission Types")
    ax.set_ylabel("Counts")
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualdir,exist_ok=True)
        savepath = os.path.join(visualdir,"countplot_categories.png")
        fig.savefig(savepath)
        print(f" ✅ --  You have successfully saved the countplot to {savepath} -- ")
    #plt.show()
    st.pyplot(fig)
#histplot to show Kms_Driven distribution variable
def plot_histplot(df,target = 'Kms_Driven',save=False):
    fig,ax = plt.subplots(figsize=(10,6))
    sns.histplot(data=df,x=target,bins=10,kde=True)
    ax.set_title("Car Kms Inventory")
    ax.set_xlabel("Kms Range Driven")
    ax.set_ylabel("Counts")
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualdir,exist_ok=True)
        savepath = os.path.join(visualdir,"histplot.png")
        fig.savefig(savepath)
        print(f"✅ -- You have successfully saved the histplot in {savepath} -- ")
    #plt.show()
    st.pyplot(fig)
#reconstruct seller_type constructed data to enable plotting
def reconstruct_sellertype(df):
    seller_col = ['Seller_Type_Individual']    
    #check if all expected columns exist
    missing_col = [col for col in seller_col if col not in df.columns]
    if missing_col:
        print(f" ⚠️ There are missing columns for reconstruction in {missing_col}")
        return df
    #default column to dealer seller type
    df['Seller_Type'] = 'Dealer'
    df.loc[df['Seller_Type_Individual'] == 1, 'Seller_Type'] = 'Individual'
    return df
#piechart to show sellertype distribution counts with matplotlib
#however it doesnot accept categorical data so we need to convert 
# it to numeric first by counting total seller types of each category
def plot_piechart(df,target='Seller_Type',save=False):
    fig,ax = plt.subplots(figsize=(10,6))
    #countvariable
    sellercount = df[target].value_counts()
    plt.pie(sellercount,labels=sellercount.index,colors=sns.color_palette("pastel"),autopct='%1.1f%%',startangle=90)
    ax.set_title("Seller Type Distribution")
    if save:
        basedir = os.path.dirname(os.path.abspath(__file__))
        visualdir = os.path.join(basedir,"..","visuals")
        os.makedirs(visualdir,exist_ok=True)
        savepath = os.path.join(visualdir,"piechart.png")
        fig.savefig(savepath)
        print(f"✅ -- You have successfully saved the pie chart in {savepath} -- ")
    #plt.show()
    st.pyplot(fig)
if __name__== "__main__":
    #load dataset after transformation
    df = pd.read_csv("data/transformed/transformed_clean_car_dataset.csv")
    plot_price_correlation(df,save=True)
    df = reconstruct_fuel_type(df)
    plot_violin_pricedistr(df,targets=['Fuel_Type','Present_Price'],save=True)
    plot_scatter_pricedistr(df,targets=['Kms_Driven','Present_Price'],save=True)
    yearavg = df.groupby('Year')['Present_Price'].mean().reset_index()
    plot_line_pricetrend(yearavg, target=['Year','Present_Price'],save=True)
    df = reconstruct_transmission(df)
    plot_box_pricestrend(df,target=['Transmission','Present_Price'],save=True)
    plot_countcategories(df,targets=['Transmission'],save=True)
    plot_histplot(df,target='Kms_Driven',save=True)
    df = reconstruct_sellertype(df)
    plot_piechart(df,target='Seller_Type',save=True)
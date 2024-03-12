#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import os 


# In[2]:


import pandas as pd

# Replace 'latin1' with the correct encoding of your file
df = pd.read_csv(r'C:\Users\Melikzade\Desktop\SQL Course Materials\OnlineRetail.csv', encoding='latin1')


# In[56]:


df.head()


# In[4]:


list(df)


# In[5]:


df.describe()


# There are instances of negative unit prices, which might require investigation.
# A wide standard deviation of 96.76 suggests price variations.
# Presence of negative values indicates returns or cancellations.

# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[41]:


df['Description'].value_counts().head()


# In[39]:


df['Description'] = df['Description'].fillna('UNKNOWN ITEM')
df.isnull().sum()


# In[51]:


df['Description'].value_counts().tail()


# In[45]:


df['Description'].value_counts().tail()


# In[46]:


df[~df['Description'].str.isupper()]['Description'].value_counts().head()


# In[62]:


# Extract relevant features
df['YearMonth'] = df['InvoiceDate'].apply(lambda x: x.strftime('%Y-%m'))

# Group sales data by year and month
sales_by_year_month = df.groupby(['YearMonth'])['Quantity'].sum()

# Convert to a DataFrame
sales_by_year_month = sales_by_year_month.reset_index()

# Extract year and month from YearMonth
sales_by_year_month['Year'] = sales_by_year_month['YearMonth'].apply(lambda x: x[:4])
sales_by_year_month['Month'] = sales_by_year_month['YearMonth'].apply(lambda x: x[5:7])

# Pivot the DataFrame to have years as columns and months as rows
sales_pivot = sales_by_year_month.pivot(index='Month', columns='Year', values='Quantity')

# Add a 0 to the sales_pivot array to match the length of the unique years
years = sorted(sales_pivot.columns)
if len(sales_pivot) < len(years):
    sales_pivot = sales_pivot.reindex(columns=years, fill_value=0)

# Plotting
plt.figure(figsize=(10, 6))
for year in years:
    plt.plot(sales_pivot.index, sales_pivot[year], label=year, linewidth=2)
plt.title('Monthly Sales by Year')
plt.xlabel('Month')
plt.ylabel('Total Sales Quantity')
plt.legend()
plt.show()


# In[43]:


# Find the 10 most sold items
most_sold_items = df['Description'].value_counts().sort_values(ascending=False).head(10)

# Specify a different color palette, for example 'viridis'
custom_palette = sns.color_palette('viridis', 10)

# Plot the 10 most sold items using a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(y=most_sold_items.index, x=most_sold_items.values, orient='h', palette=custom_palette)
plt.xlabel("Counts")
plt.title("Which items most sold?")
plt.yticks(rotation=0)
plt.show()


# In[103]:


# Assuming 'InvoiceDate' is a datetime column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Group by 'Description' and 'InvoiceDate', then sum the quantity
items_over_time = df.groupby(['Description', pd.Grouper(key='InvoiceDate', freq='M')])['Quantity'].sum().reset_index()

# Find the 10 most sold items
top_items = items_over_time.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).index

# Filter the DataFrame for the top 10 items
top_items_data = items_over_time[items_over_time['Description'].isin(top_items)]

# Plot the sales trend over time for each of the top 10 items
plt.figure(figsize=(12, 8))
sns.lineplot(x='InvoiceDate', y='Quantity', hue='Description', data=top_items_data)
plt.title('Top 10 Most Sold Items Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Quantity')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1, 1))
plt.show()


# While the static count of the most sold items indicates 'White Hanging Heart T-Light Holder' as the leader, a nuanced exploration of sales peaks reveals an interesting twist. In reality, the 'Rabbit Night Light' claims the spotlight, experiencing the highest surge in sales during November. This intriguing insight prompts a deeper consideration of customer preferences and seasonal influences. Perhaps, the Rabbit Night Light aligns more closely with holiday themes or specific events, making it a seasonal favorite. Understanding these variations adds a layer of complexity to sales dynamics, urging businesses to delve beyond aggregate statistics and grasp the subtleties that shape consumer choices over time.

# In[57]:


# Identify and filter out cancelled transactions
cancelled_transactions = df[df['InvoiceNo'].str.startswith('C')]

# Find the 10 most cancelled items
most_cancelled_items = cancelled_transactions['Description'].value_counts().sort_values(ascending=False).head(10)

# Specify a color palette for the plot
custom_palette = sns.color_palette('viridis', 10)

# Plot the 10 most cancelled items using a horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(y=most_cancelled_items.index, x=most_cancelled_items.values, orient='h', palette=custom_palette)
plt.xlabel("Cancellation Counts")
plt.title("Top 10 Cancelled Items")
plt.yticks(rotation=0)
plt.show()


# In[58]:


# Identify and filter out cancelled transactions
cancelled_transactions = df[df['InvoiceNo'].str.startswith('C')]

# Find the 10 most cancelled items
most_cancelled_items = cancelled_transactions['Description'].value_counts().sort_values(ascending=False).head(10)

# Find the 10 most sold items
most_sold_items = df['Description'].value_counts().sort_values(ascending=False).head(10)

# Specify color palettes for the plots
cancelled_palette = sns.color_palette('viridis', 10)
sold_palette = sns.color_palette('plasma', 10)

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Plot the top 10 cancelled items
sns.barplot(y=most_cancelled_items.index, x=most_cancelled_items.values, orient='h', palette=cancelled_palette, ax=axes[0])
axes[0].set_xlabel("Cancellation Counts")
axes[0].set_title("Top 10 Cancelled Items")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

# Plot the top 10 sold items
sns.barplot(y=most_sold_items.index, x=most_sold_items.values, orient='h', palette=sold_palette, ax=axes[1])
axes[1].set_xlabel("Sales Counts")
axes[1].set_title("Top 10 Sold Items")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# In[55]:


# Group data by 'InvoiceDate' and sum the 'Quantity' for each date
sales_trends = df.groupby('InvoiceDate')['Quantity'].sum()

# Plotting as a line plot
plt.figure(figsize=(12, 6))
plt.plot(sales_trends.index, sales_trends.values, linewidth=2, color='b')
plt.title('Overall Sales Trends Over Time')
plt.xlabel('Invoice Date')
plt.ylabel('Total Sales Quantity')
plt.grid(True)
plt.show()


# #Seasonal Fluctuations:
# 
# The noticeable spikes in sales at the beginning of the year (around January) and towards the end of the year (around November) suggest the presence of seasonal fluctuations in sales.
# Potential Events or Promotions:
# 
# The spikes in sales may be indicative of specific events or promotions that occurred during these periods. For example, holiday sales, clearance events, or marketing promotions could contribute to increased sales.
# Yearly Sales Trend:
# 
# The overall trend, with fluctuations throughout the year, suggests that sales quantities vary over different periods. This could be influenced by factors such as holidays, shopping seasons, or economic conditions.
# Negative Values on the Y-axis:
# 
# The y-axis ranging from -80,000 to 80,000 suggests that the total sales quantity is not strictly positive, indicating the possibility of refunds, returns, or cancellations contributing to negative sales quantities.

# In[69]:


# Calculate the total sales for each country
sales_by_country = df.groupby('Country')['Quantity'].sum()

# Find the country with the maximum sales
max_sales_country = sales_by_country.idxmax()

# Calculate the maximum sales
max_sales = sales_by_country.max()

# Print the result
print(f"The country with the most sales by quantity is {max_sales_country} with {max_sales} sales.")


# In[82]:


plt.figure(figsize=(18,6))
sns.countplot(x=df[df['Country'] != 'United Kingdom']['Country'])
plt.xticks(rotation=90)


# In[83]:


uk_count = df[df['Country'] == 'United Kingdom']['Country'].count()
all_count = df['Country'].count()
uk_perc = uk_count/all_count
print(str('{0:.2f}%').format(uk_perc*100))


# In[85]:


plt.figure(figsize=(18,6))
plt.scatter(x=df.index, y=df['UnitPrice'])
df = df[df['UnitPrice'] < 25000]
df_quantile = df[df['UnitPrice'] < 125]
plt.scatter(x=df_quantile.index, y=df_quantile['UnitPrice'])
plt.xticks(rotation=90)
df_quantile.describe()


# Quantity:
# The average quantity of items per transaction is approximately 9.57.
# Some transactions have negative quantities, which could represent returns or errors in the data.
# InvoiceDate:
# The data spans from December 1, 2010, to December 9, 2011.
# Most transactions seem to occur around July 4, 2011.
# UnitPrice:
# The average unit price is approximately $3.28.
# There are negative unit prices, which might need further investigation (could be discounts or data errors).
# Day/Month/Year/DayOfWeek:
# These columns break down the InvoiceDate into more specific time units.
# For example:
# Day: Ranges from 1 to 31 (day of the month).
# Month: Ranges from 1 to 12 (January to December).
# Year: Mostly 2011.
# DayOfWeek: Ranges from 1 to 7 (Monday to Sunday).
# Assumptions and Considerations:
# 
# Investigate why there are negative quantities and unit prices (returns, refunds, or data errors).
# The data seems to be centered around July 4, 2011, but further context is needed to understand why.
# The standard deviation (std) provides variability information but doesn’t reveal trends over time.

# In[105]:


df.quantile([0.05, 0.95, 0.98, 0.99, 0.999])
df_quantile = df[df['UnitPrice'] < 125]
plt.scatter(x=df_quantile.index, y=df_quantile['UnitPrice'])
plt.xticks(rotation=90)


# In[106]:


df_quantile.describe()


# In[107]:


plt.figure(figsize=(18,6))



# In[112]:


# PREDICTING FUTURE SALES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Feature Engineering
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek

# Selecting relevant features
features = ['Month', 'Day', 'DayOfWeek', 'Quantity']

# Creating X (features) and y (target)
X = df[features]
y = df['Quantity']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizing the predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Month'], y_test, label='Actual')
plt.scatter(X_test['Month'], y_pred, label='Predicted', marker='x')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.legend()
plt.show()


# In[ ]:





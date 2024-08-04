#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


property_df=pd.read_excel('Book1.xlsx')


# In[4]:


property_df.head(10)


# # data exploration

# In[5]:


property_df.shape


# In[6]:


property_df.describe()


# In[7]:


property_df.info()


# # Handeling missing values

# In[8]:


missing_values = property_df.isnull().sum()


# In[9]:


missing_values


# # Imputing missing values

# In[10]:


property_df.fillna(property_df.mean(), inplace=True)


# In[11]:


property_df


# In[12]:


property_df.isnull().sum()


# # Clean column names by stripping leading/trailing whitespace

# In[13]:


property_df.columns = property_df.columns.str.strip()


# # Save cleaned data

# In[14]:


property_df.to_csv('propertycleaned_data.csv', index=False)


# # EDA

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of property prices
plt.figure(figsize=(10, 6))
sns.histplot(property_df['Sale_Price'], kde=True)
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[25]:


# Box plot of property prices by Building_Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Building_Class', y='Sale_Price', data=property_df)
plt.title('Building_Class vs price')
plt.xlabel('Building_Class')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()


# In[26]:


# Box plot of Sale_Price by zoning_Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Zoning_Class', y='Sale_Price', data=property_df)
plt.title('Zoning_Class vs price')
plt.xlabel('Zoning_Class')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()


# In[18]:


# Check if the columns exist
if 'Lot_Size' in property_df.columns and 'Sale_Price' in property_df.columns:
    # Scatter plot of property size vs. price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Lot_Size', y='Sale_Price', data=property_df)
    plt.title('Property Size vs. Price')
    plt.xlabel('Lot Size (sq ft)')
    plt.ylabel('Price')
    plt.show()
else:
    print("One or both columns 'Lot_Size' and 'Sale_Price' do not exist in the DataFrame.")


# # Correlation analysis

# In[20]:


correlations = property_df.corr()
print(correlations['Sale_Price'].sort_values(ascending=False))


# # feature selection
Key Observations
Strong Positive Correlations:

Overall_Material (0.790972): Indicates that properties with higher quality materials tend to have higher sale prices.
Grade_Living_Area (0.708584): Suggests that the greater the living area, the higher the sale price.
Garage_Size (0.640383): Larger garage sizes are associated with higher sale prices.
Total_Basement_Area (0.613792): A larger basement area correlates with higher sale prices.
First_Floor_Area (0.605971): Indicates that properties with larger first-floor areas tend to have higher prices.
# In[21]:


top_features = ['Overall_Material', 'Grade_Living_Area', 'Garage_Size', 'Total_Basement_Area', 'First_Floor_Area']

for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature, y='Sale_Price', data=property_df)
    plt.title(f'{feature} vs. Sale Price')
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.show()


# In[22]:


top_features_with_price = top_features + ['Sale_Price']
sns.pairplot(property_df[top_features_with_price], diag_kind='kde')
plt.suptitle('Pairplot of Top Features with Sale Price', y=1.02)
plt.show()


# In[23]:


categorical_features = ['Overall_Material']  # Add other categorical features if available

for feature in categorical_features:
   plt.figure(figsize=(10, 6))
   sns.boxplot(x=feature, y='Sale_Price', data=property_df)
   plt.title(f'{feature} vs. Sale Price')
   plt.xlabel(feature)
   plt.ylabel('Sale Price')
   plt.show()


# In[ ]:





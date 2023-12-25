#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm

import math
import time
import re
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[7]:


df_data= pd.read_csv('/Users/chandana/Desktop/data.csv',encoding="ISO-8859-1")


# In[8]:


df_data.head()


# In[10]:


df_data.info()


# In[11]:


df_data.shape


# In[12]:


df_data.isnull().sum()


# In[13]:


df_data[df_data['CustomerID'].isnull()].head(10)


# In[14]:


df_data[df_data['Description'].isnull() & df_data['CustomerID'].isnull()].shape


# In[15]:


## all rows with null description also have null customer id. So, it is safe to drop null values from dataframe
## since customerid is essential to our customer segment analysis.


# In[16]:


clean_data=df_data.dropna()


# In[17]:


clean_data.isnull().sum()


# In[18]:


clean_data.dtypes


# In[19]:


clean_data['Description']=clean_data['Description'].astype(str)


# In[20]:


clean_data.dtypes


# In[21]:


type(clean_data["Description"][1])


# In[22]:


clean_data['InvoiceDate']= pd.to_datetime(clean_data['InvoiceDate'])


# In[23]:


clean_data.dtypes


# In[24]:


clean_data.head(5)


# In[25]:


clean_data['CustomerID']=clean_data['CustomerID'].astype(object)


# In[26]:


clean_data.dtypes


# In[27]:


clean_data.head(5)


# In[49]:


# checking for duplicates


# In[28]:


clean_data[(clean_data['InvoiceNo']== '536365')]


# In[29]:


clean_data[clean_data.duplicated(keep=False) & (clean_data['InvoiceNo'] == '536409')]


# In[30]:


clean_data[clean_data.duplicated() & (clean_data['InvoiceNo'] == '536409')]


# In[31]:


##looking at the duplicate rows, itseems that unique id are reapeated which can be due to error in data entry.
##so, dropping all duplicates from data frame.


# In[32]:


clean_data.drop_duplicates(inplace=True)


# In[33]:


clean_data.shape


# In[34]:


clean_data['InvoiceNo']=clean_data['InvoiceNo'].astype(object)


# In[35]:


# checking for cancelled orders


# In[36]:


count_c = 0
count_neg = 0
for i,j in clean_data.iterrows():
    if j['InvoiceNo'].startswith("C"):
        count_c=count_c+1
        if j['Quantity'] >0:
            print ('this is a positive quantity')
        else:
            count_neg=count_neg+1      
print (f'Number of rows with canceled invoice = {count_c}')
print (f'Number of rows with negative quantity = {count_neg}') 


# In[127]:


# removing all canceled order, as it wont be meaningful for segmentation.


# In[37]:


clean_data[clean_data["Quantity"]<0].shape


# In[38]:


cl_data=clean_data[clean_data["Quantity"]>=0]


# In[39]:


cl_data.shape


# In[40]:


# Adding amount spent column to df
cl_data["AmountSpent"] = cl_data["Quantity"] * cl_data["UnitPrice"]


# In[41]:


cl_data.head()


# In[ ]:


# EDA


# In[203]:


corrMatrix = cl_data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[204]:


sc=cl_data['StockCode'].nunique()
print(f'There are {sc} unique products')


# In[47]:


c_ID=cl_data['CustomerID'].nunique()
print(f'There are {c_ID} customers')


# In[48]:


transaction=cl_data['InvoiceNo'].nunique()
print(f'There are {transaction} transactions')


# In[49]:


pip install tabulate


# In[50]:


from tabulate import tabulate


# In[51]:


table= (["products",sc],["Customers",c_ID],["Transactions",transaction])
print(tabulate(table))
         


# In[52]:


cl_data["StockCode"].value_counts()[:10].index.tolist()


# In[53]:


cl_data['day']=cl_data["InvoiceDate"].apply(lambda x: x.strftime('%d') )


# In[54]:


cl_data['yearmonth'] = cl_data['InvoiceDate'].apply(lambda x: (100*x.year) + x.month)


# In[210]:


# customer geographics:

customer_countries = cl_data.groupby(["Country"])["CustomerID"].sum().sort_values()
customer_countries.plot(kind='barh', figsize=(15,20))
plt.title("Customer Geography")


# In[55]:


# frequency of orders by month


# In[214]:


month=cl_data.groupby(["yearmonth"])["InvoiceNo"].count()
month.plot.bar()
plt.xlabel('Month')
plt.ylabel('Order')
plt.title('Order Vs Month')


# In[57]:


# Revenue by month


# In[213]:


rev=cl_data.groupby(["yearmonth"])["AmountSpent"].sum()
rev.plot.bar()
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Revenue Vs Month')


# In[59]:


# q4 has most sales 


# In[60]:


rev.sort_values()


# In[240]:


cl_data.groupby(["StockCode","Description","UnitPrice"])["Quantity"].count().sort_values(ascending =False)
.reset_index(name='Quantity').head(10)


# In[244]:


cl_data.groupby(["StockCode","Description"])["AmountSpent"].sum().sort_values(ascending =False)
.reset_index(name='Revenue').head(10)


# In[ ]:





# In[ ]:





# In[216]:


df_sort_shift1.head(10)


# In[61]:


df_sort=cl_data.sort_values(["CustomerID","StockCode","InvoiceDate"])
df_sort_shift1=df_sort.shift(1)
df_sort_reorder=df_sort.copy()
df_sort_reorder["reorder"]=np.where(df_sort["StockCode"]==df_sort_shift1["StockCode"],1,0)
df_sort_reorder.head(10)


# In[62]:


# Top reordered items


# In[250]:


df_sort_reorder.groupby(['yearmonth','reorder'])['AmountSpent'].sum().reset_index(name='AmountSpent')
.sort_values('AmountSpent', ascending = False).head(10)


# In[63]:


pd.DataFrame((df_sort_reorder.groupby(['Description'])['reorder'].sum()))
.sort_values('reorder', ascending = False).head(10)


# In[64]:


firstbuy= (df_sort_reorder[df_sort_reorder['reorder']==0].groupby(["yearmonth"])['AmountSpent'].sum())
reorder= (df_sort_reorder[df_sort_reorder['reorder']==1].groupby(["yearmonth"])['AmountSpent'].sum())
retention= pd.DataFrame([firstbuy,reorder],index=['FirstBuy','Reorder']).transpose()


# In[232]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))

retention.plot.bar(stacked=True, ax=axes[0])
retention.plot.box(ax=axes[1])


# In[66]:


pip install wordcloud


# In[67]:


from wordcloud import WordCloud, STOPWORDS 


# In[294]:


products_words = '' 
stopwords = set(STOPWORDS)
for val in cl_data.Description: 
      
    
    val = str(val) 
  
    splitvalue = val.split() 
      
    
    for i in range(len(splitvalue)): 
        splitvalue[i] = splitvalue[i].lower() 
      
    products_words=products_words+" ".join(splitvalue)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(products_words) 
                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

  
plt.show()


# In[68]:


cl_data.quantile([0.05, 0.95, 0.98, 0.99, 0.999])


# In[ ]:





# In[69]:


product=cl_data.groupby("StockCode")["Quantity"].count().sort_values(ascending =False)


# In[233]:


product.head(10).plot.bar()
plt.xlabel('Stockcode')
plt.ylabel('Quantity')
plt.title('Top ordered items')


# In[71]:


cl_data.groupby("StockCode")["CustomerID"].count().sort_values(ascending =False)


# In[192]:


cl_data[cl_data["StockCode"]=="POST"]


# In[72]:


clean_data.shape


# In[73]:


# checking for cancelled orders


# In[74]:


clean_data.dtypes


# In[75]:


clean_data.loc[[0]]


# In[76]:


## RFM analysis


# In[77]:


cl_data['InvoiceDate'].max()
cl_data['InvoiceDate'].min()


# In[78]:


# assigning now date to be one day later the max 'InvoiceDate'
now=pd.to_datetime('2011-12-10 12:00:00')
type(now)


# In[79]:


#creating a separete column having only "data" from "InvoiceDate"
cl_data["date"]=  pd.DatetimeIndex(cl_data['InvoiceDate']).date
cl_data.head(5)


# In[197]:


cl_data.describe()


# In[80]:


#Grouping recency:

# creating a dataframe with customer ID and thier last purchase date.
recent_df= cl_data.groupby(["CustomerID"],as_index=False)["date"].max()


# In[81]:


recent_df.columns = ['CustomerID','LastPurshaceDate']

recent_df['LastPurshaceDate'] = pd.to_datetime(recent_df['LastPurshaceDate'])

recent_df.head(10)


# In[82]:


# calculating recency: Time since last purchase

recent_df['Recency'] = recent_df['LastPurshaceDate'].apply(lambda x: (now - x).days)

recent_df.head(5)


# In[83]:


# Frequency: Total number of purchases

freq_df= cl_data.groupby(["CustomerID"],as_index=False)["InvoiceNo"].count()

freq_df.columns=["CustomerID","Frequency"]

freq_df.head()


# In[84]:


# Monetary value: Total customer spending.

# creating a new columns "total.price"

money_df= cl_data.groupby(["CustomerID"],as_index=False)["AmountSpent"].sum()

money_df.columns=['CustomerID','Monetary']

money_df.head(5)


# In[85]:


# create a RFM table, by merging recency,frequency, and monetary values

first_table= recent_df.merge(freq_df,on="CustomerID")

first_table.head(5)

rfm_df= first_table.merge(money_df, on="CustomerID")


# In[86]:


#dropping "LastPurshaceDate" column off the data as it wont be required further

rfm_df.drop('LastPurshaceDate',axis=1,inplace=True)


# In[87]:


# splitting metrics into segments is by using quartiles ( 0.25,0.5,0.75)
#We assign a score from 1 to 4 to Recency, Frequency and Monetary. 
#Four is the best/highest value, and one is the lowest/worst value. 
#A final RFM score is calculated simply by combining individual RFM score numbers.

quantiles = rfm_df.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[88]:


rfm_segmentation = rfm_df


# In[89]:


#creating RFM Segments
#We will create two segmentation classes since, high recency is bad, while high frequency and monetary value is good

# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


# In[90]:


rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))

rfm_segmentation.head(5)


# In[91]:


#combining R,F,M
rfm_segmentation["RFMScore"]=rfm_segmentation.R_Quartile.map(str)+rfm_segmentation.F_Quartile.map(str)
+rfm_segmentation.M_Quartile.map(str)


# In[234]:


# our best customers are :
#Best Recency score = 4: most recently purchase. 
#Best Frequency score = 4: most quantity purchase. 
#Best Monetary score = 4: spent the most.

# our top 10 of our best customers: 
rfm_segmentation[rfm_segmentation['RFMScore']=='444'].sort_values('Monetary', ascending=False).head(10)


# In[179]:


# how many customers in each segments
objects = ('Best\nCustomers', 'Loyal\nCustomers', 'Big\nSpenders', 'Almost\nLost', 'Lost\nCustomers', 'Lost\nCheap\nCustomers')
y_pos = np.arange(len(objects))
score = [len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']),
         len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]),
         len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]),
         len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']),
         len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']),
         len(rfm_segmentation[rfm_segmentation['RFMScore']=='111'])]


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.bar(y_pos, score, align='center', alpha=0.5, width=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('How many customers are in each segment?')


# In[94]:


from matplotlib_venn import venn3 


# In[95]:


#venn diagram of Segmentation
plt.subplot(122)
req4  = (rfm_segmentation['R_Quartile']==4)
rneq4 = (rfm_segmentation['R_Quartile']!=4)
feq4=(rfm_segmentation['F_Quartile']==4)
fneq4= (rfm_segmentation['F_Quartile']!=4)
meq4=(rfm_segmentation['M_Quartile']==4)
mneq4=(rfm_segmentation['M_Quartile']!=4)
venn3(subsets = (len(rfm_segmentation[req4 & fneq4 & mneq4]),len(rfm_segmentation[(feq4&rneq4&mneq4)]),
                 len(rfm_segmentation[req4&fneq4&mneq4]),len(rfm_segmentation[meq4&rneq4&fneq4]),len(rfm_segmentation[req4&meq4&fneq4]),
                 len(rfm_segmentation[feq4&meq4&rneq4]), len(rfm_segmentation[req4&feq4&meq4])), set_labels = ('Recency', 'Frequency', 'Monetary'), alpha = 0.5);
plt.title('Venn Diagram of RFM Score with Counts')
plt.show()


# In[96]:


rfm_segmentation.head()


# In[97]:


rfm_segmentation.describe()


# In[98]:


plt.figure(figsize=(20,10))
plt.subplot(3,1,1)
sns.distplot(rfm_segmentation["Recency"])

plt.subplot(3,1,2)
sns.distplot(rfm_segmentation["Frequency"])

plt.subplot(3,1,3)
sns.distplot(rfm_segmentation["Monetary"])
plt.show()


# In[99]:


# since RFM is right skewed. Log transformations is used to normalise the data.


# In[100]:


# applying log transformation to Recency
rfm_segmentation["Recency_log"]= np.log(rfm_segmentation["Recency"])


# In[101]:


# applying log transformation to frequency
rfm_segmentation["Frequency_log"]= np.log(rfm_segmentation["Frequency"])


# In[102]:


# applying log transformation to Monetary

rfm_segmentation["Monetary_log"]= np.log(rfm_segmentation["Monetary"])


# In[103]:


# plotting distribution for log transformed data


# In[107]:


plt.figure(figsize=(20,10))
plt.subplot(3,1,1)
sns.distplot(rfm_segmentation["Recency_log"])

plt.subplot(3,1,2)
sns.distplot(rfm_segmentation["Frequency_log"])

plt.subplot(3,1,3)
sns.distplot(rfm_segmentation["Monetary_log"])
plt.show()


# In[105]:


rfm_segmentation = rfm_segmentation[rfm_segmentation["Monetary"]!=0]


# In[106]:


rfm_segmentation[rfm_segmentation["Monetary"]==0]


# In[108]:


# From the plot above, we can see that Recency log transformed data doesn't seem to have ideal normal shape of distribution. 
# So, trying out StandardScaler method from sklearn on log transformed data. 


# In[109]:


rfm_segmentation.head()


# In[110]:


# choosing initial RFM data for StandardScaler method
rfm_initial= rfm_segmentation.iloc[:,:4]
rfm_initial.set_index('CustomerID', inplace=True)


# In[111]:


rfm_initial.head()


# In[112]:


# Log Transformation of initial data
rfm_log = np.log1p(rfm_initial)

# Initializing and fitting standard scaler 
stdscaler = StandardScaler()
stdscaler.fit(rfm_log)
rfm_normal= stdscaler.transform(rfm_log)

# Creating dataframe for clustering
rfm_normal = pd.DataFrame(data = rfm_normal, index = rfm_initial.index, columns = rfm_initial.columns)


# In[113]:


rfm_log.head()


# In[114]:


rfm_normal.mean(axis=0).round(3)


# In[115]:


# plotting to verify if the data are normalised
plt.figure(figsize=(20,10))

plt.subplot(3,1,1)
sns.distplot(rfm_normal["Recency"])

plt.subplot(3,1,2)
sns.distplot(rfm_normal["Frequency"])

plt.subplot(3,1,3)
sns.distplot(rfm_normal["Monetary"])
plt.show()


# In[116]:


# Still the Recency's distribution doesn't look normalised. 
# so, checking the statistic values of the distribution:


# In[117]:


rfm_normal.head()


# In[236]:


print('Mean value of the data: \n\n{}'.format(abs(rfm_normal.mean(axis=0).round(3))))


# In[237]:


print('\nStandard Deviation value of the data: \n\n{}'.format(rfm_normal.std(axis=0).round(3)))


# In[120]:


# this seems to be normally distributed from statistical point of view.


# In[121]:


# Lets, now implement customer segmentation algorithm using k-means clustering
# on the data from RFM analysis


# In[122]:


# Find optimum number of clusters (k) to form using the elbow method:
# Elbow method gives us an idea on 
# what a good k number of clusters would be based on the sum of squared distance (SSE)


# In[123]:


distortions = []
K = range(1,25)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(rfm_normal)
    distortions.append(kmeanModel.inertia_)

# Plot the elbow
plt.figure(figsize=(10,8))
plt.plot(K, distortions, 'bx-')
plt.xticks(np.arange(0,25,step=1))
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method with optimal k')
plt.show() 


# In[124]:


# from the elbow plot above we can see that the optimal number of cluster is 4 is the 
# k=4. Now we can run k means using 4 number of clusters.


# In[154]:


kmeanModel=KMeans(n_clusters=4)
#kmeanModel.fit(rfm_normal)
y=kmeanModel.fit_predict(rfm_normal)
rfm_normal['Cluster']= y


# In[155]:


rfm_normal.shape


# In[156]:


rfm_normal.head()


# In[128]:


from mpl_toolkits.mplot3d import axes3d


# In[129]:


rfm_normal_c0 = rfm_normal[rfm_normal['Cluster']==0]
rfm_normal_c1 = rfm_normal[rfm_normal['Cluster']==1]
rfm_normal_c2 = rfm_normal[rfm_normal['Cluster']==2]
rfm_normal_c3 = rfm_normal[rfm_normal['Cluster']==3]


# In[271]:


fig = plt.figure(figsize=(50,50))
#ax = fig.add_subplot(1,1,1,axisbg='1.0')
ax = fig.gca(projection='3d')
ax.scatter(rfm_normal_c0.Recency,
           rfm_normal_c0.Frequency,
           rfm_normal_c0.Monetary,
           alpha=0.8, c="red", s=30, label="Cluster0")
ax.scatter(rfm_normal_c1.Recency,
           rfm_normal_c1.Frequency,
           rfm_normal_c1.Monetary,
           alpha=0.8, c="blue", s=30, label="Cluster1")
ax.scatter(rfm_normal_c2.Recency,
           rfm_normal_c2.Frequency,
           rfm_normal_c2.Monetary,
           alpha=0.8, c="green", s=30, label="Cluster2")
ax.scatter(rfm_normal_c3.Recency,
           rfm_normal_c3.Frequency,
           rfm_normal_c3.Monetary,
           alpha=0.8, c="violet", s=30, label="Cluster3")
ax.scatter(kmeanModel.cluster_centers_[:,0],
           kmeanModel.cluster_centers_[:,1],
           kmeanModel.cluster_centers_[:,2],
           alpha=0.9, c="black", s=500)
plt.legend(loc=2, fontsize=40, markerscale=5)
ax.set_xlabel('Recency', fontsize=40,labelpad=35)
ax.set_ylabel('Frequency', fontsize=40,labelpad=30)
ax.set_zlabel('Monetary', fontsize=40,labelpad=30)
ax.set_title('Clusters of customers',fontsize=50, pad=50)
plt.tick_params(axis='x', labelsize=35)
plt.tick_params(axis='y', labelsize=35)
plt.tick_params(axis='z', labelsize=35)
plt.show()


# In[466]:





# In[131]:


#Comparing the clustering


# In[132]:


rfm_validate= rfm_normal.sample(frac=0.9)


# In[133]:


rfm_validate.shape


# In[134]:


kmeanModel_val=KMeans(n_clusters=4)

y1=kmeanModel_val.fit_predict(rfm_validate)
rfm_validate['Cluster']= y1


# In[157]:


rfm_validate.head()


# In[158]:


rfm_normal.head()


# In[161]:


rfm_normal.loc[14870.0]


# In[163]:



equal=0
for index, row in rfm_validate.iterrows():
    # Iteration 0:
    # Index = 12824.0
    # Row =  12824.0	0.194108	-0.378394	-0.478274	3
    #print(rfm_normal.loc[index].Cluster)
    #print(row.Cluster)
    if rfm_normal.loc[index].Cluster == row.Cluster:
        equal=equal+1      


# In[144]:


rfm_normal.loc[17164.0].Cluster


# In[164]:


equal


# In[165]:


rfm_validate.shape


# In[168]:


rfm_clusters=rfm_initial.copy()


# In[176]:


rfm_clusters["cluster"]=rfm_normal["Cluster"]


# In[181]:


rfm_clusters.head()


# In[285]:


# Calculate average RFM values for each cluster
rfm_clusters_avg = rfm_clusters.groupby(['cluster']).mean() 

# Calculate average RFM values for the total customer population
#population_avg = rfm_initial.mean()

# Calculate relative importance of cluster's attribute value compared to population
#relative_imp = cluster_avg / population_avg - 1

# Print relative importance scores rounded to 2 decimals
#print(relative_imp.iloc[:, [0, 2, 5]].round(2))


# In[286]:


rfm_clusters_avg.head()


# In[287]:


relative_imp=rfm_clusters_avg/rfm_clusters_avg.mean()


# In[288]:


# Initialize a plot with a figure size of 8 by 2 inches 
plt.figure(figsize = (8, 4))

# Add the plot title
plt.title('Relative importance of attributes')

# Plot the heatmap
sns.heatmap(data = relative_imp, annot = True, fmt='.2f', cmap='RdYlGn')
plt.show()


# In[281]:


Print(f'rfm_clusters[rfm_clusters['cluster']==0].count()


# In[278]:


rfm_clusters[rfm_clusters['cluster']==1].count()


# In[279]:


rfm_clusters[rfm_clusters['cluster']==2].count()


# In[280]:


rfm_clusters[rfm_clusters['cluster']==3].count()


# In[289]:


rfm_clusters_avg_cnt = rfm_clusters_avg


# In[290]:


rfm_clusters_avg_cnt['Count'] = rfm_clusters['cluster'].value_counts()


# In[291]:


rfm_clusters_avg_cnt


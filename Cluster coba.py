import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("E:\Mahendra\KULIAH\OPERATION RESEARCH\PYTHON\Data set\Mall_Customers.csv")
df.head()
print(df)

df.drop(["CustomerID"],axis=1,inplace=True)

#plotgatau
plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
sns.axes_style("dark")
sns.violinplot(y=df["Age"])
plt.show()

#coba diganti sendiri
plt.figure(figsize=(15,6))
plt.title("Ages Frequency")
sns.axes_style("dark")
sns.violinplot(y=df["Age"])
plt.show()

#botplot
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["Spending Score (1-100)"],color="red")
plt.subplot (1,2,2)
sns.boxplot (y=df["Annual Income (k$)"])
plt.show()

#barplot
jenis_kelamin=df.Gender.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=jenis_kelamin.index, y=jenis_kelamin.values)
plt.show()

#barplotage
age18_25=df.Age[(df.Age<=25)&(df.Age>=18)]
age26_35=df.Age[(df.Age<=35)&(df.Age>=26)]
age36_45=df.Age[(df.Age<=45)&(df.Age>=36)]
age46_55=df.Age[(df.Age<=55)&(df.Age>=46)]
age55above=df.Age[df.Age>=56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x,y=y,palette="rocket")
plt.title("Jumlah pelanggan per umur")
plt.xlabel("Umur")
plt.ylabel("Jumlah pelanggan")
plt.show()

#KMeans mengetahui jumlah cluster optimal dengan metode WCSS dan elbow method
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color="blue",marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

#Kmeans age vs annual income
km=KMeans(n_clusters=5)
clusters=km.fit_predict(df.iloc[:,1:])
df["label"]=clusters

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df.Age[df.label==0],df["Annual Income (k$)"][df.label==0],df["Spending Score (1-100)"][df.label==0],c='blue',s=60)
ax.scatter(df.Age[df.label==1],df["Annual Income (k$)"][df.label==1],df["Spending Score (1-100)"][df.label==1],c='red',s=60)
ax.scatter(df.Age[df.label==2],df["Annual Income (k$)"][df.label==2],df["Spending Score (1-100)"][df.label==2],c='green',s=60)
ax.scatter(df.Age[df.label==3],df["Annual Income (k$)"][df.label==3],df["Spending Score (1-100)"][df.label==3],c='orange',s=60)
ax.scatter(df.Age[df.label==4],df["Annual Income (k$)"][df.label==4],df["Spending Score (1-100)"][df.label==4],c='purple',s=60)
ax.view_init(30,185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

#distance centroid cluster
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
kmeans=kmeans.fit(df.iloc[:,1:])
labels=kmeans.predict(df.iloc[:,1:])
centroids=kmeans.cluster_centers_
print(centroids)

#mapping data point cluster membership
km=KMeans(n_clusters=5).fit(df.iloc[:,1:])
cluster_map=pd.DataFrame()
cluster_map['df_index']=df.index.values
cluster_map['cluster']=km.labels_
cluster_map[cluster_map.cluster==5]
print(cluster_map)





                                        

                                          

              

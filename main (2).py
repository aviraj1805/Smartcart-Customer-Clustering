# Generated from: main.ipynb
# Converted at: 2026-09-19T09:42:40.114Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("smartcart_customers.csv")

df.head()

df.isnull().sum()

# Data Preprocessing

df["Income"] = df["Income"].fillna(df["Income"].median)

df.isnull().sum()

# Feature Enginnering

df["Age"] = 2026-df["Year_Birth"]

df.head()

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"],dayfirst=True)

reference = df["Dt_Customer"].max()

df["cust_spending_days"] = (reference-df["Dt_Customer"]).dt.days

df.head()

df.columns

df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +df["MntFishProducts"] +df["MntSweetProducts"] + df["MntGoldProds"]

df["Total_childs"] = df["Kidhome"] + df["Teenhome"]

df.head()

df["Education"].value_counts()

df["Education"] = df["Education"].replace(
    {
    "Basic" : "Undergraduate",
    "2n Cycle" : "Undergraduate",
    "Graduation" : "Graduate",
    "Master" : "Post-Graduate",
    "PhD" : "Post-Graduate"
    }
)

df.head()

df["Marital_Status"].value_counts()

df["living_with"] = df["Marital_Status"].replace(
    {
    "Married" : "partner",
    "Together" : "partner",
    "Single" : "single",
    "Divorced" : "single",
    "Widow" : "single",
    "Alone" : "single",
    "Absurd" : "single",
    "YOLO" : "single"
    }
)

df.head()

df.columns

df = df.drop(columns=["ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"],axis=1)

df.shape

df.columns

cols = ["Income","Recency","Age","Total_Spending","Total_childs","living_with"]

sns.pairplot(df[cols])

df["Income"] = pd.to_numeric(df["Income"], errors='coerce')
print("with Outlier count: ",len(df))
df = df[ ( df["Age"] < 90 ) ]
df = df[ ( df["Income"] < 600_000 ) ]
print("without Outlier count: ",len(df))

# Co relation Heatmap

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,7))

sns.heatmap(
    corr,
    annot=True,
    annot_kws = {"size":8},
    linecolor="black",
    cmap="coolwarm"
)



from sklearn.preprocessing import OneHotEncoder

df.head()

cols = ["Education","living_with"]

ohe = OneHotEncoder()

ohe_data = ohe.fit_transform(df[cols]).toarray()

ohe_df = pd.DataFrame(ohe_data,columns=ohe.get_feature_names_out(cols))

df = df.drop(columns=cols,axis=1)

df = pd.concat([df,ohe_df],axis=1)

df.shape

df.head()

# Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(df)

x_scaled

from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy='mean')

x_imputed = imputer.fit_transform(x_scaled)

pca = PCA(n_components=9)
x_pca = pca.fit_transform(x_imputed)

print(sum(pca.explained_variance_ratio_)*100)

# plot

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2])

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3d projection")

pip install kneed

from sklearn.cluster import KMeans
from kneed import KneeLocator

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(x_pca)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
print("optimal K:",knee.elbow)

plt.plot(range(1, 11), wcss, marker='o',c="red")
plt.xlabel("K")
plt.ylabel("WCSS")

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(x_pca)
    score = silhouette_score(x_pca, labels)
    scores.append(score)

plt.plot(range(2, 11), scores, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette score")

#Combine plot

k_range = range(2, 11)

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(k_range, wcss[:len(k_range)], marker="o", color="blue") 
ax1.set_xlabel("K")
ax1.set_ylabel("WCSS")

ax2 = ax1.twinx()
ax2.plot(k_range, scores[:len(k_range)], marker="o", color="red", linestyle="--")
ax2.set_ylabel("SS")

# K_means

kmeans = KMeans(n_clusters=7, random_state=42)
labels_kmeans = kmeans.fit_predict(x_pca)

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection="3d")

ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=labels_kmeans)

# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

agg_clf = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels_agg = agg_clf.fit_predict(x_pca)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=labels_agg)

df["cluster"] = labels_agg

df.head()

pal = ["red", "blue", "yellow", "green"]

sns.countplot(x=df["cluster"], palette=pal, hue=df["cluster"])

sns.scatterplot(x=df["Total_Spending"], y=df["Income"], hue=df["cluster"], palette=pal)

# Cluster Summary

cluster_summary = df.groupby("cluster").mean()
print(cluster_summary)
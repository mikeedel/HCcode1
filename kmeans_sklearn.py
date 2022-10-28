# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox.csv")
# %%
dfp = df[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]]

ssd = []
ks = range(1,11)
for k in range(1,11):
    km = KMeans(n_clusters=k)
    km = km.fit(dfp)
    ssd.append(km.inertia_)

kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()

k = round(kneedle.knee)

print(f"Number of clusters suggested by knee method: {k}")
# %%
kmeans = KMeans(n_clusters=k).fit(df[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]])
#sns.scatterplot(data=df, x="Baby%", y="Home%", hue=kmeans.labels_)
#plt.show()

# %%
df["cluster"] = kmeans.labels_
cluster0 = df[df.cluster == 0]
sns.boxplot(data=pd.melt(cluster0[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]]), x="variable", y="value")
#cluster0.describe()
# %%
df["cluster"] = kmeans.labels_
sns.boxplot(data=df, x="cluster", y="Home%")

# %%

from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier()
tree.fit(df[["Age", "Annual_Income_(k$)", "Spending_Score"]], kmeans.labels_)
print(export_text(tree, feature_names=["Age", "Annual_Income_(k$)", "Spending_Score"]))

# %%
cluster0 = df[df.cluster == 3]
sns.boxplot(data=pd.melt(cluster0[["Food%","Fresh%","Drinks%","Home%","Beauty%","Health%","Baby%","Pets%"]]), x="variable", y="value")
cluster0.describe()
# %%

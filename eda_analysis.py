import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris_eda_dataset.csv")

print("DATASET SHAPE:", df.shape)
print("\nFIRST 5 ROWS:")
print(df.head())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nCORRELATION MATRIX:")
print(df.select_dtypes("number").corr())

# Species counts
df["species"].value_counts().plot(kind="bar", title="Number of Samples by Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Scatter plot
for species in df["species"].unique():
    s = df[df["species"] == species]
    plt.scatter(s["sepal_length"], s["sepal_width"], label=species)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length vs Sepal Width")
plt.legend()
plt.tight_layout()
plt.show()

# Box plot
df[["sepal_length","sepal_width","petal_length","petal_width"]].boxplot()
plt.title("Distribution of Iris Measurements")
plt.tight_layout()
plt.show()

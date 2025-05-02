# pandaya
# IRIS DATASET ANALYSIS: COMPLETE SCRIPT
# Combines Tasks 1 (Exploration), 2 (Analysis), and 3 (Visualization)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# TASK 1: LOAD AND EXPLORE THE DATASET
# ============================================

print("\n" + "="*50)
print("TASK 1: LOADING AND EXPLORING THE DATASET")
print("="*50 + "\n")

# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

# Display first 5 rows
print("First 5 rows:")
print(iris.head())

# Dataset structure
print("\nDataset info:")
print(iris.info())

# Missing values
print("\nMissing values per column:")
print(iris.isnull().sum())

# Clean dataset (though iris is typically clean)
iris_clean = iris.dropna()
print(f"\nOriginal shape: {iris.shape}, Cleaned shape: {iris_clean.shape}")

# ============================================
# TASK 2: BASIC DATA ANALYSIS
# ============================================

print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50 + "\n")

# Basic statistics
print("Descriptive statistics for numerical columns:")
print(iris_clean.describe())

# Group by species
print("\nMean values by species:")
print(iris_clean.groupby('species').mean())

# Interesting findings
print("\nKey Observations:")
print("- Setosa has smallest measurements (especially petals)")
print("- Virginica has largest measurements overall")
print("- Versicolor is intermediate in most measurements")

# ============================================
# TASK 3: DATA VISUALIZATION
# ============================================

print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50 + "\n")

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# 1. Line Chart (Sepal Length Trend)
plt.subplot(2, 2, 1)
for species in iris_clean['species'].unique():
    subset = iris_clean[iris_clean['species'] == species]
    plt.plot(subset.index, subset['sepal_length'], label=species)
plt.title('Sepal Length Trend by Species')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# 2. Bar Chart (Average Petal Length)
plt.subplot(2, 2, 2)
sns.barplot(x='species', y='petal_length', data=iris_clean, estimator='mean')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')

# 3. Histogram (Sepal Width Distribution)
plt.subplot(2, 2, 3)
sns.histplot(iris_clean['sepal_width'], bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# 4. Scatter Plot (Sepal vs Petal Length)
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris_clean)
plt.title('Sepal vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

plt.tight_layout()
plt.show()

print("\nVisualizations created successfully!")
print("1. Line chart showing sepal length trends")
print("2. Bar chart comparing petal lengths")
print("3. Histogram of sepal width distribution")
print("4. Scatter plot of sepal vs petal length")

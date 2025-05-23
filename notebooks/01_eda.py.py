# %% [markdown]
# # 📊 Multitask Feedback EDA
# Exploratory Data Analysis for multitask sentiment and category classification.
# We'll examine the dataset to understand tone distribution, category frequency, and tone-category relationships.

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# 🔽 Load data
df = pd.read_csv(r"C:\Users\StasAndLiza\port\nlp\data\raw\dataset.csv")

# %% [markdown]
# ## 🧼 Preprocessing
# We rename columns to more intuitive names and remove duplicates.

# %%
df = df.rename(columns={
    "comment_txt": "text",
    "sentiment_txt": "tone",
    "parent_class_txt": "level_1",
    "child_class_txt": "level_2"
})

# Normalize tone column: remove "SENTIMENT_" prefix and capitalize values
df["tone"] = df["tone"].str.replace("SENTIMENT_", "", regex=False).str.capitalize()

# Drop exact duplicate records based on key columns
df = df.drop_duplicates(subset=["id", "text", "tone", "level_1", "level_2"])

# %% [markdown]
# ## 🎯 Sentiment Distribution
# Visualizing how often each sentiment class appears.

# %%
sns.countplot(data=df, x="tone", order=df["tone"].value_counts().index)
plt.title("Sentiment Distribution")
plt.xlabel("Tone")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## 🧭 Top Parent Categories (Level 1)
# Most common high-level complaint topics.

# %%
df["level_1"].value_counts().head(10).plot(kind="barh", title="Top Level 1 Categories")
plt.xlabel("Count")
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# ## 🪜 Top Child Categories (Level 2)
# Most frequent detailed complaint topics.

# %%
df["level_2"].value_counts().head(15).plot(kind="barh", title="Top Level 2 Categories")
plt.xlabel("Count")
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# ## 📌 Tone vs. Level 1 Category
# Analyze how sentiment is distributed within each major complaint category.

# %%
tone_cat = pd.crosstab(df["level_1"], df["tone"])  # Count of tone per category
tone_cat = tone_cat.div(tone_cat.sum(axis=1), axis=0)  # Convert to proportions per row

tone_cat.plot(kind="barh", stacked=True, colormap="coolwarm")
plt.title("Tone Distribution by Category (Level 1)")
plt.xlabel("Proportion")
plt.legend(title="Tone", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# ✅ Data is clean and well understood. We're ready to proceed to vectorization and multitask modeling!

"""main.py"""

## Step 1: Environment Setup

# step 1: import all modules

#------- complete the step-------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import datetime
from scipy import stats
#------- complete the step-------------

# Configuration
sns.set(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None

# Output folders
os.makedirs('images', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("Environment setup complete!")


## Step 2: Data Loading

# step 1: read all csv files into list and assign to csv_files
#------- complete the step-------------
csv_files = glob.glob('data/*.csv')
csv_files = [f for f in csv_files if 'videos' in f.lower() and 'cleaned' not in f.lower()]
#------- complete the step-------------
print(f"Found {len(csv_files)} CSV files")

# Load each CSV with country code
dataframes = []

# step 2: read csv files into dataframe and append dataframe into dataframes
for file in csv_files:
    # Extract country code from filename and assign to country_code
    #------- complete the step-------------
    country_code = file.replace('data/', '').replace('data\\', '').split('videos')[0].upper()
    #------- complete the step-------------
    print(country_code)

    # read file to dataframe
    #------- complete the step-------------
    df = pd.read_csv(file, encoding='utf-8')
    #------- complete the step-------------

    # adding a new column to the DataFrame (df) that identifies the country
    #------- complete the step-------------
    df['country'] = country_code
    #------- complete the step-------------

    dataframes.append(df)

# Verify loading
print("\nSample data from first dataframe:")
print(dataframes[0].head(2))


## Step 3: Data Cleaning

"""
Step 1: define a function clean_dataframe takes a dataframe parameter. 
        the function will convert columns : ['video_id', 'title', 'channel_title', 'category_id', 'tags', 'thumbnail_link']
        into string, and handling missing Tags
"""
def clean_dataframe(df):
    """Clean and standardize a single dataframe"""

    # Convert specified columns to string type
    str_cols = ['video_id', 'title', 'channel_title', 'category_id', 'tags', 'thumbnail_link']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Handle missing / placeholder tags
    df['tags'] = df['tags'].replace('[none]', '').replace('nan', '').fillna('')

    return df

# Clean all dataframes
cleaned_dataframes = [clean_dataframe(df) for df in dataframes]

# Verify cleaning
print("\nData types after cleaning:")
print(cleaned_dataframes[0].dtypes)


## Step 4: Missing Value Analysis

"""
Step 1: define a function analyze_missing_data
        the function returns a missing report that contains country code, missing count and missing percent
"""

def analyze_missing_data(df_list, country_codes):
    """Analyze missing values across all dataframes"""
    records = []
    for df, code in zip(df_list, country_codes):
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = round(missing_count / len(df) * 100, 2)
            records.append({
                'country': code,
                'column': col,
                'missing_count': missing_count,
                'missing_percent': missing_pct,
            })
    return pd.DataFrame(records)

# defines a list contains all country code.
country_codes = [os.path.basename(f).split('videos')[0].upper() for f in csv_files]

missing_report = analyze_missing_data(cleaned_dataframes, country_codes)

# Display results
print("Missing value summary:")
print(missing_report[missing_report['missing_count'] > 0])


## Step 5: Data Integration

# Step 1: Combine all dataframes
#------- complete the step-------------
combined_df = pd.concat(cleaned_dataframes, ignore_index=True)
#------- complete the step-------------

# Step 2: Create backup before deduplication
#------- complete the step-------------
combined_df_backup = combined_df.copy()
#------- complete the step-------------

# Step 3: Remove duplicate videos (keeping first occurrence) by video_id
#------- complete the step-------------
combined_df = combined_df.drop_duplicates(subset='video_id', keep='first')
#------- complete the step-------------

# Step 4: Set video_id as index
#------- complete the step-------------
combined_df = combined_df.set_index('video_id')
#------- complete the step-------------

# Final dataset info
print("\nCombined dataset information:")
print(f"Total videos: {len(combined_df)}")
print(f"Countries: {combined_df['country'].unique().tolist()}")


## Step 6: Feature Engineering

# Engagement metrics
# Step 1: add a column named 'like_ratio', computed as likes/dislikes
#------- complete the step-------------
combined_df['like_ratio'] = combined_df['likes'] / (combined_df['dislikes'] + 1)
#------- complete the step-------------

# Step 2: add a column named 'engagement_rate', computed as (likes + dislikes + comment_total)/views
#------- complete the step-------------
combined_df['engagement_rate'] = (
    combined_df['likes'] + combined_df['dislikes'] + combined_df['comment_total']
) / (combined_df['views'] + 1)
#------- complete the step-------------


# Text features
# Step 3: add a column named 'title_length', computed as length of title
#------- complete the step-------------
combined_df['title_length'] = combined_df['title'].str.len()
#------- complete the step-------------

# Step 4: add a column named 'title_word_count', computed as number of words in title
#------- complete the step-------------
combined_df['title_word_count'] = combined_df['title'].str.split().str.len()
#------- complete the step-------------

# Step 5: add a column named 'title_has_exclamation', computed as bool value (you can make it 0 or 1) of if title contains exclamation mark
#------- complete the step-------------
combined_df['title_has_exclamation'] = combined_df['title'].str.contains('!').astype(int)
#------- complete the step-------------

# Tag analysis
# Step 6: add a column named 'tags_count', computed as number of tags
#------- complete the step-------------
combined_df['tags_count'] = combined_df['tags'].apply(
    lambda x: len(x.split('|')) if x != '' else 0
)
#------- complete the step-------------


# Display new features
print("\nNew features created:")
print(combined_df[['like_ratio', 'engagement_rate',
                    'title_length', 'tags_count']].describe())


## Step 7: Exploratory Data Analysis

# Set up the figure
plt.figure(figsize=(18, 12))

# Plot 1: Views vs. Likes
# Step 1: make a dot plot shows views and likes. Take 1000 sample from all countries. x-axis will be views and y-axies will be views.
#         using different colors indicate different country. Add legend and plot title.

#------- complete the step-------------
plt.subplot(1, 2, 1)
sample = combined_df.sample(min(1000, len(combined_df)), random_state=42)
for country, group in sample.groupby('country'):
    plt.scatter(group['views'], group['likes'], label=country, alpha=0.5, s=20)
plt.xlabel('Views')
plt.ylabel('Likes')
plt.title('Views vs. Likes (1000 sample)')
plt.legend(title='Country')
#------- complete the step-------------

# Plot 2: Engagement Rate by Country
# Step 2: make a box plot shows different country engagement_rate
#------- complete the step-------------
plt.subplot(1, 2, 2)
sns.boxplot(data=combined_df, x='country', y='engagement_rate')
plt.title('Engagement Rate by Country')
plt.xlabel('Country')
plt.ylabel('Engagement Rate')
#------- complete the step-------------

plt.tight_layout()
plt.savefig('images/eda_plots.png', dpi=150)
plt.close()
print("Saved -> images/eda_plots.png")


## Step 8: Country-Specific Analysis

# Step 1: define a function that makes bar plot of country's views and dislike distributions

def analyze_country(df, country_code):
    """Generate country-specific analysis"""

    #------- complete the step-------------
    country_df = df[df['country'] == country_code]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{country_code} - Views & Dislikes Distribution', fontsize=13)

    # Top 10 channels by total views
    top_views = (
        country_df.groupby('channel_title')['views']
        .sum()
        .nlargest(10)
        .reset_index()
    )
    axes[0].barh(top_views['channel_title'], top_views['views'], color='steelblue')
    axes[0].set_title('Top 10 Channels by Views')
    axes[0].set_xlabel('Total Views')
    axes[0].invert_yaxis()

    # Top 10 channels by total dislikes
    top_dislikes = (
        country_df.groupby('channel_title')['dislikes']
        .sum()
        .nlargest(10)
        .reset_index()
    )
    axes[1].barh(top_dislikes['channel_title'], top_dislikes['dislikes'], color='tomato')
    axes[1].set_title('Top 10 Channels by Dislikes')
    axes[1].set_xlabel('Total Dislikes')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'images/country_{country_code}.png', dpi=150)
    plt.close()
    print(f"Saved -> images/country_{country_code}.png")
    #------- complete the step-------------

# Analyze sample countries
for country in ['US', 'GB']:
    analyze_country(combined_df, country)


## Step 9: Advanced Insights

# Step 1: can you think of a analysis and visualize it?
# Analysis: Does having an exclamation mark in the title lead to more views?

exclaim    = combined_df[combined_df['title_has_exclamation'] == 1]['views']
no_exclaim = combined_df[combined_df['title_has_exclamation'] == 0]['views']

t_stat, p_val = stats.ttest_ind(exclaim, no_exclaim, equal_var=False)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: Statistically significant difference in views between titles with and without '!'")
else:
    print("Result: No statistically significant difference.")

avg_views = combined_df.groupby('title_has_exclamation')['views'].mean()
labels = ['No Exclamation', 'Has Exclamation']

plt.figure(figsize=(7, 5))
plt.bar(labels, avg_views.values, color=['steelblue', 'orange'])
plt.title('Average Views: Exclamation Mark in Title?')
plt.ylabel('Average Views')
plt.tight_layout()
plt.savefig('images/insight_exclamation.png', dpi=150)
plt.close()
print("Saved -> images/insight_exclamation.png")


## Step 10: Saving Results

# Step 1: Save cleaned data into csv file
#------- complete the step-------------
combined_df.to_csv('output/cleaned_youtube_trending_data.csv')
#------- complete the step-------------

print("Saved cleaned data to 'output/cleaned_youtube_trending_data.csv'")

# Step 2: Save top 5 global category bar plot visualizations
#------- complete the step-------------
top_categories = (
    combined_df.groupby('category_id')['views']
    .sum()
    .nlargest(5)
    .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(top_categories['category_id'].astype(str),
        top_categories['views'], color='mediumseagreen')
plt.title('Top 5 Global Categories by Total Views')
plt.xlabel('Category ID')
plt.ylabel('Total Views')
plt.tight_layout()
plt.savefig('output/top_categories.png', dpi=150)
plt.close()
#------- complete the step-------------

print("Saved visualization to 'output/top_categories.png'")
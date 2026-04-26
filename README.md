# YouTube Video Data Analysis

Analyzes trending YouTube video data from the US and GB to explore patterns in views, likes, engagement, and categories.

## Files

- `main.py` — main script that runs the full analysis
- `requirements.txt` — required Python packages
- `data/` — contains `USvideos.csv` and `GBvideos.csv`

## Output Files

- `output/cleaned_youtube_trending_data.csv` — cleaned and combined dataset
- `output/top_categories.png` — top 5 global categories by total views
- `images/eda_plots.png` — views vs likes scatter plot and engagement rate boxplot
- `images/country_US.png` / `images/country_GB.png` — top 10 channels by views and dislikes per country
- `images/insight_exclamation.png` — analysis of exclamation marks in titles vs views

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run the script:
```
python main.py
```

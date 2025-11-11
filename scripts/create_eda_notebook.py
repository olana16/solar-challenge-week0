"""Script to create a Jupyter notebook for solar data EDA."""

import json
import os
from pathlib import Path

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Solar Data EDA - Benin\n",
                "\n",
                "## Introduction\n",
                "This notebook performs exploratory data analysis (EDA) on solar data from Benin. "
                "The analysis includes data profiling, cleaning, visualization, and statistical analysis "
                "to understand the solar energy potential in the region."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup and Data Loading\n",
                "Let's import the necessary libraries and load the dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Standard library imports\n",
                "import os\n",
                "import warnings\n",
                "\n",
                "# Data manipulation\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from scipy import stats\n",
                "\n",
                "# Visualization\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "from plotly.subplots import make_subplots\n",
                "\n",
                "# Configuration\n",
                "warnings.filterwarnings('ignore')\n",
                "pd.set_option('display.max_columns', 100)\n",
                "plt.style.use('seaborn')\n",
                "%matplotlib inline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Update this path to your actual data file\n",
                "data_path = '../data/benin_data.csv'\n",
                "\n",
                "# Load the data\n",
                "try:\n",
                "    df = pd.read_csv(data_path, parse_dates=['Timestamp'])\n",
                "    print(f'Data loaded successfully. Shape: {df.shape}')\n",
                "    display(df.head())\n",
                "except Exception as e:\n",
                "    print(f'Error loading data: {e}') \n",
                "    # Create a sample dataframe if the file doesn't exist\n",
                "    print('Creating a sample dataframe for demonstration purposes...')\n",
                "    dates = pd.date_range('2023-01-01', periods=100, freq='H')\n",
                "    np.random.seed(42)  # For reproducibility\n",
                "    \n",
                "    df = pd.DataFrame({\n",
                "        'Timestamp': dates,\n",
                "        'GHI': np.random.normal(400, 100, 100).clip(0),\n",
                "        'DNI': np.random.normal(500, 150, 100).clip(0),\n",
                "        'DHI': np.random.normal(300, 80, 100).clip(0),\n",
                "        'Tamb': np.random.normal(25, 5, 100),\n",
                "        'RH': np.random.uniform(30, 90, 100),\n",
                "        'WS': np.random.uniform(0, 10, 100),\n",
                "        'WD': np.random.uniform(0, 360, 100),\n",
                "        'WSgust': np.random.uniform(5, 15, 100),\n",
                "        'ModA': np.random.normal(30, 5, 100).clip(0),\n",
                "        'ModB': np.random.normal(32, 4, 100).clip(0),\n",
                "        'BP': np.random.normal(1013, 5, 100)\n",
                "    })\n",
                "    display(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Profiling\n",
                "Let's perform initial data exploration to understand the structure and quality of our dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Basic information about the dataset\n",
                "print('\\n=== Dataset Info ===')\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Summary statistics\n",
                "print('\\n=== Summary Statistics ===')\n",
                "display(df.describe().T)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check for missing values\n",
                "print('\\n=== Missing Values ===')\n",
                "missing = df.isna().sum()\n",
                "missing_pct = (missing / len(df)) * 100\n",
                "missing_df = pd.concat([missing, missing_pct], axis=1)\n",
                "missing_df.columns = ['Missing Values', 'Percentage']\n",
                "display(missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Cleaning\n",
                "Let's clean the data by handling missing values and outliers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a copy of the original dataframe for reference\n",
                "df_clean = df.copy()\n",
                "\n",
                "# Function to detect outliers using Z-score\n",
                "def detect_outliers_zscore(data, columns, threshold=3):\n",
                "    """Detect outliers using Z-score method."""\n",
                "    outliers = pd.DataFrame(False, index=data.index, columns=columns)\n",
                "    for col in columns:\n",
                "        z_scores = np.abs(stats.zscore(data[col].dropna()))\n",
                "        outliers[col] = z_scores > threshold\n",
                "    return outliers\n",
                "\n",
                "# Columns to check for outliers\n",
                "numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()\n",
                "outlier_cols = [col for col in numeric_cols if col not in ['Timestamp']]\n",
                "\n",
                "# Detect outliers\n",
                "outliers = detect_outliers_zscore(df_clean, outlier_cols)\n",
                "outlier_counts = outliers.sum().sort_values(ascending=False)\n",
                "\n",
                "print('Number of outliers detected per column:')\n",
                "display(outlier_counts[outlier_counts > 0])\n",
                "\n",
                "# Mark rows with any outlier for visualization\n",
                "df_clean['has_outlier'] = outliers.any(axis=1)\n",
                "print(f'Total rows with at least one outlier: {df_clean[\"has_outlier\"].sum()}')\n",
                "\n",
                "# Option 1: Remove outliers\n                "# df_clean = df_clean[~df_clean['has_outlier']]\n",
                "\n",
                "# Option 2: Impute outliers with median (commented out by default)\n                "# for col in outlier_cols:\n",
                "#     median_val = df_clean[col].median()\n",
                "#     df_clean.loc[outliers[col], col] = median_val\n",
                "\n",
                "# Handle missing values\n",
                "# For demonstration, we'll forward fill then backfill any remaining NAs\n",
                "df_clean = df_clean.ffill().bfill()\n",
                "\n",
                "# Verify no missing values remain\n",
                "print('\\n=== Missing values after cleaning ===')\n",
                "print(df_clean.isna().sum().sum(), 'missing values remaining')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Time Series Analysis\n",
                "Let's explore how the solar metrics vary over time."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set Timestamp as index for time series analysis\n",
                "ts_df = df_clean.set_index('Timestamp')\n",
                "\n",
                "# Resample to daily frequency for smoother visualization\n",
                "daily_df = ts_df.resample('D').mean()\n",
                "\n",
                "# Plot time series of key metrics\n",
                "fig = make_subplots(\n",
                "    rows=3, cols=1,\n",
                "    subplot_titles=('Global Horizontal Irradiance (GHI)', 'Ambient Temperature', 'Wind Speed'),\n",
                "    vertical_spacing=0.1\n",
                ")\n",
                "\n",
                "# GHI\n",
                "fig.add_trace(\n",
                "    go.Scatter(x=daily_df.index, y=daily_df['GHI'], mode='lines', name='GHI'),\n",
                "    row=1, col=1\n",
                ")\n",
                "\n",
                "# Temperature\n",
                "fig.add_trace(\n",
                "    go.Scatter(x=daily_df.index, y=daily_df['Tamb'], mode='lines', name='Ambient Temp'),\n",
                "    row=2, col=1\n",
                ")\n",
                "\n",
                "# Wind Speed\n",
                "fig.add_trace(\n",
                "    go.Scatter(x=daily_df.index, y=daily_df['WS'], mode='lines', name='Wind Speed'),\n",
                "    row=3, col=1\n",
                ")\n",
                "\n",
                "fig.update_layout(\n",
                "    height=800,\n",
                "    showlegend=True,\n",
                "    title_text='Time Series Analysis of Key Metrics'\n",
                ")\n",
                "\n",
                "fig.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Heart Disease Analysis Project

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Clustering Techniques](#clustering-techniques)
- [Visualization](#visualization)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Project Overview
This project aims to analyze a dataset of medical records to identify patients at high risk of developing heart disease. By employing various unsupervised learning techniques, the analysis seeks to uncover patterns and risk factors associated with heart disease.

## Dataset
The dataset used in this project is the heart disease dataset from the UCI Machine Learning Repository. It contains 303 instances with 14 attributes, including:
- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Serum Cholesterol
- Target Variable (presence or absence of heart disease)

## Installation
To run this project, you'll need to have Python and the required libraries installed. Clone the repository using the following command:
```bash
git clone <repository-url>
```

### Requirements
Make sure you have the following libraries:
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Usage
1. Open the Jupyter Notebook included in this repository.
2. Run each cell sequentially to execute the analysis step-by-step.
3. Visualize the results and draw conclusions based on the output.

## Key Features
- **Data Loading and Preprocessing**: Load the heart disease dataset, handle missing values, scale features, and encode categorical variables.
- **Exploratory Data Analysis (EDA)**: Gain insights into the dataset through descriptive statistics and visualizations.
- **Clustering Techniques**:
  - **K-means Clustering**: Group patients based on their medical history, optimizing for the best number of clusters.
  - **Hierarchical Clustering**: Visualize the hierarchical relationships between patients using dendrograms.
  - **DBSCAN Clustering**: Identify dense regions of patients based on their features.
- **Visualization**: Use PCA and t-SNE to visualize the clusters and understand variable relationships.
- **Risk Factor Identification**: Apply Gaussian Mixture Models (GMM) to identify key risk factors for heart disease.
- **Clustering Evaluation**: Assess the performance of clustering algorithms using metrics such as silhouette score and Davies-Bouldin index.
- **Comparative Analysis**: Determine which clustering algorithm yields the best results and the reasons behind it.

## Clustering Techniques
1. **K-means Clustering**: Effective for partitioning patients into distinct groups based on their characteristics.
2. **Hierarchical Clustering**: Provides a dendrogram to visualize the nested grouping of patients.
3. **DBSCAN Clustering**: Useful for identifying clusters of varying shapes and densities.

## Visualization
The project employs PCA and t-SNE to reduce dimensionality and visualize the clusters, helping to understand the relationships between different variables.

## Evaluation Metrics
Clustering performance is evaluated using:
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**: Evaluates the average similarity ratio of each cluster with the one that is most similar to it.


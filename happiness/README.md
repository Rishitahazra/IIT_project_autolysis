# Happiness: An Analysis of Life Satisfaction Factors

## Data Set Overview

In our exploration of happiness, we are presented with a dataset that encompasses 2,363 features across 11 distinct data points. The richness of this dataset allows for a comprehensive analysis of various factors influencing happiness and life satisfaction globally.

### Data Structure

- **Number of Data Points:** 11
- **Number of Features (Columns):** 2,363

### Unique Values in Columns and Their Percentage

The dataset consists of various columns that shed light on life satisfaction. Notably, we have 165 unique countries, 19 years, and a staggering 1,814 unique values for the Life Ladder, which indicates levels of life satisfaction experienced across nations.

| Column                           | Unique Count | Percentage of Unique Values (%) |
|----------------------------------|--------------|---------------------------------|
| Country name                     | 165          | 0.07                            |
| Year                             | 19           | 0.01                            |
| Life Ladder                      | 1,814        | 0.77                            |
| Log GDP per capita               | 1,760        | 0.75                            |
| Social support                   | 484          | 0.20                            |
| Healthy life expectancy at birth  | 1,126        | 0.48                            |
| Freedom to make life choices     | 550          | 0.23                            |
| Generosity                       | 650          | 0.28                            |
| Perceptions of corruption         | 613          | 0.26                            |
| Positive affect                  | 442          | 0.19                            |
| Negative affect                  | 394          | 0.17                            |

### Null Values

This dataset is notably robust, with an overall null count of zero, showcasing the reliability of the data collected:

| Column                           | Null Count | Percentage (%)            |
|----------------------------------|------------|---------------------------|
| Country name                     | 0          | 0                         |
| Year                             | 0          | 0                         |
| Life Ladder                      | 0          | 0                         |
| Log GDP per capita               | 28         | 2.36                      |
| Social support                   | 13         | 1.10                      |
| Healthy life expectancy at birth  | 63         | 5.34                      |
| Freedom to make life choices     | 36         | 3.05                      |
| Generosity                       | 81         | 6.84                      |
| Perceptions of corruption         | 125        | 10.59                     |
| Positive affect                  | 24         | 2.03                      |
| Negative affect                  | 16         | 1.36                      |

### Zero Values Overall and Column Wise

The analysis indicates a total of 10 zero entries, which are exclusively found in the "Generosity" column, suggesting areas where generosity is perceived as absent or unmeasured.

| Column                           | Zero Count |
|----------------------------------|------------|
| Generosity                       | 10         |

### Interesting Findings

1. **Generosity** is the singular column recorded with zero values, underscoring a potential issue in measuring altruistic behaviors across various nations.
2. The dataset is fairly representative of 165 unique countries; however, the variation in continuous measures leads to notable diversity about socio-economic factors affecting happiness.

## Data Categories

In our dataset, we identify the following data types:

```json
{
  "Id_columns": [],
  "Numerical_columns": [
    "year",
    "Life Ladder",
    "Log GDP per capita",
    "Social support",
    "Healthy life expectancy at birth",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
    "Positive affect",
    "Negative affect"
  ],
  "Categorical_columns": [
    "Country name"
  ],
  "Text_columns": [],
  "Others": []
}
```

## Missing Observation

The dataset does not contain any missing values. This robustness highlights the reliability of our dataset for generating insights:

- **Total number of columns containing missing values:** 0
- **Name of the above Columns:** None
- **Top 3 columns with highest percentage of missing values:** None

## Categorical Data Distribution

Analyzing the frequency distribution of the "Country name" column showcases that each of the 30 unique countries is represented exactly 18 times, resulting in a total of 540 records attributed to this categorical column.

1. **Uniform Distribution**: This uniformity might suggest a systematic issue with the data collection process, as real-world datasets often exhibit more variability.
2. **Anomalies**: The perfect balance raises concerns about data integrity, possibly stemming from data entry errors or flawed sampling methodologies.

## Outlier Report

### Overview of Outlier Data

In examining the outlier presence across various columns, we derive the following insights:

| Column                               | Outlier Counts |
|--------------------------------------|----------------|
| Year                                 | 0              |
| Life Ladder                          | 2              |
| Log GDP per capita                   | 1              |
| Social Support                       | 48             |
| Healthy Life Expectancy at Birth      | 20             |
| Freedom to Make Life Choices         | 16             |
| Generosity                           | 39             |
| Perceptions of Corruption            | 194            |
| Positive Affect                      | 9              |
| Negative Affect                      | 31             |

### Analysis of Outliers

The analysis uncovers high variability, particularly in the "Perceptions of Corruption", which has an alarming 194 outliers, suggesting significant discrepancies that merit further investigation.

## Correlation Summary

### Analytical Report on Correlation Matrix

The relationships between the different variables show intriguing patterns:

- **Strongest Positive Correlations**:
  - **Log GDP per capita and Life Ladder (0.78)** – Linking financial conditions and life satisfaction.
  - **Social support and Life Ladder (0.72)** – Indicative of the value of social networks.
  
- **Strongest Negative Correlations**:
  - **Perceptions of corruption and Life Ladder (-0.43)** – Reflects how corruption perceptions substantially decrease life satisfaction.

### Conclusion
This analysis illustrates critical relationships dictating life satisfaction, positioning economic and social factors at the forefront of improving overall happiness.

## Regression Report

The Ordinary Least Squares (OLS) regression analysis provides insight into the predictability of the year based on various factors:

- **Significant Predictors**: 
  - **Freedom to make life choices** shows a substantial positive impact, while **Perceptions of corruption** and **Generosity** depict negative influences on year.
  - The overall R-squared of 0.145 indicates a moderate fit of the model.

## Data Characteristics

The relationships between paired variables further emphasize the importance of various social, economic, and psychological influences on happiness, paving the pathway for future research and policy implications.

## Topics in Data

The dataset comprehensively addresses several topics tied to happiness, touching on themes of economy, social frameworks, and personal well-being through numerous variables.

### Conclusion

The analysis presented uncovers significant patterns regarding happiness and wellbeing indicators across various dimensions. The pronounced presence of outliers, especially in the perceptions of corruption, requires immediate attention and deeper examinations. This dataset serves as the foundation for future studies exploring the interplay between socio-economic factors and life satisfaction on a global scale, guiding policymakers to focus on enhancing social support networks, transparency, and the impacts of generosity on community wellbeing.
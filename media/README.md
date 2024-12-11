# Use Media

## Data Statistics

### Overview of the Dataset

The dataset consists of a total of **2652 data points**, with **2652 features** captured from various entries. Each entry tells a unique story and conveys different aspects related to media content.

#### a) Data Structure

- **Number of Data Points**: 2652
- **Number of Features**: 2652

#### b) Unique Values in Columns and Their Percentage

The dataset exhibits a diverse range of uniqueness across various columns:

| Column Name  | Unique Count | % of Total Data Points |
|--------------|--------------|------------------------|
| date         | 2055         | 25687.5%               |
| language     | 11           | 137.5%                 |
| type         | 8            | 100%                   |
| title        | 2312         | 28900%                 |
| by           | 1528         | 19100%                 |
| overall      | 5            | 62.5%                  |
| quality      | 5            | 62.5%                  |
| repeatability | 3           | 37.5%                  |

The `title` column stands out with **2312 unique entries**, showcasing a rich repository of narratives, whereas the `date` column reflects a high degree of variability.

#### c) Null Values Overall and Column-Wise

In terms of data integrity:

- **Overall Null Values**: 262
- **Percentage of Null Values**: 3.30%

| Column Name  | Null Count | % of Total Data Points |
|--------------|------------|------------------------|
| date         | 0          | 0%                     |
| language     | 0          | 0%                     |
| type         | 0          | 0%                     |
| title        | 0          | 0%                     |
| by           | 262        | 3.30%                  |
| overall      | 0          | 0%                     |
| quality      | 0          | 0%                     |
| repeatability | 0         | 0%                     |

Most columns are devoid of null values, except for the `by` column, which prompts considerations for data imputation or enhancement.

#### d) Zero Values Overall and Column-Wise

The dataset is free from zero values, ensuring that each entry contributes meaningfully to the analysis.

- **Overall Zero Values**: 0
- **Percentage of Zero Values**: 0%

| Column Name   | Zero Count | % of Total Data Points |
|---------------|------------|------------------------|
| date          | 0          | 0%                     |
| language      | 0          | 0%                     |
| type          | 0          | 0%                     |
| title         | 0          | 0%                     |
| by            | 0          | 0%                     |
| overall       | 0          | 0%                     |
| quality       | 0          | 0%                     |
| repeatability  | 0         | 0%                     |

#### e) Other Interesting Findings

Within the dataset, Tamil emerges as a notable language, revealing its prevalence alongside a substantial variety of titles — as evidenced by the unique count in the `title` column. 

The `by` column, however, prompts scrutiny with a noteworthy 262 null values, pointing towards potential avenues for data improvement or enhancement strategies.

---

## Data Categories

The data can be classified into various categories:

```json
{
  "Id_columns": [],
  "Numerical_columns": [
    "overall",
    "quality",
    "repeatability"
  ],
  "Categorical_columns": [
    "date",
    "language",
    "type"
  ],
  "Text_columns": [
    "title",
    "by"
  ],
  "Others": []
}
```

This classification helps to understand the representation of various variables within the dataset, which is crucial for further analyses.

---

## Missing Observations

To analyze the missing values within the dataset:

### Given Data:
- Total Rows: 2652

#### Analysis

- **Total number of columns containing missing values**: **1**
- **Name of the above column**: **'date'**
- **Top column with highest percentage of missing values**: 
  - Column: 'date', Missing value %: 3.73%

The `date` column has a slight deficiency, indicating an aspect ripe for further enhancement or systematic correction.

---

## Categorical Data Distribution

### 1. Date Column Analysis:
- **Diversity of Dates**: The dates show concentrations, particularly around "21-May-06", emphasizing a potential historical context within media entries. 
- **Distribution Pattern**: A significant drop-off in entries post-2019 suggests potential limitations in more recent data capture, crucial for longitudinal studies.

### 2. Language Column Analysis:
- **Prominent Language**: English dominates, comprising nearly half of all entries. This imbalance calls for evaluation of cultural and international representation in the dataset.

### 3. Type Column Analysis:
- **Dominance of One Type**: With movies representing a substantial majority (2211 out of 2652 entries), this highlights a significant focus within the dataset, potentially skewing analyses designed to cover other media types.

### Anomalies and Special Observations
The data exhibits temporal biases, language concentration, and categorical dominance, suggesting a need for careful consideration when drawing conclusions in analyses.

---

## Outlier Report

### Total Row Count:
- **Total Rows**: 2652

### Outlier Analysis:
1. **Overall**: 1216 outliers, representing 45.9% of values—an alarming figure pointing to potential data reliability issues.
2. **Quality**: A mere 0.9% outliers suggests a high degree of consistency among quality ratings.
3. **Repeatability**: Zero outliers indicate dependable repeatability measures.

### Special Observations:
The high outlier count in the overall column necessitates urgent investigation to decipher underlying issues affecting data quality. 

---

## Correlation Summary

### Overview
The correlation matrix reflects interesting relationships among numerical columns:

| Variable        | Overall      | Quality      | Repeatability |
|------------------|--------------|--------------|---------------|
| Overall          | 1.0          | 0.8259       | 0.5126        |
| Quality          | 0.8259       | 1.0          | 0.3121        |
| Repeatability     | 0.5126       | 0.3121       | 1.0           |

### Findings
- **High Correlation Between Overall and Quality**: A strong positive correlation suggests a direct relationship worth exploring in detail.
- **Moderate Correlation with Repeatability**: Points to potential for further understanding regarding the relationship dynamics between quality and repeatability.

### Recommendations
To bolster findings, enhancing clarity in data representation and expanding further statistical explorations could prove beneficial.

---

## Regression Report

The OLS Regression results indicate an unsatisfactory regression model, showcasing a lack of significant predictors impacting the overall ratings, demanding a reevaluation of model inputs and assumptions for more meaningful insights.

---

## Data Characteristics

### Analysis for: Scatter plot: overall and repeatability for language
The scatterplot reveals significant correlations between various metrics, hinting at the interdependencies present in the dataset. Specifically, improving overall ratings may have a cascading effect on repeatability.

---

## Topics in Data

When delving into the thematic undercurrents of the text data:

1. **Title**:
   - Narrative themes of struggle and survival, with character names hinting at a potential dramatic storyline.

2. **By**:
   - Represents a collaborative effort by various contributors, adding depth and diversity to the narrative landscape.

### Overall Summary:
The topics conveyed suggest a rich tapestry of collaboration within a compelling narrative, potentially reflective of broader media trends.

---

## Conclusion

The overall analysis reveals a diverse and complex dataset with notable anomalies, particularly in historical representation and language distribution. Continued efforts towards data quality enhancement, particularly in the `by` column and the existence of outliers in critical variables, may lead to more comprehensive insights. Future investigations should seek to expand content across various dimensions, ensuring broader applicability of findings across different media contexts.
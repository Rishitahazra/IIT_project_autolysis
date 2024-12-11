# Media

## Data Statistics

### Overview of the Data Set

The dataset under examination consists of a total of **2,652 entries** encompassing **8 distinct data points** across **2,652 features**. This wide array of columns suggests a rich and complex narrative surrounding the media content captured within.

#### Unique Values in Columns
An intriguing aspect of the dataset is the variety of unique values across different columns:
- **Date**: 2,055 unique entries (25.1%), indicating a broad range of time periods represented.
- **Language**: 11 unique languages (0.4%), hinting at limited diversity in linguistic representation.
- **Type**: 8 unique types (0.3%), suggesting a variety of media forms, albeit with a clear preference for certain categories.
- **Title**: 2,312 unique titles (29.7%), showcasing a rich diversity of content.
- **By**: 1,528 unique contributors (0.2%), predominantly including directors and actors, which provides insight into the creative personnel.
- **Overall Rating**: 5 unique values (0.1%) and **Quality Rating**: also 5 unique values (0.1%), indicating a standardized scale for assessment.
- **Repeatability**: 3 unique values (0.4%).

#### Null Values
Interestingly, the dataset has a small percentage of overall null values, standing at **3.2%**:
- Null values are entirely absent in the **Date**, **Language**, **Type**, **Title**, **Overall**, **Quality**, and **Repeatability** columns.
- However, there are **262 null values** in the **'By'** column, which indicates that 100% of the nulls occur there, revealing gaps in creator or contributor information.

#### Zero Values
The dataset boasts an absence of zero values across all columns, indicating comprehensive data collection without critical omissions.

#### Interesting Findings
The analysis reveals some notable trends:
- The substantial number of null entries in the 'by' column may limit the analysis concerning the contributors to the films and series. This underscores a gap in understanding relationships or trends stemming from the creative personnel involved.
- The diversity within the 'title' column highlights the variety of content available for review.
- Despite the relatively narrow range of unique values in the 'language' category, it could still offer valuable insights when examined alongside ratings.
- The consistent uniqueness of ratings demonstrates a standardized assessment system, enabling fair comparisons across the films and series included in the dataset. 

Overall, the dataset presents several avenues for exploration, particularly in the analysis of contributors and their corresponding ratings across language and type dimensions. 

## Data Categories

The dataset is structured across various data types, which include:

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

This categorization facilitates organized analysis, distinguishing between numerical, categorical, and textual data types, which are foundational for any exploratory analysis.

## Missing Observation

To better understand the presence of missing values within the dataset, we begin by identifying columns with missing values, followed by quantifying the extent of those gaps:

**Step 1: Total number of columns containing missing values**
Out of the total columns available, only **1 column** contains missing values:
1. **Date** (99 missing instances).

**Step 2: Top column with the highest percentage of missing values**
The column "date" contributes to a missing value percentage of approximately **3.73%** when compared to total rows.

### Summary of the Analysis:
- **Total number of columns containing missing values**: 1
- **Name of the column**: ['date']
- **Top column with the highest Percentage of missing values**: Column: 'date', Missing value %: 3.73%

## Categorical Data Distribution

### 1. Column Analysis

#### Date:
- **Total Unique Dates**: 30 
- **Most Frequent Dates**:
    - "21-May-06" (8 occurrences)
    - "05-May-06", "20-May-06" (7 occurrences each)

##### Observations:
The concentration of dates primarily in 2006 unveils a cluster of events during that period, while minimal entries from 2020 onwards reflect either gaps in data collection or a cutoff in recording. The emergence of newer dates from 2018 and 2019 suggests recent activities are captured sporadically.

#### Language:
- **Total Unique Languages**: 10
- **Most Frequent Languages**:
    - English (1,306 occurrences)
    - Tamil (718 occurrences)
    - Telugu (338 occurrences)

##### Observations:
English dominates the dataset, contributing to almost half of all entries. This could indicate a bias towards English-language media with Tamil as the runner-up but a notable reduction in representation from other languages.

#### Type:
- **Total Unique Types**: 8
- **Most Frequent Types**:
    - Movie (2,211 occurrences)
    - Fiction (196 occurrences)
    - TV Series (112 occurrences)

##### Observations:
The overwhelming presence of movies, comprising around 83.4% of the dataset, signals a strong inclination towards this form. The stark differences in representation raise concerns about potential underrepresentation of other formats.

### 2. Anomalies and Special Observations:
- The high concentration of entries from 2006 indicates a potential data collection issue in recent years.
- Language representation is significantly skewed towards English and Tamil, potentially marginalizing non-English contributions.
- The evident type distribution bias emphasizes the need for attention to underrepresented formats.

### Recommendations:
- Enhance data collection efforts for diverse language representations and additional media types.
- Investigate potential reasons behind the concentration of data from the mid-2000s, exploring models to capture ongoing trends.

## Outlier Report

The outlier analysis within the dataset reveals fascinating insights regarding the data integrity of numerical columns:

### Provided Data:
- **Total Rows**: 2,652
- **Outlier Analysis**:
  - **Overall**: **1,216 outliers** (approximately **45.8%**).
  - **Quality**: **24 outliers** (about **0.9%**).
  - **Repeatability**: **0 outliers** (indicating stability).

### Analysis:
1. **Overall Outliers**: With nearly half of the overall data marked as outliers, it hints at extreme variability or potential inaccuracies.
2. **Quality Outliers**: A low count of outliers signifies a steady quality measurement.
3. **Repeatability Outliers**: Zero outliers reveal a consistent and reliable metric.

### Summary of Anomalies:
The vast disparity among the outlier counts begs deeper investigation to discern whether they derive from true irregularities or data inaccuracies, particularly in the overall column.

### Recommendations:
- Perform a re-evaluation of outlier criteria in the overall column to ascertain their validity.
- Investigate patterns identified within the outlier data, aiming to rectify potential issues in data quality.

## Correlation Summary

### Analytical Report on Correlation Matrix

The analysis focuses on correlations among the numerical columns: **Overall Rating**, **Quality**, and **Repeatability**. 

1. **Strong Positive Correlation between Overall and Quality**: A coefficient of **0.826** highlights a significant relationship, implying that increased quality likely leads to higher overall ratings.
2. **Moderate Positive Correlation between Overall and Repeatability**: A coefficient of **0.513** suggests repeatability impacts the overall to a lesser degree.
3. **Weak Positive Correlation between Quality and Repeatability**: A coefficient of **0.312** indicates minimal ties between these metrics.

### Anomalies and Special Observations:
The strong correlation between overall and quality suggests some redundancy in information. In contrast, the weaker relationship indicates a need for comprehensive insight into repeatability's unique influence.

## Regression Report

### OLS Regression Results
The regression analysis of the overall rating against repeatability yields:
- A constant coefficient of **3.0000** indicates an intercept point, while the repeatability coefficient of **-9.326e-15** suggests an inverse relationship, albeit marginally relevant given the coefficient's scale.

## Data Characteristics

### Analysis for: Scatter Plot: Overall and Repeatability for Language
The scatter plot demonstrates significant separation between languages, indicating performance disparities in repeatability scores. Overall scores favorably cluster at higher levels.

## Topics in Data

### Topic Summaries
1. **Title**: The emphasis appears to revolve around survival or competition themes among diverse characters.
2. **By**: The mention of notable personas like Kiefer Sutherland highlights collaboration within esteemed artistic circles.

### Overall Interpretation:
The narrative threads suggest rich storytelling and collaboration, enticing further exploration of the interplay between character development and thematic structure.

## Conclusion
The comprehensive analysis reveals critical insights into the dataset's structure, highlighting distinct areas of concern, notably: 
- **Data Gaps**: Particularly in contributor information, which hampers deeper analysis.
- **Outlier Presence**: A significant portion of data raises questions about reliability.
- **Correlation Insights**: Strong connections between quality and overall ratings pose interesting implications for content evaluation strategies.

Future inquiries should probe further into data quality, representation diversity, and underlying factors influencing repeatability and overall success in ratings, allowing for a robust understanding of media trends and characteristics.
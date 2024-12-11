# Goodreads

## Data Analysis Story

### Overview of the Dataset
In the vast world of literature, this dataset serves as a digital library containing insightful information about various books. It comprises **10,000 distinct titles** delineated by an extensive array of **10,000 features**, taking our exploration beyond just numbers to a richer narrative of authorial diversity and literary expression.

#### a) Data Structure Overview
This dataset encompasses a total of **23 data points (rows)**, each representing a unique book, contributing significantly to our understanding of literary trends and reader engagement.

#### b) Unique Values in Columns and Their Percentages
The dataset exhibits a fascinating variety of unique values across its features:
- **book_id**: 10,000 unique values (100%)
- **goodreads_book_id**: 10,000 unique values (100%)
- **best_book_id**: 10,000 unique values (100%)
- **work_id**: 10,000 unique values (100%)
- **authors**: 4,664 unique values (~46.64%)
- **original_publication_year**: 293 unique values (~2.93%)

This diverse array reflects a vast selection of authors and their works, suggesting a rich tapestry of literary contributions within the dataset.

#### c) Null Values Overview
While the dataset is comprehensive, it contains **2,022 overall null values**. Notably, the **language_code** feature accounts for the highest number of missing entries with **1,084 nulls**, highlighting an opportunity for improvement in capturing linguistic diversity.

#### d) Zero Values Overview
Interestingly, this dataset reports **no zero values**, indicating that every entry holds relevant data across all features. The absence of zero entries reinforces the completeness and reliability of the information contained within.

#### e) Other Interesting Findings
The dataset is not just vast but also rich in qualitative aspects:
- The **average_rating** shows a high degree of variation, implying diverse reader opinions.
- Despite extensive unique values for **isbn** and **isbn13**, many entries are missing. This suggests a significant number of books may not have been indexed correctly, leaving gaps in bibliographic details.
- Peaks in the **ratings_count** hint at highly popular titles, while the majority of ratings skew favorably towards the higher end of the scale.

### Data Categories
The diverse nature of data can be categorized into different groups:

```json
{
  "Id_columns": [
    "book_id",
    "goodreads_book_id",
    "best_book_id",
    "work_id",
    "isbn",
    "isbn13"
  ],
  "Numerical_columns": [
    "books_count",
    "original_publication_year",
    "average_rating",
    "ratings_count",
    "work_ratings_count",
    "work_text_reviews_count",
    "ratings_1",
    "ratings_2",
    "ratings_3",
    "ratings_4",
    "ratings_5"
  ],
  "Categorical_columns": [
    "authors",
    "language_code"
  ],
  "Text_columns": [
    "original_title",
    "title"
  ],
  "Others": [
    "image_url",
    "small_image_url"
  ]
}
```

### Missing Observation
Among the numerous features, only the **language_code** column harbors missing values. It stands out as a focal point for further analysis, particularly because it represents **10.84%** of its total data. 

### Categorical Data Distribution
The dataset provides fascinating insights into author representation and language distribution:

#### Frequency Distribution of Authors
- Dominant names like **Stephen King** (60 works) and **Nora Roberts** (59 works) showcase a concentration of well-known writers, while the diversity narrows down, revealing only **34 authors** recorded within **10,000 entries**. 

#### Frequency Distribution of Language Codes
- A striking preference for English is evident with **6341 occurrences** of the **'eng'** code, suggesting potential limitations on international representation within the dataset.

### Outlier Report
This dataset has been thoroughly examined for outliers, especially in the following key columns:
- Finding substantial outliers, particularly in **ratings_count** and **work_ratings_count** (over **1,140** each), suggests an uneven distribution that may reflect inconsistencies in user engagement or publication circumstances.

### Correlation Summary
A correlation matrix reveals intriguing relationships among numerical attributes:
1. **`ratings_count` and `work_ratings_count`** showcase a perfect correlation (0.995), suggesting that robust ratings attract similar counts of user engagement.
2. **`books_count` shows a slight negative correlation with `original_publication_year` (-0.32)**, indicating newer books are more prolific, likely due to evolving reader preferences and publishing trends.

### Regression Report
An Ordinary Least Squares (OLS) regression identified a significant relationship between variables:
1. **books_count** indicates a negative impact on **original_publication_year** illustrating how the number of publications is rising while the publication years are shifting positively towards more recent works.

### Data Characteristics
Scatterplot analyses offer a glimpse into critical relationships:
- A clear positive correlation between **average ratings** and **ratings counts** further underscores how reader engagement translates into perceived book quality.

### Topics in Data
Engaging with the vast textual data within the titles reveals interwoven themes:
1. Recurrent topics of **life, love, and death**, enriched with narratives that delve deeply into human experience, suggest that storytelling remains central to the dataset's corpus.

### Conclusion and Special Observations
The analysis reveals rich insights into the dataset's structure and its literary content. While the dominance of specific authors and the English language tells a compelling story of market trends, the missed opportunities for linguistic and authorial diversity highlight areas for future improvement. By broadening the dataset’s representations, we can enhance the narratives, allowing for a more expansive literary exploration that captures the plurality of human storytelling across global languages and cultures.

## ::--------Strat of code description---------------------:: ##

# Author: Rishita Hazra
# Created Date: 12/11/2024
# Modified Date: 12/11/2024
# Version: 1.0
# Description: This code analyse a csv file containing a data set and provides useful information in a readme file.
#              This code also creates a directory with the filename and stores the images as .png file and the readme
#              file. 
# Tools: Python and GPT gpt-4o-mini
# Prerequisites: Store GPT API call key in environment as variable AIPROXY_TOKEN

## ::-------End of code description---------------------:: ##

## following section install all the dependencies 

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nltk>=3.9.1",
#     "pandas>=2.2.3",
#     "scikit-learn>=1.6.0",
#     "seaborn>=0.13.2",
#     "statsmodels>=0.14.4",
#     "requests>=2.32.3"
# ]
# ///

## Import libraries that will be used in the code
import pandas as pd
import numpy as np
import requests
import sys
import os
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import json
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import re
import statsmodels.api as sm
import base64


## Retrive the API key, if not raise error
try:
    # Retrieve the API key
    api_key = os.environ.get('AIPROXY_TOKEN')
    if not api_key:
        raise EnvironmentError("AIPROXY TOKEN not found")
except EnvironmentError as e:
    print(e)
    sys.exit(1)


## Definition of functions to be used in the code 

# Call text generation API of GPT model and return output as text
def get_txt_api(prompt, model = 'gpt-4o-mini'):
    # Define the endpoint URL and headers
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        'Content-Type': "application/json"
    }

    json_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=json_data)
    json_output = json.loads(response.text)
    return json_output['choices'][0]['message']['content']

# Function to generate frequency tables for all columns
def frequency_tables(df):
    freq_tables = {}
    for column in df.columns:
        freq_tables[column] = df[column].value_counts().head(30).to_dict()
    return freq_tables


# Function for Outlier Analysis
def outlier_analysis(dframe, columns):
    """
    This is to analyse outliers
    Arguments: df-Data Source; columns - input columns
    """
    df = dframe.copy(deep=True)
    # Using IQR
    list_boxplot = []
    for i in columns:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        UWH = Q3 + 1.5*IQR
        LWH = Q1 - 1.5*IQR

        outlier_count =  df[i][(df[i] < LWH) | (df[i] > UWH) ].count()
        list_boxplot.append({"Column_name": i, "Outlier_count": outlier_count})
         
    boxplot_df =  pd.DataFrame(list_boxplot)            
    return boxplot_df

# Function to get VIF values
def get_vif(df):
    """"
    df: Dataframe with independent continuous features
    """
    vif_columns = list(df.columns)
    variables = list(range(df.shape[1]))
    vif = [variance_inflation_factor(df.iloc[:, variables].values, ix)
                   for ix in range(df.iloc[:, variables].shape[1])]
    
    vif_df = pd.DataFrame.from_dict({'Features': vif_columns, 'VIF': vif})
    return vif_df

# Function for Outlier treatment
def outlier_treatment(dframe):
    """
    This is to treat outliers
    Arguments: dframe - pandas dataframe object
    Output: df_treated - pandas dataframe object
    """
    import numpy as np
    df_treated = dframe.copy(deep=True)
    # Using IQR
    # list_boxplot = []
    for i in df_treated.columns:
        Q1 = df_treated[i].quantile(0.25)
        Q3 = df_treated[i].quantile(0.75)
        IQR = Q3 - Q1
        UWH = Q3 + 1.5*IQR
        LWH = Q1 - 1.5*IQR

        nc = i +'treated' # New column name
        df_treated[i] = np.where( (df_treated[i] > UWH), UWH, np.where( (df_treated[i] < LWH), LWH, df_treated[i])) 
           
    return df_treated

## Function to call image analysis API of GPT model and return output as text
def analyse_img_api(prompt, list_base64_image, model = 'gpt-4o-mini'):
    # Define the endpoint URL and headers
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        'Content-Type': "application/json"
    }

    json_data = {
    "model": 'gpt-4o-mini',
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text":prompt},
                        {"type": "image_url", "image_url": { "url": f"data:image/png;base64,{list_base64_image[0]}"}}
                        ]  
        },
       
                ]
    
            }

    response = requests.post(url, headers=headers, json=json_data)
    json_output = json.loads(response.text)
    return json_output['choices'][0]['message']['content']

# Function to preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

## Initialise the variables which will store text values
text_data_stat = ""
text_feature_understand = ""
text_missing_value = ""
text_frequency_distribution = ""
text_outlier_report = ""
text_correlation_report = ""
text_regression_summary = ""
text_image_analysis = ""
text_topic_summary = ""

## Read the data-set from command line for analysis
if len(sys.argv) == 2:
    dataset_path = sys.argv[1]
    # read data
    df_data = pd.read_csv(dataset_path, encoding='latin1')
else:    
    print("No data-set provided")
    sys.exit(1)



## Create a directory in the curent working directory to store readme and png files
# Get the directory name from data-set name
if '/' in dataset_path:
    str_csv = dataset_path.split("/")
    list_dir_name = str_csv[-1].split(".")
    dir_name = list_dir_name[0]
else:
    list_dir_name = dataset_path.split(".")
    dir_name = list_dir_name[0]  

# create directory
os.mkdir(dir_name)


## This section's code creates a data statistics report and store in a variable as text

Number_of_columns = df_data.shape[0]
Number_of_rows = df_data.shape[1]

## Get unique and null count
df_unique = pd.DataFrame({'columns': list(df_data.nunique().index), 'unique_count':list(df_data.nunique().values), 'null_count': list(df_data.isna().sum().values)})
json_unique = df_unique.to_json(orient = 'records', lines = True)

## Get zero count
# Count the number of zeros in each column
zero_counts = (df_data == 0).sum()
# convert to json
json_zero = zero_counts.to_json()

# Sample data
json_data = df_data.head(20).to_json(orient = 'records', lines = True)

prompt_data_stat = f""" You are an AI tool designed to give overview of a data set. Provide the overview in a structured way 
covering following points - 

a) A brief overview of the data structure including number of data pointa and number of features

b) About unique values in the columns and their %  

c) About null values overall and column wise and their %

d) About zero values overall and column wise and their %

e) Any other interesting finding from the sample data provided 


Sample JSON data to use for reference only is: {json_data}. Note: this is not full data dont consider it for any calculation.
Number of rows in full data set: {Number_of_rows}
Number of column in full data set:{Number_of_columns}
Also use Json {json_unique} which contains column names, unique count and null count
Also use Json {json_zero} column wise zero count"""

# Call GPT API to get the output
try:
    text_data_stat = get_txt_api(prompt_data_stat, model = 'gpt-4o-mini')
except: 
    text_data_stat = ""


## This section's code categories the data into several categories

prompt_feature_understand = f""" You are an AI tool designed to identify and categorize important data types from a given dataset. 
Based on the column names and the first few rows of data in JSON format, your task is to classify the columns into the following categories:

a) Id_columns: These columns are used as unique identifiers for the rows, often nominal in nature. The values are mostly unique. 
Examples include IDs, phone numbers, ISBNs, PIN codes, etc. 

b) Numerical_columns: These columns contain numerical values used as measures for data analysis, typically continuous in nature. 
Be careful to exclude ID columns from this category.

c) Categorical_columns: Also known as 'Dimensions' in data analytics, these columns can contain both text and numbers but 
are used to categorize data. Again, be careful to exclude ID columns. 
Examples include month, year, day, country, region, 

d) Text_columns: Columns which has long text values of length gretar than > 15 can be consider here. 
Examples include: Title, book name, feedback, summary etc.
Exceptions are:URLs, email addresses, links etc. which are not pure text values.

e) Others: Any columns that do not fit into the above three categories. 
Examples include URLs, email addresses, links, etc.

Based on the provided JSON data, categorize each column accordingly and return the results in JSON format.

Sample JSON data to use for analysis is: {json_data}. Note this is sample data not full data.

Output format: Provide the output only in json format """

# Call the GPT API for categorisation

try:
    text_feature_understand = get_txt_api(prompt_feature_understand, model = 'gpt-4o-mini')
except:
    text_feature_understand = ""



## This section segregate's numbers, dimnesions and text of the data-set into different dataframes
jsont = json.loads(text_feature_understand.strip('```json\n').strip('```'))

df_num = df_data[jsont['Numerical_columns']] 
df_cat = df_data[jsont['Categorical_columns']]
df_text = df_data[jsont['Text_columns']]


## This section's code analyse missing values for categorical and numerical columns and creates missing value report
df_missing_num = pd.DataFrame({'Column': df_num.columns, 'Missing_count': df_num.isna().sum().values})
json_missing_num = df_missing_num.to_json(orient = 'records', lines = True)

df_missing_cat = pd.DataFrame({'Column': df_cat.columns, 'Missing_count': df_cat.isna().sum().values})
json_missing_cat = df_missing_cat.to_json(orient = 'records', lines = True)


prompt_missing_value = f""" You are an AI tool designed to provide an analysis on missing values of a data set. You will be provided the 
total row count and a json file containing column names and missing value count. Analyze the given data and provide the following details:

a) Total number of columns containing missing values:

b) Name of the above Columns:

c) Top 3 columns with highest Percentage of missing values. For this provide column names and percentage in the folowing format:
Example : Column: 'Abc', Missing value %: 20% 

For point c if the number of columns containing missing values are less that 3 lets say 2or 1 then provide only 2 or one columns

Use following data:
total rows: {df_data.shape[0]}
JSON data to use for analysis is: {json_missing_cat}
"""
try:
    text_missing_value = get_txt_api(prompt_missing_value, model = 'gpt-4o-mini')
except:
    text_missing_value = ""
## This section's code generate Frequency distribution report for categorical variable

# Generate frequency tables
if len(df_cat) != 0:
    freq_tables = frequency_tables(df_cat)

    # Convert frequency tables to a JSON string
    json_frequency_dist = json.dumps(freq_tables, indent=4)


    # Frequency distribution report
    prompt_frequency_dist = f""" You are an AI tool designed to provide an analysis on frequency distribution of the columns. You will be provided the 
    total row count and a json file containing column names and their frequency distribution. Analyze the given data and provide the following details:

    Any anomalies found in the column catogories or any special observation

    Use following data:
    total rows: {df_data.shape[0]}
    JSON data to use for analysis is: {json_frequency_dist}
    """

    text_frequency_distribution = get_txt_api(prompt_frequency_dist, model = 'gpt-4o-mini')



## This section generates rreport for Outliers for numerical columns
if len(df_num) != 0:
    numcolmn = list(df_num.columns)

    df_boxplot = outlier_analysis(df_num, numcolmn)
    json_boxplot = df_boxplot.to_json(orient = 'records', lines = True)

    # Prompt for Outlier report
    prompt_outlier_report = f""" You are an AI tool designed to provide an analysis on outliers of columns. You will be provided the 
    total row count and a json file containing column names and their outlier count. Analyze the given data and provide the following details:

    Any anomalies found in the column catogories or any special observation

    Use following data:
    total rows: {df_data.shape[0]}
    JSON data to use for analysis is: {json_boxplot}
    """
    text_outlier_report = get_txt_api(prompt_outlier_report, model = 'gpt-4o-mini')


## This section does Correlation analysis and generates heatmap image
if len(df_num) != 0:
    # Generate correlation matrix
    matrix_correlation = df_num.corr()
    json_matrix_corr = matrix_correlation.to_json(orient = 'records', lines = True)

    # Generate and save heatmap
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 8))

    # Draw the heatmap 
    sns.heatmap(matrix_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=.5)

    # Add meaningful titles and labels and save
    plt.title('Correlation Heatmap')
    plt.savefig(f'./{dir_name}/Correlation_Heatmap.png')
    plt.clf()
    # Correlation report
    prompt_correlation_report = f""" You are an AI tool designed to provide an analysis report on correlations  of the numerical columns. 
    You will be provided a json file containing correlation matrix. Analyze the given data and provide an analytical report with findings.
    Any anomalies found in the column catogories or any special observation

    Use following data:
    JSON data to use for analysis is: {json_matrix_corr}
    """

    text_correlation_report = get_txt_api(prompt_correlation_report, model = 'gpt-4o-mini')

    ## This section's code determines uncorrelated variables from correlation matrix. This list will be used for regression 
    #  analysis in the next section
    list_uncorrelated_columns = list(matrix_correlation.columns)

    for c in range(matrix_correlation.shape[1]):
        for r in range(1+c,matrix_correlation.shape[1]):
            if c == r:
                print("same r c")
                break
            else:
                if matrix_correlation.iloc[r,c] > 0.75:
                    try:
                        list_uncorrelated_columns.remove(matrix_correlation.columns[r])
                    except:
                        continue 


## This section does VIF analysis for regression relationship
if len(df_num) != 0:
    # Missing value imputation before regression
    df_num_impute = df_num[list_uncorrelated_columns].apply(lambda x: x.fillna(x.mean()),axis=0)

    # Outlier treatment before regression
    df_outlier_treated = outlier_treatment(df_num_impute)

    # VIF 
    df_non_nan_num = df_outlier_treated.dropna()    
    df_vif =  get_vif(df_non_nan_num) 

    ## Regression for top VIF column
    df_vif.sort_values(by = 'VIF', ascending=False,inplace=True)

    dep_variable= df_vif.reset_index()['Features'][0]


    # Regression analysis and report
    Y = df_non_nan_num[dep_variable]
    X = df_non_nan_num.drop(dep_variable, axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(Y,X)
    results = model.fit()

    text_regression_summary = results.summary()

## This section ceates category wise scatter plot

# Keep only those categorical columns which has less than 30 unique values, more values will distort the plot
if len(df_cat) != 0:
    df_cat_filtered = df_cat[df_cat.columns[df_cat.nunique() < 30]]
    # Create pairs from the numeric columns
    list_col_pairs = list(itertools.combinations(list_uncorrelated_columns, 2))

    for pairs in list_col_pairs:
        for cat in df_cat_filtered.columns:
            sns.scatterplot(data=df_data, x=pairs[0], y=pairs[1], hue=cat, palette="deep")
        
            plt.title(f'Scatter plot: {pairs[0]} and {pairs[1]} for {cat}')
            plt.savefig(f'./{dir_name}/Scatterplot_{pairs[0]}_and_{pairs[1]}_for_{cat}.png')
            plt.clf()
## This section creates Sctterplot report
    prompt_image_analysis = f""" You are an image analyser expert in analyzing scatterplot images generated by Python Seaborn package. 
    Analyse the 6 images provided and give findings. Make the analysis crisp, avoid providing axis information.

    Segregate each image analysis by the image header.
    Example: 
    Analysis for: Image header name
    Findings: scatterplot shows strong correlation between rating and books published ect."""

    # Get the list of all files in the current directory
    files_in_directory = os.listdir(f'./{dir_name}')

    # Filter out the .png files that start with 'scatterplot'
    scatterplot_png_files = [file for file in files_in_directory if file.endswith('.png') and file.startswith('Scatterplot')]

    if len(scatterplot_png_files) > 10:
        list_scatterplot_png_files = scatterplot_png_files[0:10]
    else:    list_scatterplot_png_files = scatterplot_png_files 

    # Open the image file and encode it as a base64 string
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    list_base64_image = [encode_image(f'./{dir_name}/{image_path}') for image_path in list_scatterplot_png_files]

    try:
        text_image_analysis = analyse_img_api(prompt_image_analysis, list_base64_image)
    except:
        text_image_analysis = ""
## This section creates topics from text fields

if len(df_text) != 0:
    df_text_na_removed =  df_text.dropna()
    list_topic = []
    for col in df_text_na_removed.columns:
        df_text_na_removed['processed_text'] = df_text_na_removed[col].apply(preprocess_text)

        # Convert the text data to a document-term matrix
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(df_text_na_removed['processed_text'])

        # Perform LDA
        lda = LatentDirichletAllocation(n_components=1, random_state=42)
        lda.fit(dtm)

        # Display the top words for each topic
        def display_topics(model, feature_names, no_top_words):
            
            for topic_idx, topic in enumerate(model.components_):
                dict_topic = {col: " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])}
                return dict_topic
            
        no_top_words = 15
        tf_feature_names = vectorizer.get_feature_names_out()
        dict_topic = display_topics(lda, tf_feature_names, no_top_words)
        
        list_topic.append(dict_topic)


    prompt_topic_summary = f""" You are a topic summary provider from topics generated from LDA analysis. 
    You provide the summary in human readable format wich is easily comprehensible.

    Analyse the list of topic created from text from a dataframe columns and provide the summary column wise.

    you are provided with a list {list_topic} wich containf column names as dictionary key and topics as value format"""

    text_topic_summary = get_txt_api(prompt_topic_summary, model = 'gpt-4o-mini')


## This final section creates a readme file and save the analysis
prompt_create_readme = f""" You are a story writer who writes data analysis story about a data set.
You will be given different sections and corresponding text. Your job is to elaborate 
the section from the information available in the given text as human readable story in markdown format.

Note: If the given text is blank in any section, drop that section.

1. Story Title: {dir_name}

2. Data statistics: In this section provide data statistics using {text_data_stat}

3. Data categories: This section talks about different data types. Use {text_feature_understand}

4. Missing observation: This section talks about missing values in data. Use {text_missing_value}

5. Categorical data distribution: This section talks about frequency distribution of categorical columns. Use {text_frequency_distribution}

6. Outlier report: This section talks about outliers of numerical columns. Use {text_outlier_report}

7. Correlation summary: This section talks about correlation in data. Use {text_correlation_report}

8. Regression report: This section talks about regression analysis. Use {text_regression_summary}

9. Data Characteristics: This section talks about relationship between data pairs. Use {text_image_analysis}

10. Topics in data: This section talks about topcs present in text data {text_topic_summary}


Finally provide any special observation from overall analysis as conclusion.

"""

text_create_readme = get_txt_api(prompt_create_readme, model = 'gpt-4o-mini')

# Write the Markdown content to README.md file 
with open(f'./{dir_name}/README.md', 'w') as file:
    file.write(text_create_readme)

##:: ---------------------------$$:: End of code ::$$---------------------------------##

from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

import numpy as np
import pickle
import sqlite3

import base64

import nltk 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# def cosine_similarity(a, b):
#     cos_sim = np.dot(a.T, b)/(np.linalg.norm(a.T)*np.linalg.norm(b.T))
    # return cos_sim
def load_data_to_dataframe(db_file, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)

    # Read data from the table into a DataFrame
    query = 'SELECT * FROM {}'.format(table_name)
    df = pd.read_sql_query(query, conn).iloc[:, 1:]

    # Close the connection
    conn.close()

    return df

def clean_text(text):
    punctuation = '!"$%&\’()*+,-./”/“/:;<=>?[\\]^_`{|}~•@'
    if isinstance(text, float) and np.isnan(text):
        return text
    text  = "".join([char for char in text if char not in punctuation]).lower().split()
    text = " ".join([word.strip() for word in text if word.strip() not in stopwords.words('english')])
    
    text = re.sub("'", "", text) # to avoid removing contractions in english
    text = re.sub(r'http\S+', '', text) # remove htt
    text = re.sub('[()!?]', ' ', text) # This line of code replaces any parentheses, exclamation marks, or question marks in the text with spaces
    text = re.sub('\[.*?\]',' ', text) # 
    text = re.sub("[^a-z0-9]"," ", text) # This line of code removes all non-alphanumeric characters from the text, replacing them with spaces.
    text = re.sub(r'([A-Za-z])\1{2,}', r'\1', text)
    text = re.sub(r'\b\w\b', ' ',text)
    return text

def clean_experience_string(experience_string):
    # Remove extraneous characters
    experience_string = " ".join(experience_string.split(",")).replace("'", "").replace("   ", "").replace("\"", "")
    experience_string = " ".join([word for word in experience_string.split(' ') if word not in stopwords.words('english')])
    # Remove strings containing multiple hyphens
    experience_string = ' '.join([s for s in experience_string.split(' ') if s.count('-') <= 1])
    # Remove non-alphanumeric characters
    regex_pattern = r"[^a-zA-Z0-9\s]+"
    experience_string = re.sub(regex_pattern, "", experience_string)
    return experience_string


def tokenise(linked_df):
    linked_df['Word tokenize']= [word_tokenize(entry) for entry in linked_df.content]
    return linked_df


def clean_more(df_clean):
    df_clean=df_clean.replace(to_replace ="\[.", value = '', regex = True)
    df_clean=df_clean.replace(to_replace ="'", value = '', regex = True)
    df_clean=df_clean.replace(to_replace =" ", value = '', regex = True)
    df_clean=df_clean.replace(to_replace ='\]', value = '', regex = True)

    return df_clean

def extract_salary(string):
    if not isinstance(string, str):
        return None
    res = []
    ranges = string.split(' - ')
    ranges = [val.replace(',','').replace('₹','') for val in ranges]
    regex = re.compile(r'\d+')
    for rng in ranges:
        matches = regex.findall(rng)
        for m in matches:
            val = float(m)            
            if val < 100.0:
                val = val*100000
            res.append(val)
    return np.mean(res)


def find_all():
    # linked_df = pd.read_csv("Employee_data.csv", index_col = 0)
    linked_df = load_data_to_dataframe('Employee_data.db', 'Employee')

    linked_df.columns = ['Current Role', 'About me', 'Education',
       'Years', 'Skills', 'Experience', 'TEXT',
       'Notice Period', 'Expected CTC', 'Offered Location', 'Offered Salary',
       'Current Salary', 'Current Location', 'Name', 'label']
    return linked_df[['Years', 'Current Salary']]


def find_most_similar(job_desc_list = ['Python developer', '5 years of experience in python 2 yesr sin django', 'This is very suitable job']):
    # linked_df = pd.read_csv("Employee_data.csv", index_col = 0)
    linked_df = load_data_to_dataframe('Employee_data.db', 'Employee')

    linked_df.columns = ['Current Role', 'About me', 'Education',
       'Years', 'Skills', 'Experience', 'TEXT',
       'Notice Period', 'Expected CTC', 'Offered Location', 'Offered Salary',
       'Current Salary', 'Current Location', 'Name', 'label']

    linked_df[['Skills', 'Experience','About me']] = linked_df[['Skills', 'Experience','About me']].fillna('Unknown')

    linked_df['About me'] = linked_df['About me'].apply(lambda x: clean_text(x))
    linked_df['Skills'] = linked_df['Skills'].apply(lambda x: clean_experience_string(x))
    linked_df['Experience'] = linked_df['Experience'].apply(lambda x: clean_experience_string(x))


    # print(linked_df.head())
    non_numeric_cols = ['Skills', 'Experience','About me']
    linked_df['TEXT'] = linked_df[non_numeric_cols].apply(lambda x: ' '.join(x), axis=1)

    if job_desc_list != []:
        job_text = ' '.join(job_desc_list)
        job_text = [clean_text(job_text)]

        people_text = linked_df['TEXT'].to_list()

        corpus = job_text + people_text

        vectorizer = TfidfVectorizer(max_features = 1000)
        tfidf_matrix_people = vectorizer.fit_transform(people_text)
        tfidf_matrix_job = vectorizer.transform(job_text)

        cosine_sim = []
        for unit_tfidf_matrix_people in tfidf_matrix_people:
            cosine_sim.append(cosine_similarity(tfidf_matrix_job, unit_tfidf_matrix_people)[0][0])

        cosine_sim = np.array(cosine_sim)
        linked_df['Cosine sim'] = cosine_sim
        cosine_sim_descending = cosine_sim.argsort()[::-1]
        return linked_df.iloc[cosine_sim_descending, :]



# Function to convert image into base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def fig_to_image(fig):
    """
    Convert Matplotlib figure to base64 encoded image
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_data = buf.getvalue()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    return encoded_image


def load_data_for_testing():
    # linked_df = pd.read_csv("Employee_data.csv", index_col = 0)
    linked_df = load_data_to_dataframe('Employee_data.db', 'Employee')

    linked_df.columns = ['Current Role', 'About me', 'Education',
       'Years', 'Skills', 'Experience', 'TEXT',
       'Notice Period', 'Expected CTC', 'Offered Location', 'Offered Salary',
       'Current Salary', 'Current Location', 'Name', 'label']

    linked_df[['Skills', 'Experience','About me']] = linked_df[['Skills', 'Experience','About me']].fillna('Unknown')

    linked_df['About me'] = linked_df['About me'].apply(lambda x: clean_text(x))
    linked_df['Skills'] = linked_df['Skills'].apply(lambda x: clean_experience_string(x))
    linked_df['Experience'] = linked_df['Experience'].apply(lambda x: clean_experience_string(x))
    return linked_df
    
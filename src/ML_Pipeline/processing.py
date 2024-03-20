
import warnings

import pandas as pd
from pandas import factorize
import pickle
from sklearn import preprocessing


warnings.filterwarnings("ignore")



def preprocess_data(df):
    # Preprocessing
    df['Lead created'] = pd.to_datetime(df['Lead created'], format="%d-%m-%Y %H:%M")
    df['Lead Last Update time'] = pd.to_datetime(df['Lead Last Update time'], format="%d-%m-%Y %H:%M")
    df['Next activity'] = pd.to_datetime(df['Next activity'], format="%d-%m-%Y %H:%M")
    df['Demo Date'] = pd.to_datetime(df['Demo Date'], format="%d-%m-%Y %H:%M")
    df = df[~df['Interest Level'].isin(["Not called", "Closed", "Invalid Number"])]
    df['Interest Level'] = df['Interest Level'].apply(lambda x: 1 if x in ["Slightly Interested", "Fairly Interested", "Very Interested"] else 0)

    df = df.drop(["Lead Id", "Lead Location(Auto)", "Next activity", "What are you looking for in Product ?",
                  "Lead Last Update time", "Lead Location(Manual)", "Demo Date", "Demo Status", "Closure date"], axis=1)

    df['hour_of_day'] = df['Lead created'].dt.hour
    df['day_of_week'] = df['Lead created'].dt.weekday

    df = df.drop(["Lead created"], axis=1)

    labels, categories = factorize(df["Creation Source"])

    df["labels"] = labels

    df = df.drop(["labels"], axis=1)

    df['What do you do currently ?'] = df['What do you do currently ?'].apply(lambda x: 1 if 'student' in str(x).strip().lower() else 0)

    df = df.drop(["Website Source"], axis=1)

    df['Marketing Source'].fillna("Unknown", inplace=True)

    label_encoder1 = preprocessing.LabelEncoder()
    df['Marketing Source'] = label_encoder1.fit_transform(df['Marketing Source'])

    label_encoder2 = preprocessing.LabelEncoder()
    df['Lead Owner'] = label_encoder2.fit_transform(df['Lead Owner'])

    label_encoder3 = preprocessing.LabelEncoder()
    df['Creation Source'] = label_encoder3.fit_transform(df['Creation Source'])

    return df
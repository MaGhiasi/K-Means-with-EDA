import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings as wr
from sklearn.preprocessing import StandardScaler


# wr.filterwarnings('ignore')


def read_data(addr):
    # loading and reading dataset
    return pd.read_csv(addr)


def show_data_inf(df):
    print(df.head())

    # shape of the data
    print(df.shape)

    # data information
    df.info()


def describe_features(df):
    # describing the data and print for each column
    num_descript = df.describe()
    # Summary statistics for categorical columns
    cat_descript = df.describe(include=['object'])
    print(cat_descript)
    for i in range(0, 13, 2):
        print(num_descript.iloc[:, i:i+2])


def show_null_unique_vals(df):
    # check for missing values:
    print('\nNull Values\n', df.isnull().sum(), sep='')

    # checking duplicate values
    print('\nUnique Values\n', df.nunique(), sep='')


def drop_null_columns(df, column_names_list):
    # Drop Null columns
    df = df.drop(columns=column_names_list[-4:])

    # Convert categorical data types
    for i in range(0, 2):
        uniques = df[column_names_list[i]].unique()
        map_dict = dict(zip(uniques, range(len(uniques))))
        df = df.replace({column_names_list[i]: map_dict})

    return df


def convert_date(df, column_names_list):
    df[column_names_list[2]] = pd.to_datetime(df[column_names_list[2]])

    # Extract features from the timestamp
    df['hour'] = df[column_names_list[2]].dt.hour
    df['day_of_week'] = df[column_names_list[2]].dt.dayofweek
    df['day_of_month'] = df[column_names_list[2]].dt.day
    df['month'] = df[column_names_list[2]].dt.month
    df['year'] = df[column_names_list[2]].dt.year
    df = df.drop(columns=[column_names_list[2]])
    return df


def normalize_data(df):
    column_names_list = df.columns.tolist()
    features = df[column_names_list]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


def perform_eda(path):
    df = read_data(path)
    # column to list
    column_names = df.columns.tolist()

    show_data_inf(df)
    describe_features(df)
    show_null_unique_vals(df)
    df = drop_null_columns(df, column_names)
    df = convert_date(df, column_names)
    df = normalize_data(df)

    return df


if __name__ == '__main__':
    perform_eda('Live.csv')

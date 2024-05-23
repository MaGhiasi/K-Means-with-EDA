import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import re
import matplotlib.pyplot as plt


def basic_data_inf():
    global data_frame
    print('First 5 rows:\n', data_frame.head())
    column_names = data_frame.columns.tolist()  # column names as list
    print('Data shape: ', data_frame.shape, 'Column names:\n', column_names)

    # Data information
    print('Data information: \n \n')
    data_frame.info()

    print('Data description: \n \n')
    # Describing the categorical data
    cat_descript = data_frame.describe(include=['object'])
    print(cat_descript)

    # Describing the data
    num_descript = data_frame.describe()
    for i in range(0, 13, 2):
        print(num_descript.iloc[:, i:i+2])


def show_null_unique_vals():
    global data_frame
    # Check for missing values
    print('\nNull Values for each column:\n', data_frame.isnull().sum(), sep='')

    # Check for unique values
    print('\nUnique Values for each column:\n', data_frame.nunique(), sep='')


def handle_converting_data():
    global data_frame
    # Drop Empty columns
    data_frame.dropna(axis=1, how='all', inplace=True)

    # Split categorical columns
    non_numeric_cols = list(data_frame.select_dtypes(exclude=[np.number]).columns)

    # Remove date time columns
    for nn_col in non_numeric_cols:
        pattern = r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$'
        date_time_str = data_frame[nn_col][0]     # check a sample of that column
        if re.match(pattern, date_time_str):
            data_frame.drop(columns=[nn_col], inplace=True)
            non_numeric_cols.remove(nn_col)

    # Remove categorical data if it is almost unique for each
    for nn_col in non_numeric_cols:
        unique_vals = list(data_frame[nn_col].unique())
        if len(unique_vals) > 0.9*data_frame.shape[0]:
            non_numeric_cols.remove(nn_col)
            data_frame = data_frame.drop(columns=[nn_col])

    handle_cat_vals(non_numeric_cols)


def handle_cat_vals(categorical_cols):
    for cat in categorical_cols:
        # Label Encoding
        label_encoder = LabelEncoder()
        encode_label = cat + '_encoded'
        data_frame[encode_label] = label_encoder.fit_transform(data_frame[cat])
        data_frame.drop(columns=[cat], inplace=True)


def normalize_data():
    global data_frame
    # Normalize columns based on min and max of each
    column_names_list = data_frame.columns.tolist()
    features = data_frame[column_names_list]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    data_frame = pd.DataFrame(scaled_features, columns=data_frame.columns)


def plot_show():
    global data_frame
    # Distributions for numerical columns
    data_frame.hist(bins=30, figsize=(20, 15))
    plt.show()


def perform_eda(path):
    global data_frame
    data_frame = pd.read_csv(path)
    basic_data_inf()    # Get a basic information
    show_null_unique_vals()

    # Remove null column , handle date time columns and encode categorical columns
    handle_converting_data()

    # Normalize and convert data to numpy array
    normalize_data()
    plot_show()

    return data_frame


if __name__ == '__main__':
    data_frame = pd.DataFrame()

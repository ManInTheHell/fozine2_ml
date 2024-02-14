import pandas as pd


def load_dataset(file_path):
    dataframe = pd.read_csv(file_path, converters={'Fundos (US$)': convert_to_numeric})
    dataframe = dataframe.drop(columns=['sequence', 'university', 'overall_score'])

    data = dataframe.drop(columns=['rank'])
    tag = dataframe['rank']

    return data, tag


def convert_to_numeric(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return None


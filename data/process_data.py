from sqlalchemy import create_engine
import pandas as pd
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the data provided by messages and categories into a pandas Dataframe.

    Parameters:
    messages_filepath (str): filepath for the messages data
    categories_filepath (str): filepath for the categories data

    Returns:
    pandas.DataFrame: DataFrame with messages and categories merged
    """
    
    # Reads and merge data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df
    

def clean_data(df):
    """
    Cleans messages data with corresponding categories.
    Eeach category is represented by a different column one hot encoded.
    The resulting dataframe also has duplicates dropped.

    Parameters:
    df (pandas.DataFrame): DataFrame containg messages data with their corresponding categories

    Returns:
    pandas.DataFrame: DataFrame with messages cleaned
    """
    
    # Expand categories in different columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda c: c[:-2])
    categories.columns = category_colnames
    
    # Set category values to be the last character of the string (binary value)
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    # Concatenate the resulting transformed categories data
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Saves the cleaned dataset of messages in a database provided by database_filename

    Parameters:
    df (pandas.DataFrame): DataFrame containing messages cleaned
    database_filename (str): filepath of the database onto which to be written
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
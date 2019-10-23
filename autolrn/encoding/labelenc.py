"""Label encode and One-Hot encode dataframes."""

from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import merge
import numpy as np


# Auto encodes any dataframe column of type category or object.
def dummy_encode(df):
    """
    Encode any dataframe column of type category or object.

    ---
    df: pandas dataframe
    """
    columnsToEncode = list(
        df.select_dtypes(include=['category', 'object']))

    df1 = df.copy()

    for feature in columnsToEncode:
        le = LabelEncoder()
        try:
            df1[feature] = le.fit_transform(df[feature].astype(str))
        except Exception as e:
            print(e)
            print('Error encoding ' + feature)
    return df1


#######

def le_encode_column(column):
    """Label-encode pandas DataFrame column or Series"""
    le = LabelEncoder()
    le_column = le.fit_transform(column.astype(str))
    if isinstance(column, DataFrame):
        le_column = Series(le_column).to_frame(column.name)
        # equivalent to:
        # le_column = DataFrame(le_column, columns=[df_column.name])
    elif isinstance(column, Series):
        le_column = Series(le_column)
    else:
        raise TypeError(
            "'column' should be of type pandas.DataFrame/Series")

    return le_column


def encode_df_column(df_column):
    """Convert dataframe column 'df_column' into df w dummy variables."""
    # print("column name: ", df_column.name)

    # if df_column.isin([' ']):
    #     df_column = df_column.str.replace(' ', '_')

    try:
        enc_df_column = get_dummies(
            df_column, prefix=df_column.name, prefix_sep='_')
    except MemoryError as me:
        print(me)
        print("MemoryError! Column: " + df_column.name)
        print("Proceed to label-encoding")
        enc_df_column = le_encode_column(df_column)
    except KeyError as ke:
        print(ke)
        print("KeyError! Column: " + df_column.name)
    except ValueError as ve:
        print(ve)
        print("ValueError! Column: " + df_column.name)
    except Exception:
        print('Error encoding feature ' + df_column.name)

    # print("column head", enc_df_column.head(1))

    assert (len(enc_df_column) == len(df_column)), \
    "Ouch! Encoded column's different length than original's!"

    return enc_df_column


def get_date_features(df, freqs=None):
    """
    Get dates objects from dataframe.

    ---
    df: pandas Dataframe
    freqs: frequencies of datetime objects
    """
    new_df = DataFrame()

    if freqs is None:
        freqs = ['Year', 'Month', 'Day', 'Week', 'hour', 'min']
    else:
        for f in freqs:
            if f not in ('Year', 'Month', 'Day', 'Week', 'hour', 'min'):
                raise ValueError(
                    "'%s' is not a valid value. Valid values are:"
                    "['Year', 'Month', 'Day', 'hour', 'min']"
                    % f)

    for feature in df.columns:
        if df[feature].dtype == 'datetime64[ns]':
            for f in freqs:
                try:
                    if f == 'Year':
                        new_df[f] = df[feature].dt.year
                    elif f == 'Month':
                        new_df[f] = df[feature].dt.month
                    elif f == 'Day':
                        new_df[f] = df[feature].dt.day
                    elif f == 'Week':
                        new_df[f] = df[feature].dt.week
                    elif f == 'hour':
                        new_df[f] = df[feature].dt.hour
                    else:
                        new_df[f] = df[feature].dt.minute
                except KeyError as ke:
                    print(ke)
                    print("KeyError! Column: " + feature)
                except ValueError as ve:
                    print(ve)
                    print("ValueError! Column: " + feature)
                except Exception as e:
                    raise e
        else:
            new_df[feature] = df[feature]

    assert (len(new_df.index) == len(df.index)), \
    "Ouch, encoded dataframe's different length than original's!"

    # remove 0-columns
    new_df = new_df.loc[:, (new_df != 0).any(axis=0)]

    return new_df


def get_dummies_or_label_encode(df, target=None, dummy_cols=10, ohe_dates=False):
    """
    Label or One-Hot encode columns.

    ---
    df: pandas Dataframe

    ohe_dates: enable one-hot encoding of eventual date features
    """
    df.reset_index(drop=True, inplace=True)
    if target is None:
        new_df = DataFrame()
        cols = df.columns
    else:
        new_df = df[target].to_frame(name=target)
        cols = df.drop([target], axis=1).columns

    # print()
    # print("New df's columns:\n", cols)
    # print()

    original_length = len(df.index)

    columns_to_encode = list(
        df.select_dtypes(include=['category', 'object', 'int64']))

    for feature in cols:
        col = df[feature]
        if df[feature].dtype in (np.int64, 'object'):
            col = col.astype('category')
        nr_uniques = len(col.unique())
        # print("%s's nr_uniques:" % feature, nr_uniques)
        # if df[feature].dtype.name in column_types:
        if feature in columns_to_encode:
            try:
                if new_df.empty:
                    # print("new_df is empty")
                    if ohe_dates:
                        if feature in ('Year', 'Month', 'Day', 'Week', 'hour', 'min'):
                            new_df = encode_df_column(col)
                        else:
                            new_df = le_encode_column(col).to_frame(feature)
                    else:
                        if nr_uniques < dummy_cols:
                            new_df = encode_df_column(col)
                        else:
                            new_df = le_encode_column(col)
                            if isinstance(new_df, Series):
                                new_df = new_df.to_frame(feature)	# you forgot this
                else:
                    # merge() more efficient than concat
                    if ohe_dates:
                        if feature in ('Year', 'Month', 'Day', 'Week', 'hour', 'min'):
                            new_df = merge(
                                new_df, encode_df_column(col), left_index=True, 
                                right_index=True)
                        else:
                            new_df = merge(
                                new_df, le_encode_column(col).to_frame(feature), 
                                left_index=True, right_index=True)
                    else:
                        new_df = merge(
                            new_df, 
                            encode_df_column(col) if len(col.unique()) < dummy_cols \
                            else le_encode_column(col).to_frame(feature),
                            left_index=True, right_index=True)
                        # new_df = concat([
                        #     new_df, 
                        #     encode_df_column(col) if len(col.unique()) < dummy_cols \
                        #     else le_encode_column(col).to_frame(feature)], axis=1)
            except KeyError as ke:
                print(ke)
                print("KeyError! Column: " + feature)
            except ValueError as ve:
                print(ve)
                print("ValueError! Column: " + feature)
            except Exception as e:
                raise e
        else:
            if new_df.empty:
                # print("new_df is empty")
                new_df = col.to_frame(feature)
            else:
                # more efficient than concat
                new_df = merge(
                    new_df, col.to_frame(feature), 
                    left_index=True, right_index=True)
                # new_df = concat(
                #     [new_df, col.to_frame(feature)], axis=1)
        # print("New df's head:\n", new_df.head())
        # print("New df's length:", len(new_df.index))

    assert (len(new_df.index) == original_length), \
    "Ouch, encoded dataframe's different length than original's!"

    print()
    print("New final df's head:\n", new_df.head(3))
    print("New df's tail:\n", new_df.tail(3))
    # print("New df's columns:", list(new_df.columns))
    # print("New df's length:", len(new_df.index))
    # print("Old df's length:", original_length)

    return new_df



import pandas as pd


def unique_counts(df1):
    lst = []
    print('----------------------- (START) #1 - Unique values for DF -----------------------')
    print("returns purchases in total and unique other columns")
    print("Total purchases", ": ", len(df1))
    for i in df1.columns:
        count = df1[i].nunique()
        lst.append(str(i) + ": " + str(count))
        print(i, ": ", count)

    print('----------------------- (END) #1 - Unique values for DF -----------------------')
    return lst


def change_column_to_date(df, column_name):
    df[column_name] = df[column_name].astype(str) + ' 00:00:00'
    df[column_name] = df[column_name].apply(pd.to_datetime)
    return df


def print_output(type_of_string, string_to_print):
    if type_of_string == 'div':
        print('--------------------------- ' + string_to_print + ' ---------------------------')
    else:
        print(string_to_print)

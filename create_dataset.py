import pandas as pd
import numpy as np
from math import floor, ceil
from pandas.core.reshape.merge import merge
#from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def merge_data_sets():
    # ADD ALCOHOL RATE
    df_ar = pd.read_csv('data/alcohol rate.csv', delimiter=';')
    df_ar = df_ar.loc[df_ar['Dim1ValueCode'] == 'SA_TOTAL']
    df_ar = df_ar[['ParentLocationCode','ParentLocation','SpatialDimValueCode','Location','Period','FactValueNumeric']]
    df_ar = df_ar.rename(columns={'FactValueNumeric': 'AR'})

    # ADD SUICIDE RATE
    df_suicide = pd.read_csv('data/suicide rate.csv', delimiter=';')
    df_suicide = df_suicide[['Location','Period','Dim1ValueCode','FactValueNumeric']]
    df_suicide = df_suicide.rename(columns={'FactValueNumeric': 'SR'})
    data_set = pd.merge(df_ar, df_suicide, left_on=['Location', 'Period'], right_on=['Location', 'Period'])

    # ADD BMI
    df_bmi = pd.read_csv('data/mean bmi.csv', delimiter=';')
    df_bmi = df_bmi[['Location','Period','Dim1ValueCode','FactValueNumeric']]
    df_bmi = df_bmi.rename(columns={'FactValueNumeric': 'MBMI'})
    data_set = pd.merge(data_set, df_bmi, left_on=['Location', 'Period', 'Dim1ValueCode'], right_on=['Location', 'Period', 'Dim1ValueCode'])

    # ADD Fine particulate matter
    df_fpm = pd.read_csv('data/fine particulate matter.csv', delimiter=';')
    df_fpm = df_fpm.loc[df_fpm['Dim1ValueCode'] == 'TOTL']
    df_fpm = df_fpm[['Location','Period','FactValueNumeric']]
    df_fpm = df_fpm.rename(columns={'FactValueNumeric': 'FPM'})
    data_set = pd.merge(data_set, df_fpm, left_on=['Location', 'Period'], right_on=['Location', 'Period'])

    # ADD adolescent birth rate
    df_abr = pd.read_csv('data/adolescent birth rate.csv', delimiter=';')
    df_abr = df_abr[['Location','Period','FactValueNumeric']]
    df_abr = df_abr.rename(columns={'FactValueNumeric': 'ABR'})
    data_set = pd.merge(data_set, df_abr, left_on=['Location', 'Period'], right_on=['Location', 'Period'])

    return data_set

def calculate_intervals(max, min, no_int):
    interval_size = (max-min)/no_int
    intervals = []
    for i in range(no_int):
        interval = [floor(min+interval_size*i), ceil(min+interval_size*(i+1))]
        intervals.append(interval)
    return intervals

def set_intervals(df, ar_no_int=4, sr_no_int=4, bmi_no_int=4, fpm_no_int=4, abr_no_int=4):
    intervals = []
    for attribute, no_int in zip(("SR", "AR", "MBMI", "FPM", "ABR"), (sr_no_int, ar_no_int, bmi_no_int, fpm_no_int, abr_no_int)):
        # normalize
        #df[attribute] = (df[attribute]-df[attribute].min())/(df[attribute].max()-df[attribute].min())
        interval = calculate_intervals(df[attribute].max(), df[attribute].min(), no_int)
        intervals.append(interval)
    return intervals


def get_interval(value, intervals, name):
    for interval in intervals:
        if interval[0] <= value <= interval[1]:
            return name + ': ' + str(interval)


def convert_value_to_intervals(df, no_int_sr=40, no_int_ar=40, no_int_bmi=40, no_int_fpm=40, no_int_abr=40):
    data_set = df.copy()
    sr_intervals, ar_intervals, bmi_intervals, fpm_intervals, abr_intervals = set_intervals(data_set, no_int_ar, no_int_sr, no_int_bmi, no_int_fpm, no_int_abr)
    for attribute, int in zip(("SR", "AR", "MBMI", "FPM", "ABR"), (sr_intervals, ar_intervals, bmi_intervals, fpm_intervals, abr_intervals)):
        data_set[attribute] = data_set[attribute].apply(lambda x: get_interval(x, int, attribute))
    return data_set

def generate_rules(df):
    interval_range = range(3, 8)
    sr_interval_range = range(8, 14)
    data_sets = [(convert_value_to_intervals(df, sr, ar, bmi, fpm, abr), (sr, ar, bmi, fpm, abr)) for sr in sr_interval_range 
                                                                       for ar in interval_range
                                                                       for bmi in interval_range
                                                                       for fpm in interval_range
                                                                       for abr in interval_range]
    for data_set, interval_size_tuple in data_sets:
        rules = run_apriori(data_set)
        with open('rules/rule_output_' + str(interval_size_tuple), 'w') as f:
            f.write(str(rules.head(50)))


def run_apriori(data_set):
    #data_set["Period"] = data_set["Period"].apply(lambda x: str(x))
    data_set = data_set[['Dim1ValueCode', 'AR', 'SR', 'MBMI', 'FPM']]
    df = data_set.values.tolist()
    te = TransactionEncoder()
    te_array = te.fit(df).transform(df)
    df = pd.DataFrame(te_array, columns=te.columns_)
    print("Starting apriori algorithm...")
    frq_items = apriori(df, min_support = 0.05, use_colnames = True)
    # Collecting the inferred rules in a dataframe
    rules = association_rules(frq_items, metric="lift", min_threshold = 1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    print(rules.head(50))
    return rules


if __name__ == '__main__':
    data_set = merge_data_sets()
    generate_rules(data_set)
    #data_set = convert_value_to_intervals(data_set)
    #run_apriori(data_set.loc[data_set["Period"] == 2014])
    
from pandas.core.reshape import merge
from create_dataset import merge_data_sets


def write_to_csv():
    data_set = merge_data_sets()
    data_set.to_csv("data/data_set.csv", index=False)


if __name__ == "__main__":
    write_to_csv()
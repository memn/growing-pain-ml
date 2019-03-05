import numpy as np
import pandas as pd


def attribute_names():
    data = pd.read_csv("../data/growing_pain_sample_data.csv")
    print(data.keys())


def print_unique_column_name(column_name='frequencyofpain'):
    data = pd.read_csv("../data/growing_pain_sample_data.csv")
    localization = data[column_name]
    print(set(localization.unique()))


def count_of_non_empty_in(column_name):
    data = pd.read_csv("../data/growing_pain_sample_data.csv")
    column = data[column_name]
    print(column.value_counts())


def init_data():
    """
    drops columns having more than 10 na values.
    drops some predefined columns
    :return:
    """
    data = pd.read_csv("../data/growing_pain_sample_data.csv")
    data.replace('^\\s*$', np.nan, regex=True, inplace=True)
    dropping = [c for j, c in enumerate(data.keys()) if data[c].isna().sum() > 10]
    print(dropping)
    dropped = data.drop(dropping, axis=1)
    target = dropped['tanıaçıkadı']
    drop_anyway = ['dateofbirth', 'tanıaçıkadı', 'imaging', 'bonemarrowaspiration']
    drop_candidates = ['hxofarthritis', 'hxoftrauma', 'morningstiffness', 'limping', 'limitationofactivities',
                       'comorbidities', 'physicalexamination', 'Beightonscore',
                       'hypermobility', 'flatfoot']
    dropped.drop(drop_anyway + drop_candidates, axis=1, inplace=True)

    return dropped, target


def make_transformations(data):
    transformed = data.replace(['ayda 1',
                                'haftada 1 gün',
                                'haftada 1-2 gün',
                                'haftada 2 gün',
                                'haftada 2-3 gün',
                                'haftada 3 gün',
                                'haftada 3-4 gün',
                                'haftada 4 gü',
                                'haftada 4-5 gün',
                                'haftada 5 gün',
                                'hergün',
                                'her gün'],
                               [1 / 30,
                                4 / 30,
                                6 / 30,
                                8 / 30,
                                1 / 3,
                                2 / 5,
                                14 / 30,
                                16 / 30,
                                18 / 30,
                                2 / 3,
                                1.,
                                1.])

    columns = ['kol', 'bacak', 'baldır', 'diz', 'bel', 'dirsek', 'uyluk', 'kalça', 'ayak']
    actarr = np.array([[column in act for column in columns] for act in data['localizationofpain']])
    actdf = pd.DataFrame(actarr, columns=columns)
    transformed = transformed.join(actdf)
    transformed.drop(['localizationofpain'], axis=1, inplace=True)
    transformed = transformed.fillna(0)
    print(pd.isnull(transformed).any())
    return transformed


def update_labels(target):
    return target.replace(['büyüme ağrısı', 'era', 'jia', 'fmf'], [1, 2, 3, 4])


def split():
    data, target = init_data()
    data = make_transformations(data)
    target = update_labels(target)
    ba = data[:100], target[:100]
    era = data[100:125], target[100:125]
    jia = data[125:134], target[125:134]
    fmf = data[134:], target[134:]

    ba_d, ba_t = ba
    era_d, era_t = era
    jia_d, jia_t = jia
    fmf_d, fmf_t = fmf

    train_data = ba_d[:-8].append(era_d[:-3]).append(jia_d[:-2]).append(fmf_d[:-4])
    test_data = ba_d[-8:].append(era_d[-3:]).append(jia_d[-2:]).append(fmf_d[-4:])

    train_target = ba_t[:-8].append(era_t[:-3]).append(jia_t[:-2]).append(fmf_t[:-4])
    test_target = ba_t[-8:].append(era_t[-3:]).append(jia_t[-2:]).append(fmf_t[-4:])

    train = train_data, train_target
    test = test_data, test_target

    return train, test

    # büyüme ağrısı    100 : 8
    # era               25 : 3
    # jia                9 : 2
    # fmf               41 : 4

    # print(train_data + test_data)


if __name__ == '__main__':
    split()

import os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from copy import deepcopy
import argparse
from sklearn.model_selection import train_test_split
import scipy


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'ratings.dat')
        self.fpath_user = os.path.join(data_dir, 'users.dat')
        self.fpath_item = os.path.join(data_dir, 'movies.dat')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='::',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])
        df_user = pd.read_csv(self.fpath_user,
                              sep='::',
                              engine='python',
                              names=['user', 'gender', 'age', 'occupation', 'Zip-code'],
                              usecols=['user', 'gender', 'age'])
        df_item = pd.read_csv(self.fpath_item,
                              sep='::',
                              engine='python',
                              names=['item', 'title', 'genre'],
                              usecols=['item', 'genre'],
                              encoding='unicode_escape')

        df = pd.merge(df_rate, df_user, on='user')
        df = df.dropna(axis=0, how='any')
        # df = df[df['rate'] > 3]
        df = df.reset_index().drop(['index'], axis=1)
        df = pd.merge(df, df_item, on='item')
        df.user = df.user - 1
        # df.item = df.item - 1
        df, item_mapping = convert_unique_idx(df, 'item')

        # print("item_mapping:", item_mapping)

        return df, item_mapping


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    # print("df:", df[column_name])
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def gender_index(df):
    gender_dic = df.groupby('user')['gender'].apply(list).to_dict()
    index_F = []
    index_M = []
    for i in range(0, len(gender_dic)):
        if 'f' in gender_dic[i] or 'F' in gender_dic[i]:
            index_F.append(i)
        else:
            index_M.append(i)
    index_F = np.array(index_F)
    index_M = np.array(index_M)
    return index_F, index_M


def age_mapping(age):
    if age < 18:
        return 0
    if age <= 24:
        return 1
    if age <= 34:
        return 2
    if age <= 44:
        return 3
    if age <= 49:
        return 4
    if age <= 55:
        return 5
    else:
        return 6


def age_mapping_ml1m(age):
    if age == 1:
        return 0
    if age <= 18:
        return 1
    if age <= 25:
        return 2
    if age <= 35:
        return 3
    if age <= 45:
        return 4
    if age <= 50:
        return 5
    else:
        return 6


def age_index(df, user_size):
    age_dic = df.groupby('user')['age'].apply(list).to_dict()
    index_age = [[], [], [], [], [], [], []]
    for i in range(0, len(age_dic)):
        if 0 in age_dic[i]:
            index_age[0].append(i)
        elif 1 in age_dic[i]:
            index_age[1].append(i)
        elif 2 in age_dic[i]:
            index_age[2].append(i)
        elif 3 in age_dic[i]:
            index_age[3].append(i)
        elif 4 in age_dic[i]:
            index_age[4].append(i)
        elif 5 in age_dic[i]:
            index_age[5].append(i)
        elif 6 in age_dic[i]:
            index_age[6].append(i)

    for i in range(len(index_age)):
        index_age[i] = np.array(index_age[i])

    age_type = 7
    age_mask = torch.zeros(age_type, user_size)
    for i in range(user_size):
        for k in range(age_type):
            if i in index_age[k]:
                age_mask[k][i] = 1

    return index_age, age_mask


def pop_index(df):
    count = Counter(df['item'])
    common = count.most_common()
    item_size = len(set(df['item']))

    index_pop = [[], [], [], [], []]
    for i in range(item_size):
        if count[i] > common[int(0.2 * len(common))][1]:
            index_pop[0].append(i)
        elif count[i] > common[int(0.4 * len(common))][1]:
            index_pop[1].append(i)
        elif count[i] > common[int(0.6 * len(common))][1]:
            index_pop[2].append(i)
        elif count[i] > common[int(0.8 * len(common))][1]:
            index_pop[3].append(i)
        else:
            index_pop[4].append(i)

    for i in range(len(index_pop)):
        index_pop[i] = torch.tensor(index_pop[i])

    pop_size = 5
    pop_mask = torch.zeros(pop_size, item_size)
    for i in range(item_size):
        for k in range(pop_size):
            if i in index_pop[k]:
                pop_mask[k][i] = 1

    return index_pop, pop_mask


def genre_ml100k_index(df):
    df_genre = df[
        ['item', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15',
         'g16', 'g17', 'g18']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True).drop(columns=['item'])
    genre_name = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15',
                  'g16', 'g17', 'g18']
    index_genre = []
    for genre in genre_name:
        index_genre.append(torch.tensor(np.flatnonzero(df_genre[genre])).long())

    genre_mask = df_genre.to_numpy().T
    genre_mask = torch.FloatTensor(genre_mask)

    return index_genre, genre_mask


def genre_ml1m_index(df):
    df_genre = df[['item', 'genre']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    ls = df_genre['genre'].tolist()
    for i in range(len(ls)):
        ls[i] = ls[i].split("|")
    for i in range(len(ls)):
        for j in range(len(ls[i])):
            if ls[i][j] == 'Action':
                ls[i][j] = 0
            elif ls[i][j] == 'Adventure':
                ls[i][j] = 1
            elif ls[i][j] == 'Animation':
                ls[i][j] = 2
            elif ls[i][j] == "Children's":
                ls[i][j] = 3
            elif ls[i][j] == 'Comedy':
                ls[i][j] = 4
            elif ls[i][j] == 'Crime':
                ls[i][j] = 5
            elif ls[i][j] == 'Documentary':
                ls[i][j] = 6
            elif ls[i][j] == 'Drama':
                ls[i][j] = 7
            elif ls[i][j] == 'Fantasy':
                ls[i][j] = 8
            elif ls[i][j] == 'Film-Noir':
                ls[i][j] = 9
            elif ls[i][j] == 'Horror':
                ls[i][j] = 10
            elif ls[i][j] == 'Musical':
                ls[i][j] = 11
            elif ls[i][j] == 'Mystery':
                ls[i][j] = 12
            elif ls[i][j] == 'Romance':
                ls[i][j] = 13
            elif ls[i][j] == 'Sci-Fi':
                ls[i][j] = 14
            elif ls[i][j] == 'Thriller':
                ls[i][j] = 15
            elif ls[i][j] == 'War':
                ls[i][j] = 16
            elif ls[i][j] == 'Western':
                ls[i][j] = 17

    # print("ls:", ls)

    genre_mask = torch.zeros(18, len(df_genre))
    for i in range(len(df_genre)):
        for k in range(0, 18):
            if k in ls[i]:
                genre_mask[k][i] = 1

    index_genre = []
    for i in range(genre_mask.shape[0]):
        index_genre.append(torch.tensor(np.where(genre_mask[i] == 1)[0]).long())

    return index_genre, genre_mask


def preprocessing(args):
    data_dir = os.path.join('/home/haolun/projects/InterFair/data', args.data)

    if args.data == 'ml-1m':
        df, item_mapping = MovieLens1M(data_dir).load()

    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())

    print("user_size:", user_size)
    print("item_size:", item_size)

    """construct matrix_label"""
    df_rate = df[["user", "item", "rate"]]
    df_rate = df_rate[df_rate['rate'] > 3]
    df_rate = df_rate.reset_index().drop(['index'], axis=1)
    df_rate['rate'] = 1

    matrix_label = scipy.sparse.csr_matrix(
        (np.array(df_rate['rate']), (np.array(df_rate['user']), np.array(df_rate['item']))))

    return df, item_mapping, matrix_label, user_size, item_size


def obtain_group_index(df, args):
    user_size = len(df['user'].unique())

    index_F, index_M = gender_index(df)
    index_gender = [torch.tensor(index_F).long(), torch.tensor(index_M).long()]
    index_age, age_mask = age_index(df, user_size)
    index_pop, pop_mask = pop_index(df)
    # print("index_age:", index_age)
    # print("index_pop:", index_pop)
    index_genre = []
    if args.data == 'ml-100k':
        index_genre, genre_mask = genre_ml100k_index(df)
    elif args.data == 'ml-1m':
        index_genre, genre_mask = genre_ml1m_index(df)

    # print("pop_mask:", pop_mask, pop_mask.shape)
    # print("genre_mask:", genre_mask, genre_mask.shape)

    return index_F, index_M, index_gender, index_age, index_genre, index_pop, age_mask, pop_mask, genre_mask


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml-1m")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    df, item_mapping, matrix_label, user_size, item_size = preprocessing(args)
    print("matrix_label:", matrix_label.todense().shape)

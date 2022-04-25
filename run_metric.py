import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from copy import deepcopy
import argparse
import time
from argparse import ArgumentParser
# from . import read_data
from read_data import preprocessing, obtain_group_index
from scipy.special import softmax
from tqdm import tqdm, trange
from Disparity_Metrics import *
import json


def parser_args():
    parser = ArgumentParser(description="JMEF")
    parser.add_argument('--data', type=str, default='ml-100k', choices=['ml-1m', 'ml-100k'],
                        help="File path for data")
    parser.add_argument('--gpu_id', type=int, default=0)
    # Seed
    parser.add_argument('--seed', type=int, default=0, help="Seed (For reproducability)")
    # Model
    parser.add_argument('--model', type=str, default='Pop')
    parser.add_argument('--gamma', type=float, default=0.8, help="patience factor")
    parser.add_argument('--temp', type=float, default=0.1, help="temperature. how soft the ranks to be")
    parser.add_argument('--s_ep', type=int, default=100)
    parser.add_argument('--r_ep', type=int, default=1)
    parser.add_argument('--norm', type=str, default='N')
    parser.add_argument('--coll', type=str, default='Y')
    parser.add_argument('--age', type=str, default='N')

    parser.add_argument('--conduct', type=str, default='st')
    return parser.parse_args()


def normalize_matrix_by_row(matrix):
    sum_of_rows = matrix.sum(axis=1)
    normalized_matrix = matrix / sum_of_rows[:, np.newaxis]
    return normalized_matrix


def eval_function_stochas(save_df, user_label, item_label, matrix_label, args, rand_tau=1):
    # construct E_target
    num_rel = matrix_label.sum(1, keepdims=True).reshape(-1, 1).astype("float")
    num_rel[num_rel == 0.0] = 1.0

    exposure_rel = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float"))
    E_target = exposure_rel / num_rel * matrix_label

    # construct E_collect
    if args.coll == 'Y':
        E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()
    else:
        E_collect = np.zeros((E_target.shape[0], E_target.shape[1]))

    # construct E_system
    user_size = E_target.shape[0]

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)
    if args.norm == 'Y':
        top_score = normalize_matrix_by_row(top_score)
    weight = softmax(top_score / rand_tau, axis=1)
    # weight = softmax(np.power(top_score, rand_tau), axis=1)

    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))

    # put the exposure value into the selected positions
    IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp = [], [], [], [], [], []
    sample_times = args.s_ep
    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    for sample_epoch in trange(sample_times, ascii=False):
        E_system_tmp = np.zeros((E_target.shape[0], E_target.shape[1]))
        exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")  # pre-compute the exposure_vector
        for i in range(user_size):
            tmp_selected = np.random.choice(top_item_id[i], 100, replace=False, p=weight[i])
            E_system_tmp[i][tmp_selected] = exp_vector
        E_system += E_system_tmp

        # if sample_epoch < 10:
        #     E_system_tmp = torch.from_numpy(E_system_tmp)
        #     IIF_sp.append(II_F(E_system_tmp, E_target, E_collect, indicator))
        #     GIF_sp.append(GI_F_mask(E_system_tmp, E_target, E_collect, user_label, indicator))
        #     AIF_sp.append(AI_F_mask(E_system_tmp, E_target, E_collect, indicator))
        #     IGF_sp.append(IG_F_mask(E_system_tmp, E_target, E_collect, item_label, indicator))
        #     GGF_sp.append(GG_F_mask(E_system_tmp, E_target, E_collect, user_label, item_label, indicator)[:3])
        #     AGF_sp.append(AG_F_mask(E_system_tmp, E_target, E_collect, item_label, indicator))

    E_system /= sample_times

    E_system = torch.from_numpy(E_system)

    IIF_all = II_F(E_system, E_target, E_collect, indicator)
    GIF_all = GI_F_mask(E_system, E_target, E_collect, user_label, indicator)
    AIF_all = AI_F_mask(E_system, E_target, E_collect, indicator)
    IGF_all = IG_F_mask(E_system, E_target, E_collect, item_label, indicator)
    GGF_all = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[:3]
    AGF_all = AG_F_mask(E_system, E_target, E_collect, item_label, indicator)
    # GG_target_stochas = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
    # GG_system_stochas = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[4]

    return IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all  # , IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp
    # GG_system_stochas, GG_target_stochas


def eval_function_static(save_df, user_label, item_label, matrix_label, args):
    # construct E_target
    num_rel = matrix_label.sum(1, keepdims=True).reshape(-1, 1).astype("float")
    num_rel[num_rel == 0.0] = 1.0

    exposure_rel = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float"))
    E_target = exposure_rel / num_rel * matrix_label

    # construct E_system
    user_size = E_target.shape[0]

    # construct E_collect
    if args.coll == 'Y':
        E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()
    else:
        E_collect = np.zeros((E_target.shape[0], E_target.shape[1]))

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)

    # put the exposure value into the selected positions
    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
    for i in range(user_size):
        E_system[i][top_item_id[i]] = exp_vector

    # print("E_target:", E_target.sum())
    # print("E_system:", E_system.sum())
    # print("E_collect:", E_collect.sum())

    E_system = torch.from_numpy(E_system)
    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    IIF = II_F(E_system, E_target, E_collect, indicator)
    GIF = GI_F_mask(E_system, E_target, E_collect, user_label, indicator)
    AIF = AI_F_mask(E_system, E_target, E_collect, indicator)
    IGF = IG_F_mask(E_system, E_target, E_collect, item_label, indicator)
    GGF = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[:3]
    AGF = AG_F_mask(E_system, E_target, E_collect, item_label, indicator)
    # GG_target = GG_F_mask(E_system, E_target, user_label, item_label, indicator)[1]
    # GG_system = GG_F_mask(E_system, E_target, user_label, item_label, indicator)[2]

    return IIF, GIF, IGF, GGF, AIF, AGF


def compute_stochas(args):
    save_df = pd.read_csv('./saved_model/run-{}-ml-1M-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    save_df = save_df.rename(columns={0: "user", 2: "item", 4: "score"})
    save_df.user = save_df.user - 1
    save_df['item'] = save_df['item'].map(item_mapping)
    save_df = save_df.sort_values(["user", "score"], ascending=[True, False])
    save_df = save_df.reset_index().drop(['index'], axis=1)
    if args.model == 'LDA':
        save_df["score"] = save_df["score"] * 1000
    elif args.model == 'Pop':
        save_df["score"] = save_df["score"] * 10
    elif args.model in ["PLSA", "RM1", "RSV", "CHI2", "HT", "KLD", "SVD", "UIR", "RM2", "LMWU", "LMWI", "NNU", "NNI"]:
        args.norm = 'Y'

    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    save_IIF_sp, save_IGF_sp, save_GIF_sp, save_GGF_sp, save_AIF_sp, save_AGF_sp = [], [], [], [], [], []
    save_IID_sp, save_IGD_sp, save_GID_sp, save_GGD_sp, save_AID_sp, save_AGD_sp = [], [], [], [], [], []
    save_IIR_sp, save_IGR_sp, save_GIR_sp, save_GGR_sp, save_AIR_sp, save_AGR_sp = [], [], [], [], [], []

    rand_tau_list = [8, 4, 2, 1, 0.5, 0.25, 0.125]
    # rand_tau_list = [1, 1e-1, 1e-2, 5e-2, 1e-3, 5e-3]
    len_tau = len(rand_tau_list)

    """evaluate on whole"""
    for epoch in range(args.r_ep):
        print("epoch:", epoch)
        for i in range(len_tau):
            print("tau={}".format(rand_tau_list[i]))

            # IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all, IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp \
            IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all \
                = eval_function_stochas(save_df, user_label, item_label, matrix_label, args, rand_tau=rand_tau_list[i])

            save_IIF.append(IIF_all[0].item())
            save_GIF.append(GIF_all[0].item())
            save_IGF.append(IGF_all[0].item())
            save_GGF.append(GGF_all[0].item())
            save_AIF.append(AIF_all[0].item())
            save_AGF.append(AGF_all[0].item())

            save_IID.append(IIF_all[1].item())
            save_GID.append(GIF_all[1].item())
            save_IGD.append(IGF_all[1].item())
            save_GGD.append(GGF_all[1].item())
            save_AID.append(AIF_all[1].item())
            save_AGD.append(AGF_all[1].item())

            save_IIR.append(IIF_all[2].item())
            save_GIR.append(GIF_all[2].item())
            save_IGR.append(IGF_all[2].item())
            save_GGR.append(GGF_all[2].item())
            save_AIR.append(AIF_all[2].item())
            save_AGR.append(AGF_all[2].item())

    dict_all = {"IIF": save_IIF, "IGF": save_IGF, "GIF": save_GIF, "GGF": save_GGF, "AIF": save_AIF, "AGF": save_AGF,
                "IID": save_IID, "IGD": save_IGD, "GID": save_GID, "GGD": save_GGD, "AID": save_AID, "AGD": save_AGD,
                "IIR": save_IIR, "IGR": save_IGR, "GIR": save_GIR, "GGR": save_GGR, "AIR": save_AIR, "AGR": save_AGR}

    for key in dict_all:
        if args.age == 'Y':
            with open("./save_exp/{}/{}_all_{}_Y.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict_all[key], fp)
        else:
            with open(
                    "./save_exp/{}/{}_all_{}.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict_all[key], fp)


def compute_static(args):
    save_df = pd.read_csv('./saved_model/run-{}-ml-1M-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    save_df = save_df.rename(columns={0: "user", 2: "item", 4: "score"})
    save_df.user = save_df.user - 1
    save_df['item'] = save_df['item'].map(item_mapping)
    save_df = save_df.sort_values(["user", "score"], ascending=[True, False])
    save_df = save_df.reset_index().drop(['index'], axis=1)

    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all \
        = eval_function_static(save_df, user_label, item_label, matrix_label, args)
    # print("IIF_all:", IIF_all)
    # print("IIF_all[0]:", IIF_all[0])
    # print("IIF_all[0]:", IIF_all[0].item())
    save_IIF.append(IIF_all[0].item())
    save_GIF.append(GIF_all[0].item())
    save_IGF.append(IGF_all[0].item())
    save_GGF.append(GGF_all[0].item())
    save_AIF.append(AIF_all[0].item())
    save_AGF.append(AGF_all[0].item())

    save_IID.append(IIF_all[1].item())
    save_GID.append(GIF_all[1].item())
    save_IGD.append(IGF_all[1].item())
    save_GGD.append(GGF_all[1].item())
    save_AID.append(AIF_all[1].item())
    save_AGD.append(AGF_all[1].item())

    save_IIR.append(IIF_all[2].item())
    save_GIR.append(GIF_all[2].item())
    save_IGR.append(IGF_all[2].item())
    save_GGR.append(GGF_all[2].item())
    save_AIR.append(AIF_all[2].item())
    save_AGR.append(AGF_all[2].item())

    print("save_IIF:", save_IIF)
    print("save_IID:", save_IID)
    print("save_IIR:", save_IIR)

    dict = {"IIF": save_IIF, "IGF": save_IGF, "GIF": save_GIF, "GGF": save_GGF, "AIF": save_AIF, "AGF": save_AGF,
            "IID": save_IID, "IGD": save_IGD, "GID": save_GID, "GGD": save_GGD, "AID": save_AID, "AGD": save_AGD,
            "IIR": save_IIR, "IGR": save_IGR, "GIR": save_GIR, "GGR": save_GGR, "AIR": save_AIR, "AGR": save_AGR}

    for key in dict:
        if args.age == 'Y':
            with open("./save_exp/{}/{}_all_{}_static_Y.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict[key], fp)
        else:
            with open("./save_exp/{}/{}_all_{}_static.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict[key], fp)


def compute_exp_matrix(args):
    save_df = pd.read_csv('./saved_model/run-{}-ml-1M-fold1.txt.gz'.format(args.model),
                          compression='gzip', header=None, sep='\t', quotechar='"', usecols=[0, 2, 4])
    save_df = save_df.rename(columns={0: "user", 2: "item", 4: "score"})
    save_df.user = save_df.user - 1
    save_df['item'] = save_df['item'].map(item_mapping)
    save_df = save_df.sort_values(["user", "score"], ascending=[True, False])
    save_df = save_df.reset_index().drop(['index'], axis=1)
    if args.model == 'LDA':
        save_df["score"] = save_df["score"] * 1000

    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    # rand_tau_list = [2, 4, 8, 16]
    rand_tau_list = [0.125, 8]
    len_tau = len(rand_tau_list)

    # """evaluate on whole"""
    # for i in range(len_tau):
    #     rand_tau = rand_tau_list[i]
    #     print("tau={}".format(rand_tau))
    #     # construct E_target
    #     num_rel = matrix_label.sum(1, keepdims=True).reshape(-1, 1).astype("float")
    #     num_rel[num_rel == 0.0] = 1.0
    #
    #     exposure_rel = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float"))
    #     E_target = exposure_rel / num_rel * matrix_label
    #
    #     # construct E_collect
    #     E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()
    #
    #     # construct E_system
    #     user_size = E_target.shape[0]
    #
    #     top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    #     top_score = np.array(list(save_df["score"])).reshape(-1, 100)
    #     # top_score = normalize_matrix_by_row(top_score)
    #     weight = softmax(top_score / rand_tau, axis=1)
    #
    #     # put the exposure value into the selected positions
    #     sample_times = 100
    #     E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    #     for _ in trange(sample_times, ascii=False):
    #         E_system_tmp = np.zeros((E_target.shape[0], E_target.shape[1]))
    #         exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
    #         for i in range(user_size):
    #             tmp_selected = np.random.choice(top_item_id[i], 100, replace=False, p=weight[i])
    #             E_system_tmp[i][tmp_selected] = exp_vector
    #         E_system += E_system_tmp
    #     E_system /= sample_times
    #
    #     E_system = torch.from_numpy(E_system)
    #     E_target = torch.from_numpy(E_target)
    #     E_collect = torch.from_numpy(E_collect)
    #     indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    #     GG_target_stochas = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
    #     GG_system_stochas = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[4]
    #
    #     with open("./save_exp/{}/GG_MT_{}_{}.json".format(args.data, rand_tau, args.model), "w") as fp:
    #         json.dump(np.array(GG_target_stochas).tolist(), fp)
    #     with open("./save_exp/{}/GG_MS_{}_{}.json".format(args.data, rand_tau, args.model), "w") as fp:
    #         json.dump(np.array(GG_system_stochas).tolist(), fp)

    # construct E_target
    num_rel = matrix_label.sum(1, keepdims=True).reshape(-1, 1).astype("float")
    num_rel[num_rel == 0.0] = 1.0

    exposure_rel = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float"))
    E_target = exposure_rel / num_rel * matrix_label

    # construct E_collect
    E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()

    # construct E_system
    user_size = E_target.shape[0]

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)

    # put the exposure value into the selected positions
    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
    for i in range(user_size):
        E_system[i][top_item_id[i]] = exp_vector

    E_system = torch.from_numpy(E_system)
    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    GG_target_static = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
    GG_system_static = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[4]
    GG_collect = GG_F_mask(E_system, E_target, E_collect, user_label, item_label, indicator)[5]

    print("GG_target_static:", GG_target_static)
    print("GG_system_static:", GG_system_static)
    print("GG_collect:", GG_collect)

    with open("./save_exp/{}/GG_MT_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_target_static).tolist(), fp)
    with open("./save_exp/{}/GG_MS_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_system_static).tolist(), fp)
    with open("./save_exp/{}/GG_collect_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_collect).tolist(), fp)


if __name__ == '__main__':
    args = parser_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    """read data and attribute label"""
    df, item_mapping, matrix_label, user_size, item_size = preprocessing(args)
    index_F, index_M, index_gender, index_age, index_genre, index_pop, age_matrix, pop_mask, genre_matrix \
        = obtain_group_index(df, args)
    gender_matrix = torch.zeros(2, len(index_F) + len(index_M))
    for ind in index_F:
        gender_matrix[0][ind] = 1
    for ind in index_M:
        gender_matrix[1][ind] = 1
    print("gender_matrix:", gender_matrix.shape)
    print("genre_matrix:", genre_matrix.shape)
    # print("gender_matrix:", gender_matrix.sum(1))
    # print("genre_matrix:", genre_matrix.sum(1))
    gender_num = len(index_gender)
    genre_num = len(index_genre)
    print("gender_num:", gender_num)
    print("genre_num:", genre_num)

    if args.age == 'Y':
        user_label = age_matrix
    else:
        user_label = gender_matrix  # .to(args.device)
    item_label = genre_matrix  # .to(args.device)

    print("user_label:", user_label.shape)
    print("item_label:", item_label.shape)
    print("item_label:", item_label[0])

    # matrix_label = torch.from_numpy(matrix_label.todense()).float().to(args.device)
    matrix_label = np.array(matrix_label.todense())
    # print("matrix_label:", matrix_label, type(matrix_label))

    print("norm:", args.norm)
    print("coll:", args.coll)
    print("model:", args.model)

    if args.conduct == 'sh':
        compute_stochas(args)
    elif args.conduct == 'st':
        compute_static(args)
    # compute_exp_matrix(args)
    print("_________________________________")

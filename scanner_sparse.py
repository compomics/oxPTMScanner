import argparse
from itertools import combinations
from collections import Counter
import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from operator import itemgetter


from scipy.stats import randint
from scipy.stats import uniform
from numpy.random import ranf
import scipy

from math import log

import numpy as np
import numpy.random as np_random
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
from sklearn.metrics import auc


def parse_msp(msp_entry, tic_normalization=True, min_perc=False, top=50, max_dist=275):  # 0.0005
    identifier = ""
    mz_list = []
    intensity_list = []

    if tic_normalization: tot_tic = 0.0

    for line in msp_entry:
        line = line.rstrip()
        if line == "": continue
        if line.startswith("Name: "):
            identifier = line.lstrip("Name: ").replace(",","_")
            continue
        if ":" in line: continue

        splitline = line.split("\t")

        mz_list.append(float(splitline[0]))
        intensity_list.append(float(splitline[1]))
        if tic_normalization: tot_tic += intensity_list[-1]

    if tic_normalization:
        for index, intens in enumerate(intensity_list):
            intensity_list[index] = intens / tot_tic

    intensity_list.insert(0, 1.0)
    mz_list.insert(0, 0.0)

    gr_intensity_list = []
    gr_mz_list = []
    if min_perc:
        for i, mz in zip(intensity_list, mz_list):
            if i > min_perc:
                gr_intensity_list.append(i)
                gr_mz_list.append(mz)
    if top:
        if len(intensity_list) > top:
            idxs = np.sort(np.argpartition(np.array(intensity_list), -top)[-top:])
            gr_mz_list = [mz_list[idx] for idx in idxs]
            gr_intensity_list = [intensity_list[idx] for idx in idxs]
        else:
            gr_mz_list = mz_list
            gr_intensity_list = intensity_list

    return (identifier, gr_mz_list, gr_intensity_list)


def parse_mgf(msp_entry, tic_normalization=True, min_perc=False, top=100):  # 0.0005
    identifier = ""
    mz_list = [0.0]
    intensity_list = [1.0]

    if tic_normalization: tot_tic = 0.0

    for line in msp_entry:
        line = line.rstrip()
        if line == "": continue
        if line.startswith("TITLE="):
            identifier = line.lstrip("TITLE=")
            continue
        if "=" in line: continue

        splitline = line.split(" ")

        mz_list.append(float(splitline[0]))
        intensity_list.append(float(splitline[1]))
        if tic_normalization: tot_tic += intensity_list[-1]

    if tic_normalization:
        for index, intens in enumerate(intensity_list):
            intensity_list[index] = intens / tot_tic

    gr_intensity_list = []
    gr_mz_list = []
    if min_perc:
        for i, mz in zip(intensity_list, mz_list):
            if i > min_perc:
                gr_intensity_list.append(i)
                gr_mz_list.append(mz)
    if top:
        if len(intensity_list) > top:
            idxs = np.sort(np.argpartition(np.array(intensity_list), -top)[-top:])
            gr_mz_list = [mz_list[idx] for idx in idxs]
            gr_intensity_list = [intensity_list[idx] for idx in idxs]
        else:
            gr_mz_list = mz_list
            gr_intensity_list = intensity_list

    return (identifier, gr_mz_list, gr_intensity_list)


def get_feats(mz_list, intensity_list, feat_matrix, instance_index, feats, max_dist=500, allowed_c=[]):
    spectrum = zip(mz_list, intensity_list)
    allowed_c = set(allowed_c)
    dists_mz = []
    dists_mz_intens = []
    for c in combinations(spectrum, 2):
        dist_mz = abs(c[0][0] - c[1][0])
        if dist_mz > max_dist: continue
        if len(allowed_c) != 0:
            if dist_mz not in allowed_c: continue
        dists_mz.append(dist_mz)
        dists_mz_intens.append(c[0][1] + c[1][1])
    # print(c)
    # feat_matrix[instance_index,np.digitize(dist_mz,feats)-1] += c[0][1]+c[1][1]
    # print(np.digitize(dists_mz,feats))

    index_bins = np.digitize(dists_mz, feats)
    # np.searchsorted(dists_mz,feats)
    # np.digitize(dists_mz,feats)
    for index, intens in zip(index_bins, dists_mz_intens):
        # feat_matrix[instance_index,index-1] += intens
        feat_matrix[instance_index, index - 1] += intens

    # print(np.searchsorted(dists_mz,feats))
    # np.searchsorted(dists_mz,feats)
    # print(dists_mz)
    return (feat_matrix)


def get_feats_intens(mz_list, intensity_list, max_dist=275, feature_dict={}):
    spectrum = zip(mz_list, intensity_list)
    for c in combinations(spectrum, 2):
        dist_mz = abs(round(c[0][0] - c[1][0], 2))
        if dist_mz > max_dist: continue
        try:
            feature_dict[dist_mz] += c[0][1] + c[1][1]
        except KeyError:
            feature_dict[dist_mz] = c[0][1] + c[1][1]
    return (feature_dict)


def num_instances_msp(msp_file):  # 0.0005
    #infile = open(msp_file)

    num_instances = 0
    for line in msp_file:
        if line.startswith("Name: "):
            num_instances += 1
    return (num_instances)


def read_msp(infile, feat_lim_file="",
             sum_feats=False, selected_features=[],
             max_dist=275, step_size=0.005, feat_bins=[],
             top_peaks=50):

    if len(feat_lim_file) > 0:
        selected_features = [float(f.strip()) for f in open(feat_lim_file).readlines()]
    infile = infile.readlines()
    counter = 0
    temp_entry = []
    instance_names = []
    num_instances = num_instances_msp(infile)
    print(num_instances)

    if len(feat_bins) == 0: feat_bins = np.arange(0, max_dist + step_size, step_size)
    # feat_matrix = scipy.sparse.csr_matrix((num_instances, len(feat_bins)),dtype=np.float32)
    feat_matrix = scipy.sparse.lil_matrix((num_instances, len(feat_bins)), dtype=np.float32)

    for line in infile:
        if line.startswith("Name: "):
            if len(temp_entry) == 0:
                temp_entry.append(line.strip())
                continue

            identifier, mz_list, intensity_list = parse_msp(temp_entry, max_dist=max_dist, top=top_peaks)
            instance_names.append(identifier)
            feat_matrix = get_feats(mz_list, intensity_list, feat_matrix, counter, feat_bins,
                                    allowed_c=selected_features)

            temp_entry = [line]

            print(counter)
            counter += 1

        temp_entry.append(line.strip())

    if len(temp_entry) == 0:
        temp_entry.append(line.strip())
        return (feat_matrix.asformat("csr"), feat_bins, instance_names, counter)

    identifier, mz_list, intensity_list = parse_msp(temp_entry, max_dist=max_dist, top=top_peaks)
    instance_names.append(identifier)
    feat_matrix = get_feats(mz_list, intensity_list, feat_matrix, counter, feat_bins, allowed_c=selected_features)

    print(counter)
    counter += 1

    return (feat_matrix.asformat("csr"), feat_bins, instance_names, counter)


def read_mgf(infile_name, feat_lim_file="", sum_feats=False, selected_features=[]):
    infile = open(infile_name)

    if len(feat_lim_file) > 0:
        selected_features = [float(f.strip()) for f in open("selected_features.txt").readlines()]

    counter = 0
    temp_entry = []
    feature_dict = {}
    pep_list = set()

    identifier = ""
    mz_list = []
    intensity_list = []

    for line in infile:
        if line.startswith("END IONS"):
            identifier, mz_list, intensity_list = parse_mgf(temp_entry)
            if sum_feats:
                feature_dict = get_feats_intens(mz_list, intensity_list, feature_dict=feature_dict)
            else:
                feature_dict[identifier] = get_feats(mz_list, intensity_list, allowed_c=selected_features)
            counter += 1
            print(counter)
            temp_entry = []
            continue
        if line.startswith("BEGIN IONS"):
            continue
        temp_entry.append(line)
    return (feature_dict, counter)


def read_mgf2(infile_name, feat_lim_file="", sum_feats=False, selected_features=[]):
    reader = mgf.read(infile_name)

    if len(feat_lim_file) > 0:
        selected_features = [float(f.strip()) for f in open("selected_features.txt").readlines()]

    counter = 0
    temp_entry = []
    feature_dict = {}
    pep_list = set()

    for spectrum in reader:
        if sum_feats:
            feature_dict = get_feats_intens(spectrum["m/z array"], spectrum["intensity array"],
                                            feature_dict=feature_dict)
        else:
            feature_dict[spectrum["params"]["title"]] = get_feats(spectrum["m/z array"], spectrum["intensity array"],
                                                                  allowed_c=selected_features)
        counter += 1
        print(counter)
    return (feature_dict)


def get_features_groups_min_occ(feature_dict, feature_dict2, min_occ=1):
    f_names = []
    for instance_name in feature_dict.keys():
        for f_name in feature_dict[instance_name].keys():
            f_names.append(f_name)

    for instance_name in feature_dict2.keys():
        for f_name in feature_dict2[instance_name].keys():
            f_names.append(f_name)
    f_names = [f for f, count in Counter(f_names).items() if count > min_occ]
    f_names.sort()
    print("Total number of features: %s" % (len(f_names)))
    # raw_input("stop")
    return (f_names)


def feats_to_matrix(feature_dict, selected_features):
    matrix = []

    counter = 0
    # column_names = ["identifier","y"]
    column_names = selected_features
    for instance_name in feature_dict.keys():
        print(counter)
        counter += 1
        feat_vector = []
        for fn in selected_features:
            try:
                feat_vector.append(feature_dict[instance_name][fn])
            except KeyError:
                feat_vector.append(0.0)
        matrix.append(feat_vector)
    matrix = np.array(matrix, dtype="float32")
    # matrix_pd = pd.DataFrame(matrix,dtype="float32")
    # matrix_pd.columns = column_names
    # matrix_pd.index = list(feature_dict.keys())
    # matrix_pd.to_csv("train_oxid_HS.csv",index=False)
    return (matrix)


def train_xgb(X, y):
    xgb_handle = xgb.XGBClassifier()

    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)

    param_dist = {
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        "reg_alpha": from_zero_positive,
        "min_child_weight": from_zero_positive,
    }

    n_iter_search = 25
    random_search = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
                                       n_iter=n_iter_search, verbose=10, scoring="roc_auc",
                                       n_jobs=1, cv=5)

    random_search_res_xgb = random_search.fit(X, y)
    xgb_model = random_search_res_xgb.best_estimator_

    return (xgb_model, random_search_res_xgb)


def train_xgb_lim(X, y, params_dist, out_dir="res/"):
    xgb_handle = xgb.XGBClassifier(**params_dist)  # .items()
    test_preds = cross_val_predict(xgb_handle, X, y, method="predict_proba")
    plot_roc(X, y, test_preds[:, 1], fname=out_dir + "roc.png")
    return (test_preds)


# raw_input("stop")
# n_iter_search = 1
# param_dist = dict([[k,[p]] for k,p in params_dist.items()])
# xgb_handle = xgb.XGBClassifier()
# random_search_feat = RandomizedSearchCV(xgb_handle, param_distributions=param_dist,
#								   n_iter=n_iter_search,verbose=10,scoring="roc_auc",
#								   n_jobs=1,cv=5)
#
# random_search_res_xgb_feat = random_search_feat.fit(X, y)
#
# xgb_handle_final = random_search_res_xgb_feat.best_estimator_
#
# return(xgb_handle_final)
# xgb_handle = xgb.XGBClassifier(params_dist)
# xgb_handle.fit(X,y)
# return(xgb_handle)


def plot_feat_imp(feats_index, feat_names, X, y, get_feat_imp=True, top_imp=100, out_dir="res/"):
    for fi in feats_index[0:10]:
        plt.boxplot([X.todense()[y == 1, :][:, fi], X.todense()[y == 0, :][:, fi]])
        plt.title(feat_names[fi])
        # plt.show()
        plt.savefig(out_dir + "%s_feat_groups.png" % (feat_names[fi]), bbox_inches='tight')
        plt.close()


def plot_train_distr(xgb_model, X, y):
    probs_oxid = xgb_model.predict_proba(X[y == 1])[:, 1]
    probs_native = xgb_model.predict_proba(X[y == 0])[:, 1]

    pd.Series(probs_oxid).plot(kind="density")
    pd.Series(probs_native).plot(kind="density")
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    # plt.show()
    plt.savefig('res_small/density_groups.png', bbox_inches='tight')
    plt.close()

    pd.Series(probs_oxid).plot(kind="density")
    pd.Series(probs_native).plot(kind="density")
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    # plt.show()
    plt.savefig('res_small/density_groups_zoomed.png', bbox_inches='tight')
    plt.close()

    plt.hist(probs_native, bins=100)
    plt.hist(probs_oxid, bins=100)
    # plt.show()
    plt.savefig('res_small/hist_groups.png', bbox_inches='tight')
    plt.close()

    plt.hist(probs_native, bins=100)
    plt.hist(probs_oxid, bins=100)
    axes = plt.gca()
    axes.set_ylim([0.0, 1000.0])
    # plt.show()
    plt.savefig('res_small/hist_groups_zoomed.png', bbox_inches='tight')
    plt.close()


def xgboost_to_wb(xgboost, outfile="model_big.pickle"):
    pickle.dump(xgboost, open(outfile, "wb"))


def get_diff_feats(df_zero, df_one, select_top=False, min_perc_diff=False, num_zero=1.0, num_one=1.0):
    mean_zero = np.mean(df_zero, axis=0) / num_zero

    # mean_zero = np.median(df_zero,axis=0)
    mean_one = np.mean(df_one, axis=0) / num_one
    # mean_one = np.median(df_one,axis=0)
    diff = np.absolute(mean_zero - mean_one)

    # perc_diff = [d/m for d,m in zip(list(diff),np.maximum(mean_zero,mean_one))]

    # plt.hist(diff,bins=100)
    # plt.show()
    # plt.close()
    # plt.scatter(diff,perc_diff)
    # plt.show()
    # plt.close()

    if select_top:
        return ([index for index, val in sorted(enumerate(list(diff)), key=itemgetter(1), reverse=True)][0:select_top])
    if min_perc_diff:
        return ([index for index, pd in enumerate(diff) if pd > min_perc_diff])
    return ([index for index, val in enumerate(list(diff))])


def plot_roc(X, y, test_preds, fname="res_small/roc.png"):
    fpr, tpr, thresholds = roc_curve(y, test_preds)
    plt.plot(fpr, tpr)
    plt.title(auc(fpr, tpr))
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def get_min_diff(zero_f="NIST/human_hcd_synthetic_oxidized.msp",
                 one_f="NIST/human_hcd_synthetic_native.msp",
                 outfile="res_small/selected_features_diff.txt",
                 top_mean=1000,
                 top_peaks=50,
                 max_distance=275,
                 distance_bins=0.005):
    if zero_f.endswith(".mgf"):
        feats_zero_sum, count_zero = read_mgf(zero_f, sum_feats=True)
    elif zero_f.endswith(".msp"):
        feats_zero_sum, feat_bins, instance_names, count_zero = read_msp(zero_f,
                                                                         sum_feats=True,
                                                                         max_dist=max_distance,
                                                                         step_size=distance_bins,
                                                                         top_peaks=top_peaks)
    else:
        return (False)

    if one_f.endswith(".mgf"):
        feats_one_sum, count_one = read_mgf(one_f, sum_feats=True)
    elif one_f.endswith(".msp"):
        feats_one_sum, feat_bins, instance_names, count_one = read_msp(one_f,
                                                                       sum_feats=True,
                                                                       max_dist=max_distance,
                                                                       step_size=distance_bins,
                                                                       top_peaks=top_peaks)
    else:
        return (False)

    # print(feats_zero_sum.median(axis=0).tolist()[0])
    diffs = [abs(m1 - m2) for m1, m2 in
             zip(feats_zero_sum.mean(axis=0).tolist()[0], feats_one_sum.mean(axis=0).tolist()[0])]
    # diffs = [abs(m1-m2) for m1,m2 in zip(np.median(feats_zero_sum,axis=0).tolist()[0],np.median(feats_one_sum,axis=0).tolist()[0])]
    # plt.scatter(feats_zero_sum.mean(axis=0).tolist()[0],feats_one_sum.mean(axis=0).tolist()[0],s=0.1)
    # plt.show()
    # plt.close()

    indexes_diff = sorted(list(enumerate(diffs)), key=itemgetter(1), reverse=True)
    # print(indexes_diff)
    selected_features_diff = [feat_bins[ind] for ind, val in indexes_diff[0:top_mean]]
    selected_features_diff.sort()

    diff_bins = [sfd + distance_bins for sfd in selected_features_diff]
    diff_bins.extend(selected_features_diff)
    diff_bins.sort()

    diff_bins = list(set(diff_bins))
    diff_bins.sort()

    # feats_zero_sum,feat_bins,count_zero = read_msp(zero_f,feat_bins=diff_bins)
    # feats_one_sum,feat_bins,count_one = read_msp(one_f,feat_bins=diff_bins)

    # diffs = [abs(m1-m2) for m1,m2 in zip(feats_zero_sum.mean(axis=0).tolist()[0],feats_one_sum.mean(axis=0).tolist()[0])]
    # diffs = [abs(m1-m2) for m1,m2 in zip(feats_zero_sum.median(axis=0).tolist()[0],feats_one_sum.median(axis=0).tolist()[0])]

    # plt.scatter(feats_zero_sum.mean(axis=0).tolist()[0],feats_one_sum.mean(axis=0).tolist()[0],s=0.1)
    # plt.show()
    # plt.close()

    outfile_feats = open(outfile, "w")
    outfile_feats.write("\n".join(map(str, diff_bins)))
    outfile_feats.close()

    return (diff_bins)


def train_initial_classifier(zero_f="NIST/human_hcd_synthetic_oxidized.msp",
                             one_f="NIST/human_hcd_synthetic_native.msp",
                             selected_features_diff=[],
                             top_mean=1000,
                             top_peaks=50,
                             max_distance=275,
                             distance_bins=0.005,
                             out_dir="res/"):
    if zero_f.endswith(".mgf"):
        feats_zero, count_zero = read_mgf(zero_f, sum_feats=False, selected_features=selected_features_diff)
    elif zero_f.endswith(".msp"):
        feats_zero, feat_bins, instance_names, count_zero = read_msp(zero_f,
                                                                     sum_feats=False,
                                                                     feat_bins=selected_features_diff,
                                                                     max_dist=max_distance,
                                                                     step_size=distance_bins,
                                                                     top_peaks=top_peaks)
    else:
        return (False)  # TODO display error!

    if one_f.endswith(".mgf"):
        feats_one, count_one = read_mgf(one_f, sum_feats=False, selected_features=selected_features_diff)
    elif one_f.endswith(".msp"):
        feats_one, feat_bins, instance_names, count_one = read_msp(one_f,
                                                                   sum_feats=False,
                                                                   feat_bins=selected_features_diff,
                                                                   max_dist=max_distance,
                                                                   step_size=distance_bins,
                                                                   top_peaks=top_peaks)
    else:
        return (False)  # TODO display error!

    # selected_features_occ = get_features_groups_min_occ(feats_zero,feats_one)

    # df_zero = feats_to_matrix(feats_zero,selected_features_occ)
    y = [0] * (count_zero)

    # df_one = feats_to_matrix(feats_one,selected_features_occ)
    y.extend([1] * (count_one))

    y = np.array(y)

    X = scipy.sparse.vstack((feats_zero, feats_one))

    xgb_model, random_search_res_xgb = train_xgb(X, y)
    print(random_search_res_xgb.best_params_)
    print(random_search_res_xgb.best_score_)

    train_xgb_lim(X, y, random_search_res_xgb.best_params_, out_dir=out_dir)

    plot_train_distr(xgb_model, X, y)
    xgboost_to_wb(random_search_res_xgb, outfile=out_dir + "model.pickle")

    # selected_features_diff = map(float,[l.strip() for l in open("res_small/selected_features_diff.txt").readlines()])
    random_search_res_xgb = pickle.load(open(out_dir + "model.pickle", "rb"))
    fscores = xgb_model.booster().get_fscore()
    fscores_list = sorted(list(fscores.items()), key=itemgetter(1), reverse=True)
    selected_features_indexes = map(int, [f.replace("f", "") for f, n in fscores_list])
    selected_features_xgboost = [selected_features_diff[sfp] for sfp in selected_features_indexes]

    plot_feat_imp(selected_features_indexes, selected_features_diff, X, y, out_dir=out_dir)

    return (random_search_res_xgb.best_params_, selected_features_xgboost)


def apply_model(infile_pred, infile_model, infile_features,filename, threshold_prob=0.5, out_dir="res/"):
    features = [f.strip() for f in open(infile_features).readlines()]
    if filename.endswith(".mgf"):
        feats, count_zero = read_mgf(infile_pred, sum_feats=True)
    elif filename.endswith(".msp"):
        feats, feat_bins, instance_names, count_inst = read_msp(infile_pred, sum_feats=False, feat_bins=features)
    else:
        return (False)
    print(feats.shape)
    print(feats)
    print(len(instance_names))
    print(instance_names)

    random_search_res_xgb = pickle.load(open(infile_model, "rb"))
    preds = pd.DataFrame(random_search_res_xgb.predict_proba(feats), index=instance_names,
                         columns=["Prob_class_0", "Prob_class_1"])
    print(preds)
    pd.Series(preds["Prob_class_1"]).plot(kind="density")
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    return(plt,preds)
    """
    plt.savefig(out_dir + "density_groups_zoomed.png", bbox_inches="tight")
    plt.close()

    print(list(preds.index[preds["Prob_class_1"] > 0.5]))
    # preds.index = instance_names
    preds.to_csv(out_dir + "predictions.csv")
"""

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--top_peaks", type=int, dest="top_peaks", default=100,
                        help="Number of peaks to extract and consider for combinations in a spectrum")

    parser.add_argument("--top_mean", type=int, dest="top_mean", default=2500,
                        help="The top bins in different mean between group A and B to learn on")

    parser.add_argument("--max_distance", type=int, dest="max_distance", default=275,
                        help="The maximum difference between peaks (maximum bin value)")

    parser.add_argument("--distance_bins", type=float, dest="distance_bins", default=0.005,
                        help="Distance in m/z of the bins")

    parser.add_argument("--file_a", type=str, dest="file_a", default="NIST/human_hcd_synthetic_native.msp",
                        help="The mgf or msp of group A")

    parser.add_argument("--file_b", type=str, dest="file_b", default="NIST/human_hcd_synthetic_oxidized.msp",
                        help="The mgf or msp of group B")

    parser.add_argument("--file_pred", type=str, dest="file_pred", default="NIST/human_hcd_synthetic_native.msp",
                        help="The mgf or msp to make predictions for")

    parser.add_argument("--out_dir", type=str, dest="out_dir", default="res/",
                        help="Directory where the results are written. WILL OVERWITE EXISTING FILES!")

    parser.add_argument("--make_pred", action="store_true",
                        help="Flag that can be included to indicate predictions are desired instead of training a model")

    parser.add_argument("--model", type=str, dest="model", default="res_small/model.pickle",
                        help="File that refers to a model that is used for predictions")

    parser.add_argument("--feats", type=str, dest="feats", default="res_small/selected_features_diff.txt",
                        help="File that refers to the features that are used in the model")

    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    results = parser.parse_args()

    return (results)


def main(infile):
    argu = parse_argument()

    if not argu.make_pred:
        selected_features_diff = get_min_diff(zero_f=argu.file_a,
                                              one_f=argu.file_b,
                                              outfile=argu.out_dir + "/selected_features.txt",
                                              top_peaks=argu.top_peaks,
                                              top_mean=argu.top_mean,
                                              max_distance=argu.max_distance,
                                              distance_bins=argu.distance_bins)

        random_search_params, selected_features_xgb = train_initial_classifier(zero_f=argu.file_a,
                                                                               one_f=argu.file_b,
                                                                               selected_features_diff=selected_features_diff)

    if argu.make_pred:
        apply_model(argu.file_pred, argu.model, argu.feats, out_dir=argu.out_dir)


if __name__ == "__main__":
    main()

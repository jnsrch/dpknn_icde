import util
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io import arff
from dimreduct import *


def load_arff(name):
        return arff.loadarff("campos_data/literature/{filename}/{filename}.arff".format(filename=name))


def prepare_data_arff(dataname, label_col=0, drop_col = None) -> object:
        data, meta = load_arff(dataname)
        flat_list = [item for sublist in data for item in sublist]
        data_array = np.array(flat_list).reshape(len(data), int(len(flat_list) / len(data)))
        df = pd.DataFrame(data=data_array)
        if drop_col is not None:
                exclude_col = df.columns[(label_col, drop_col), ]
        else:
                exclude_col = df.columns[label_col]


        data_out = df[df[df.columns[label_col]] == b'yes'].drop(exclude_col, axis=1)
        data_in = df[df[df.columns[label_col]] == b'no'].drop(exclude_col, axis=1)

        train, test = train_test_split(data_in, test_size=.2, random_state=42)

        hyper, train = train_test_split(train, test_size=.625, random_state=12)
        hyper_out, test_out = train_test_split(data_out, test_size=.5, random_state=12)

        test_labels = np.hstack((np.zeros(len(test)), np.ones(len(test_out))))
        test = np.vstack((test, test_out))

        hyper_train, hyper_test = train_test_split(hyper, test_size=.2, random_state=12)
        hyper_test_labels = np.hstack((np.zeros(len(hyper_test)), np.ones(len(hyper_out))))
        hyper_test = np.vstack((hyper_test, hyper_out))

        hyper_all = np.vstack((hyper_test, hyper_train))
        sc = StandardScaler()
        hyper_all = sc.fit_transform(hyper_all)
        hyper_all, reduct = dim_reduct(hyper_all)
        _, _, normalize = util.normalize(hyper_all, hyper_all[1:2,:]) # only fit normalizer

        train = normalize(reduct(sc.transform(train)))
        hyper_test = normalize(reduct(sc.transform(hyper_test)))
        hyper_train = normalize(reduct(sc.transform(hyper_train)))
        test = normalize(reduct(sc.transform(test)))

        # train are all inliers so we do not need labels
        return {"train": train, "test_labels": test_labels, "test": test,
                "hyper_train": hyper_train, "hyper_test": hyper_test,
                "hyper_test_labels": hyper_test_labels,
                "name": dataname}


def no_outliers(data_name):
        data = datasets2[data_name]
        return int(np.sum(data["test_labels"]))


aloi = prepare_data_arff("ALOI")
aloi["ks"] = (1,2)
aloi["k"] = 2
aloi["b"] = 4
aloi["wk"] = 2
aloi["wb"] = 3
ionosphere = prepare_data_arff("Ionosphere", label_col=-1)
ionosphere["ks"] = (1,2,4)
ionosphere["k"] = 1
ionosphere["b"] = 3
ionosphere["ks"] = (1,2)
ionosphere["wk"] = 2
ionosphere["wb"] = 4

glass = prepare_data_arff("Glass", label_col=-1)
glass["ks"] = (1,2,8,18)
glass["k"] = 2
glass["b"] = 3
glass["wk"] = 2
glass["wb"] = 3

lymphography = prepare_data_arff("Lymphography", label_col=-1) #idf
lymphography["ks"] = (1,4,14,39)
lymphography["k"] = 1
lymphography["b"] = 2
lymphography["wk"] = 14
lymphography["wb"] = 2

kddcup99 = prepare_data_arff("KDDCup99", label_col=-1) #idf
kddcup99["ks"] = (16,83,88,100)
kddcup99["k"] = 83
kddcup99["b"] = 4
kddcup99["wk"] = 83
kddcup99["wb"] = 4

pendigits = prepare_data_arff("PenDigits", label_col=-1) #v10
pendigits["ks"] = (1,2,12,19)
pendigits["k"] = 12
pendigits["b"] = 4
pendigits["wk"] = 19
pendigits["wb"] = 4


shuttle = prepare_data_arff("Shuttle", label_col=-1) #v10
shuttle["ks"] = (1,3,11)
shuttle["k"] = 11
shuttle["b"] = 2
shuttle["wk"] = 3
shuttle["wb"] = 3

waveform = prepare_data_arff("Waveform", label_col=-1) #v10
waveform["ks"] = (67,47,77,100)
waveform["k"] = 47
waveform["b"] = 4
waveform["wk"] = 47
waveform["wb"] = 4

wbc = prepare_data_arff("WBC", label_col=-1) #v10
wbc["ks"] = (3,4,5,15)
wbc["k"] = 3
wbc["b"] = 2
wbc["wk"] = 15
wbc["wb"] = 2

wdbc = prepare_data_arff("WDBC", label_col=-1) #v10
wdbc["ks"] = (14,67,44,73)
wdbc["k"] = 14
wdbc["b"] = 2
wdbc["wk"] = 73
wdbc["wb"] = 2

wpbc = prepare_data_arff("WPBC", label_col=-1)
wpbc["ks"] = (12,9,18,88)
wpbc["k"] = 12
wpbc["b"] = 4
wpbc["wk"] = 8
wpbc["wb"] = 4

# 7%
annthyroid = prepare_data_arff("Annthyroid", label_col=-1, drop_col=-2)
annthyroid["ks"] = (1,2)
annthyroid["k"] = 1
annthyroid["b"] = 3
annthyroid["wk"] = 1
annthyroid["wb"] = 3
arrhythmia = prepare_data_arff("Arrhythmia", label_col=-1, drop_col=-2) # 10%, v10
arrhythmia["ks"] = (1,23,34)
arrhythmia["k"] = 1
arrhythmia["b"] = 2
arrhythmia["wk"] = 1
arrhythmia["wb"] = 2
cardiotocography = prepare_data_arff("Cardiotocography", label_col=-1, drop_col=-2) # 10%, v10
cardiotocography["ks"] = (96,82,70,100)
cardiotocography["k"] = 100
cardiotocography["b"] = 4
cardiotocography["wk"] = 82
cardiotocography["wb"] = 2
heartdisease = prepare_data_arff("HeartDisease", label_col=-1, drop_col=-2) # 10%, v10
heartdisease["ks"] = (23,74,90,87)
heartdisease["k"] = 74
heartdisease["b"] = 2
heartdisease["wk"] = 23
heartdisease["wb"] = 2
hepatitis = prepare_data_arff("Hepatitis", label_col=-2, drop_col=-1) # orig
hepatitis["ks"] = (13,21,22,61,77)
hepatitis["k"] = 13
hepatitis["b"] = 2
hepatitis["wk"] = 13
hepatitis["wb"] = 2
internetads = prepare_data_arff("InternetAds", label_col=-1, drop_col=-2) # orig
internetads["ks"] = (6,9,13,22)
internetads["k"] = 6
internetads["b"] = 4
internetads["wk"] = 6
internetads["wb"] = 4
pageblocks = prepare_data_arff("PageBlocks", label_col=-2, drop_col=-1) # orig
pageblocks["ks"] = (41,100, 60)
pageblocks["k"] = 41
pageblocks["b"] = 4
pageblocks["wk"] = 41
pageblocks["wb"] = 4
parkinson = prepare_data_arff("Parkinson", label_col=-1, drop_col=-2) # 10%, v10
parkinson["ks"] = (1,4,5)
parkinson["k"] = 4
parkinson["b"] = 3
parkinson["wk"] = 4
parkinson["wb"] = 3
pima = prepare_data_arff("Pima", label_col=-1, drop_col=-2) # 10%, v10
pima["ks"] = (1,2,77,100)
pima["k"] = 1
pima["b"] = 3
pima["wk"] = 1
pima["wb"] = 4
spambase = prepare_data_arff("SpamBase", label_col=-1, drop_col=-2) # 10%, v10
spambase["ks"] = (6,7,9,14)
spambase["k"] = 14
spambase["b"] = 4
spambase["wk"] = 14
spambase["wb"] = 4
stamps = prepare_data_arff("Stamps", label_col=-1, drop_col=-2) # orig
stamps["ks"] = (1,15,18,77)
stamps["k"] = 15
stamps["b"] = 3
stamps["wk"] = 18
stamps["wb"] = 4
wilt = prepare_data_arff("Wilt", label_col=-1, drop_col=-2) # orig
wilt["ks"] = (1,2,4)
wilt["k"] = 1
wilt["b"] = 2
wilt["wk"] = 1
wilt["wb"] = 2


datasets = {"aloi": aloi, "glass": glass, "ionosphere": ionosphere, "kddcup99": kddcup99, "lymphography": lymphography,
             "pendigits": pendigits, "shuttle": shuttle, "waveform": waveform,
            "wbc": wbc, "wdbc": wdbc, "wpbc": wpbc}

datasets2 = {"annthyroid":annthyroid, "arrhythmia":arrhythmia, "cardiotocography":cardiotocography,
             "heartdisease":heartdisease, "hepatitis":hepatitis, "internetads":internetads, "pageblocks":pageblocks,
             "parkinson":parkinson, "pima":pima, "spambase":spambase, "stamps":stamps, "wilt":wilt}

for d,v in datasets2.items():
        print("\{data} & {n} & {outliers} & Attributes & {k} & {b} & {wk} &{wb} \\\\"
              .format(data=v["name"].lower(),outliers=int(np.sum(v["test_labels"])),
                      n=len(v["test_labels"]), k=v["k"], b=v["b"], wk=v["wk"], wb=v["wb"]))
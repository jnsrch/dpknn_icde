import sklearn.metrics

from benchmark_data import *
import numpy as np
from knn_dist import Knn, BinnedKnn
import warnings
from dimreduct import *


def hyper_select_b_k(data, weighted, bs=range(2,5), ks=range(1,3), epsilon=5, maxdepth=5, seed=4):
    train, test, test_labels = data["hyper_train"], data["hyper_test"], data["hyper_test_labels"]
    model_knn = Knn(train)
    laplace_noise_sd = 1 / epsilon

    scores = np.empty((len(ks), len(bs), len(test)))
    aurocs = np.empty((len(ks), len(bs)))

    for i, k in enumerate(ks):
        print("k= {k1} of {k2}".format(k1=k, k2=ks))
        for j, b in enumerate(bs):
            print("b= {b1} of {b2}".format(b1=b, b2=bs))
            model = BinnedKnn(bins_per_dim=b, data=train, seed=seed)

            for m, t in enumerate(test):
                scores[i,j,m] = model.query(t, k, weighted=weighted, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)

    for i, k in enumerate(ks):
        for j, b in enumerate(bs):
            aurocs[i,j] = sklearn.metrics.roc_auc_score(test_labels,scores[i,j])

    res_knn = model_knn.query(test, k=2)
    df = pd.DataFrame(data=aurocs, index=ks, columns=bs)
    return df, sklearn.metrics.roc_auc_score(test_labels,res_knn)


def bin_benchmark(data, bins_size=range(2, 4), k=5, epsilon=1., maxdepth=3, seeds=(2, 3)):
    train, test, test_labels = data["train"], data["test"], data["test_labels"]

    model_knn = Knn(train)
    res_knn = model_knn.query(test, k=k)
    res_wknn = model_knn.query(test, k=k, weighted=True)
    laplace_noise_sd = 1/epsilon

    scores = np.empty((len(bins_size), len(seeds), len(test)))
    scores_w = np.empty((len(bins_size), len(seeds), len(test)))
    scores_binw = np.empty((len(bins_size), len(test)))
    scores_bin = np.empty((len(bins_size), len(test)))

    for i, b in enumerate(bins_size):
        for s_idx, s in enumerate(seeds):
            print("b= {b1} of {b2}".format(b1=i, b2=bins_size[-1]))
            model = BinnedKnn(bins_per_dim=b, data=train, seed=s)

            for j, t in enumerate(test):
                scores[i,s_idx,j] = model.query(t, k, weighted=False, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)
                scores_w[i,s_idx,j] = model.query(t, k, weighted=True, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)

            # binned but no noise
        model = BinnedKnn(bins_per_dim=b, data=train)
        scores_binw[i] = model.batch_query(test, k=k)
        scores_bin[i] = model.batch_query(test, k=k, weighted=False)

    return dict(bins=bins_size, k=k, epsi=epsilon, dp_w_scores=scores_w, dp_scores=scores, bin_w_scores=scores_binw,
                bin_scores=scores_bin, scores=res_knn, w_scores=res_wknn, actual=test_labels)


def k_benchmark(data, do_normalize=True, bins_size=2, ks=range(3,8), epsilon=1., maxdepth=3, seeds = (1,2)):
    train, test, test_labels = data["train"], data["test"], data["test_labels"]

    if do_normalize:
        train, test = util.normalize(train, test)

    model_knn = Knn(train)
    aurocs_bknn_dp = np.zeros((len(ks), (len(seeds))))
    aurocs_bknn_w_dp =  np.zeros((len(ks), (len(seeds))))
    aurocs_bknn = np.zeros(len(ks))
    aurocs_wbknn = np.zeros(len(ks))
    auroc_knn =  np.zeros(len(ks))
    auroc_wknn = np.zeros(len(ks))

    laplace_noise_sd = 1/epsilon

    for i, k in enumerate(ks):
        res_knn = model_knn.query(test, k=k)
        res_wknn = model_knn.query(test, k=k, weighted=True)
        auroc_knn[i] = sklearn.metrics.roc_auc_score(test_labels, res_knn)
        auroc_wknn[i] = sklearn.metrics.roc_auc_score(test_labels, res_wknn)
        for s_idx, s in enumerate(seeds):
            print("k= {k1} of {k2}".format(k1=k, k2=ks[-1]))

            scores = np.empty(len(test))
            scores_w = np.empty(len(test))
            bin_model = BinnedKnn(bins_per_dim=bins_size, data=train, seed=s)
            for j, t in enumerate(test):
                scores[j] = bin_model.query(t, k, weighted=False, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)
                scores_w[j] = bin_model.query(t, k, weighted=True, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)
            # Differential private
            try:
                aurocs_bknn_dp[i, s_idx] = sklearn.metrics.roc_auc_score(test_labels, scores)
            except(ValueError):
                aurocs_bknn_dp[i, s_idx] = float("nan")
                pass
            try:
                aurocs_bknn_w_dp[i, s_idx] = sklearn.metrics.roc_auc_score(test_labels, scores_w)
            except(ValueError):
                aurocs_bknn_w_dp[i, s_idx] = float("nan")
                pass

        # binned but no noise
        bin_model = BinnedKnn(bins_per_dim=bins_size, data=train)
        aurocs_wbknn[i] = sklearn.metrics.roc_auc_score(test_labels, bin_model.batch_query(test, k=k))
        aurocs_bknn[i] = sklearn.metrics.roc_auc_score(test_labels, bin_model.batch_query(test, k=k, weighted=False))

    return dict(b=bins_size, ks=ks, epsi=epsilon, aurocs_bknn=aurocs_bknn, aurocs_wbknn=aurocs_wbknn,
                aurocs_bknn_dp=aurocs_bknn_dp, aurocs_bknn_w_dp = aurocs_bknn_w_dp, auroc_knn=auroc_knn,
                auroc_wknn=auroc_wknn)


def epsi_benchmark(data, epsi, maxdepth=3, seeds=(1,2)):
    train, test, test_labels = data["train"], data["test"], data["test_labels"]
    k = data["k"]
    b = data["b"]
    wk = data["wk"]
    wb = data["wb"]
    model_knn = Knn(train)
    res_knn = model_knn.query(test, k=k)
    res_wknn = model_knn.query(test, k=wk, weighted=True)
    scores = np.empty((len(epsi), len(seeds), len(test)))
    scores_w = np.empty((len(epsi), len(seeds), len(test)))

    for i, e in enumerate(epsi):
        for s_idx, s in enumerate(seeds):
            print("epsi= {k1} of {k2}".format(k1=i, k2=epsi))
            bin_model = BinnedKnn(bins_per_dim=b, data=train, seed=s)
            if b != wb:
                bin_model_w = BinnedKnn(bins_per_dim=wb, data=train, seed=s)
            else:
                bin_model_w = bin_model

            for j, t in enumerate(test):
                laplace_noise_sd = 1 / e
                scores[i,s_idx,j] = bin_model.query(t, k, weighted=False, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)
                scores_w[i,s_idx,j] = bin_model_w.query(t, wk, weighted=True, laplace_noise_sd=laplace_noise_sd, maxdepth=maxdepth)

    # binned but no noise
    bin_model = BinnedKnn(bins_per_dim=b, data=train)
    scores_bin = bin_model.batch_query(test, k=k)
    if wb != b:
        bin_model = BinnedKnn(bins_per_dim=wb, data=train)
    scores_binw = bin_model.batch_query(test, k=wk, weighted=False)

    return dict(b=b, k=k, wb= wb, wk = wk, epsi=epsi, dp_w_scores=scores_w, dp_scores=scores, bin_w_scores=scores_binw,
                bin_scores=scores_bin, scores=res_knn, w_scores=res_wknn, actual=test_labels
                )


def optimal_bk(data, weighted=False):
    for d,v in data.items():
        res = hyper_select_b_k(v, ks=v["ks"], weighted=weighted)
        with open("ksbs.txt", "a") as myfile:
            myfile.write("Data: {name}, weighted: {weighted}, res:\n {res}\n".format(weighted = weighted,
                                                                                     name=v["name"], res=res))


def do_benckmark(datasets):
    results = dict()
    epsi = (5, 2.5, 1.25, .6, .3, .15, .075, .035, .015)
    seeds = (2,5,8,12,20,32,52,84,100,555)
    warnings.filterwarnings("ignore")
    for d,v in datasets.items():
        dataname = v["name"]
        results[dataname] = dict()
        results[dataname]["epsi"] = epsi_benchmark(v, maxdepth=4, epsi=epsi, seeds=seeds)
        np.save("results/{dataname}.npy".format(dataname=dataname), results)

#optimal_bk(data=datasets)
#optimal_bk(data=datasets, weighted=True)
#optimal_bk(data=datasets2)
#optimal_bk(data=datasets2, weighted=True)

do_benckmark(datasets2)








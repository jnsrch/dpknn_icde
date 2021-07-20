import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from benchmark_data import *
from collections import OrderedDict


def plot_bin(data, ax, curve="auroc"):
    name = list(data.keys())[0]
    data = data[name]["bins"]
    bins_size = data["bins"]
    ax.plot(bins_size, data[curve + "s_bknn"], 'r--', label="Grid KNN")
    ax.plot(bins_size, data[curve + "s_wbknn"], 'r-', label="Grid WKNN")
    ax.errorbar(bins_size, np.nanmean(data[curve + "s_bknn_dp"], axis=1), yerr=np.nanstd(data[curve + "s_bknn_dp"], axis=1),
                                                                           fmt='b--', label="DP Grid KNN")
    jitter = .1
    ax.errorbar(np.array(list(bins_size))+jitter, np.nanmean(data[curve + "s_bknn_w_dp"], axis=1),
                yerr=np.nanstd(data[curve + "s_bknn_w_dp"], axis=1), fmt='k-',
                 label="DP Grid WKNN")
    ax.axhline(y=data[curve + "_knn"], color='g', linestyle='--', label="Baseline")
    ax.axhline(y=data[curve + "_wknn"], color='g', linestyle='-', label="Baseline (weighted)")
    ax.set_title(name, x=1.03)
    ax.text(2, 0.4, "k = {k}, ε = {epsi}".format(k=data["k"], epsi=data["epsi"]),
            bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xticks(ticks=bins_size)
    ax.set_ylabel(METRIC_LABEL)
    return ax


def summary_metric(data, actual):
    metric = np.empty((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            try:
                metric[i,j] = METRIC(actual, data[i,j])
            except(ValueError):
                metric[i] = float("nan")
                pass
    return metric


def summary_metric0(data, actual):
    metric = np.empty((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            try:
                metric[i,j] = METRIC(actual, data[i,j])
            except(ValueError):
                metric[i] = float("nan")
                pass
    return metric

def plot_epsi(data, name, ax,set_title=False):
    data = data["epsi"]
    epsi = data["epsi"]

    actual = data["actual"]
    dp_w_scores = data["dp_w_scores"]
    dp_scores = data["dp_scores"]
    bin_w_scores = data["bin_w_scores"]
    bin_scores = data["bin_scores"]
    scores = data["scores"]
    w_scores = data["w_scores"]

    ax.axhline(y=METRIC(actual, scores), color='k', linestyle='-', label="k-NN")
    ax.axhline(y=METRIC(actual, w_scores), color='k', linestyle='--', label="wk-NN")
    ax.axhline(y=METRIC(actual, bin_scores), color='darkorange', linestyle='-', label="Grid k-NN")
    ax.axhline(y=METRIC(actual, bin_w_scores), color='darkorange', linestyle='--', label="Grid wk-NN")
    ax.errorbar(epsi, np.nanmean(summary_metric(dp_scores, actual), axis=1),
                yerr=np.nanstd(summary_metric(dp_scores, actual), axis=1),
                fmt='-', color="royalblue", label="DP Grid k-NN")
    jitter = 1.1
    ax.errorbar(np.array(list(epsi)) * jitter,
                np.nanmean(summary_metric(dp_w_scores, actual), axis=1),
                yerr=np.nanstd(summary_metric(dp_w_scores, actual), axis=1), fmt='--', color="royalblue",
                label="DP Grid wk-NN")

    ax.set_xscale("log")

    #ax.text(.012, .9, "b = {bins}, k = {k}".format(bins=data["b"], k=data["k"]),
    #        bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xticks(ticks=epsi)
    ax.set_xticklabels(epsi)
    if set_title:
        ax.set_title(name)
    ax.set_ylabel(METRIC_LABEL)
    #ax.gca().invert_xaxis()
    #plt.savefig('{dataset}_epsi_k.png'.format(dataset=data["name"]))
    return ax


def plot_ks(data, ax, curve="auroc"):
    ax.plot(data["ks"], data[curve + "s_bknn"], 'r--', label="Grid k-NN")
    ax.plot(data["ks"], data[curve + "s_wbknn"], 'r-', label="Grid Wk-NN")
    ax.errorbar(data["ks"], np.nanmean(data[curve + "s_bknn_dp"], axis=1), yerr=np.nanstd(data[curve + "s_bknn_dp"], axis=1),
                fmt='b--', label="DP Grid KNN")
    jitter = .1
    ax.errorbar(np.array(list(data["ks"]))+jitter, np.nanmean(data[curve + "s_bknn_w_dp"], axis=1),
                yerr=np.nanstd(data[curve + "s_bknn_w_dp"], axis=1), fmt='k-',
                label="DP Grid WKNN")
    ax.plot(data["ks"], data[curve + "_knn"], color='g', linestyle='--', label="kNN")
    ax.plot(data["ks"], data[curve + "_wknn"], color='g', linestyle='-', label="WKNN")
    ax.set_title("b = {bins}, ε = {epsi}".format(bins=data["b"], epsi=data["epsi"]))
    ax.set_xticks(data["ks"])
    ax.set_xticklabels(data["ks"])

    return ax
    #plt.savefig('{dataset}_k.png'.format(dataset=data["name"]))

def plot_all(curve="auroc"):
    fig, axs = plt.subplots(5, 2, sharey='all', sharex='col')
    fig.set_size_inches(10.5, 20.5)

    lymph = (np.load("results/lymph.npy", allow_pickle=True).item())
    #ax = plot_ks(lymph["ks"], axs[0, 0], curve)
    ax = plot_bin(lymph, axs[0, 0], curve)
    ax = plot_epsi(lymph, axs[0, 1], curve)

    diabetes = (np.load("results/diabetes.npy", allow_pickle=True).item())
    #ax = plot_ks(diabetes["ks"], axs[1, 0], curve)
    ax = plot_bin(diabetes, axs[1, 0], curve)
    ax = plot_epsi(diabetes, axs[1, 1], curve)

    wdbc = (np.load("results/wdbc.npy", allow_pickle=True).item())
    #ax = plot_ks(wdbc["ks"], axs[2, 0], curve)
    ax = plot_bin(wdbc, axs[2, 0], curve)
    ax = plot_epsi(wdbc, axs[2, 1], curve)

    heart = (np.load("results/heart.npy", allow_pickle=True).item())
    #ax = plot_ks(heart["ks"], axs[3, 0], curve)
    ax = plot_bin(heart, axs[3, 0], curve)
    ax = plot_epsi(heart, axs[3, 1], curve)

    adult = (np.load("results/adult.npy", allow_pickle=True).item())
    #ax = plot_ks(wdbc["ks"], axs[2, 0], curve)
    ax = plot_bin(adult, axs[4, 0], curve)
    ax.set_xlabel('Grid parameter b')
    ax = plot_epsi(adult, axs[4, 1], curve)
    #ax = plot_ks(adult["ks"], axs[4, 2], curve)
    ax.set_xlabel('Privacy budget ε')

    plt.minorticks_off()
    plt.tight_layout()
    plt.subplots_adjust(left=.074, bottom=.088, right=.976, top=.96, wspace=.057, hspace=.324)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fancybox=True, loc='upper center', ncol=6)
    plt.savefig("all" + curve + ".png")

# todo. move b,k to one pane and place d as well
def plot_epsi_metric(filename):
    columns = 3
    res1 = (np.load("results/{name}.npy".format(name=filename), allow_pickle=True).item())
    res1 = OrderedDict(sorted(res1.items()))

    fig, axs = plt.subplots(len(res1), columns, sharey='all', sharex='all')
    fig.set_size_inches(15.5, 18.5)
    global METRIC
    global METRIC_LABEL

    METRIC = sklearn.metrics.average_precision_score
    METRIC_LABEL = "AP"

    ax = None
    i = 0
    for d, v in res1.items():
        name = d
        ax = plot_epsi(v, name, axs[i, 0])
        i = i + 1

    ax.set_xlabel('Privacy budget ε')

    METRIC_LABEL = "P@n"

    i = 0
    for d, v in res1.items():
        name = d
        METRIC = lambda actual, data: pan(data, actual, no_outliers(name.lower()))
        ax = plot_epsi(v, name, axs[i, 1], set_title=True)
        i= i + 1

    ax.set_xlabel('Privacy budget ε')

    METRIC = sklearn.metrics.roc_auc_score
    METRIC_LABEL = 'AUROC'
    i = 0
    for d, v in res1.items():
        name = d
        ax = plot_epsi(v, name, axs[i, 2])
        i= i + 1

    ax.set_xlabel('Privacy budget ε')

    plt.minorticks_off()
    plt.tight_layout()
    plt.subplots_adjust(left=.06, bottom=.043, right=.99, top=.956, wspace=.2, hspace=.324)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fancybox=True, loc='upper center', ncol=6)
    #plt.show()
    plt.savefig("{name}.png".format(name=filename))


def get_top_k_idx(arr, k):
    idx = np.argpartition(arr, -k)[-k:]
    return idx


def pan(scores, labels, n):
    top_score_idx = get_top_k_idx(scores, n)
    return np.sum(labels[top_score_idx])/n


def ap(labels, scores):
    ranks = np.argsort(scores)[::-1]
    outliers = np.where(labels == 1)[0]
    sum = 0
    for o in outliers:
        outlier_rank = np.where(ranks == o)[0][0] + 1
        sum = sum + pan(scores, labels, outlier_rank)
    return sum/len(outliers)


METRIC = sklearn.metrics.roc_auc_score
METRIC_LABEL = 'AUROC'

#use the last produced file
plot_epsi_metric("Wilt")


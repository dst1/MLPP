import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, \
    roc_curve, auc, roc_auc_score, \
    precision_recall_curve, average_precision_score, f1_score, \
    brier_score_loss, jaccard_similarity_score, log_loss, label_ranking_average_precision_score

from src.tools import utils


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot, top)


def single_label_legend(ax):
    """
    Removes duplicated labels in the legend, keeps first
    :param ax:
    :return:
    """
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    ax.legend(handle_list, label_list)
    return ax


def update_colors(ax, cm):
    """
    updates the colors of ax (lines and collections with a label) based on colormap cm
    :param ax:
    :param cm:
    :return: None
    """
    lines = [x for x in ax.lines if not x._label[0] == '_']
    collections = [x for x in ax.collections if not x._label[0] == '_']
    assert len(lines) == len(collections)
    colors = cm(np.linspace(0, 1, len(lines)))
    for i, c in enumerate(colors):
        lines[i].set_color(c)
        collections[i].set_color(c)


def generate_precs(y_test, probas, mean_rec=np.linspace(0, 1, 100)):
    """
    Generate precision values for the mean of CVs
    :param y_test:
    :param probas:
    :param mean_rec: the Xs where to interpolate
    :return: precs, AP
    """
    prec, rec, _ = precision_recall_curve(y_test, probas)
    precs = interp(mean_rec, rec[::-1], prec[::-1])
    precs[0] = 1.0

    AP = average_precision_score(y_test, probas)

    return precs, AP

def generate_recalls(y_test, probas, thresholds=np.linspace(0, 1, 100)):
    """
    Generate recall values for the mean of CVs
    :param y_test:
    :param probas:
    :param mean_rec: the Xs where to interpolate
    :return: precs, AP
    """
    _, rec, thresh = precision_recall_curve(y_test, probas)
    recs = interp(thresholds, np.concatenate(([0],thresh)), rec)

    return recs


def generate_tprs(y_test, probas, mean_fpr=np.linspace(0, 1, 100)):
    """
    Generate precision values for the mean of CVs
    :param y_test:
    :param probas:
    :param mean_fpr: the Xs where to interpolate
    :return: tprs, AUC
    """
    fpr, tpr, _ = roc_curve(y_test, probas)
    tprs = interp(mean_fpr, fpr, tpr)
    tprs[0] = 0.0

    AUC = roc_auc_score(y_test, probas, max_fpr=max(mean_fpr))
    return tprs, AUC


def plot_roc_cv(tprs, aucs, label, ax=None, figsize=(10, 10), title="ROC", mean_fpr=np.linspace(0, 1, 100)):
    """
    Produces ROC based on several CV splits. Thr curves are produced similarly to SKLEARN example.
    Thus the array for plotting needs to be produced beforehand!! e.g. by using the function generate tprs
    on tpr, fpr from sklearn roc curve function, aucs from each CV should be generated beforehand as well.

    :param tprs: tprs list generated as described above
    :param aucs: aucs list generated as described above
    :param label: label for the legend (and for coloring)
    :param ax: axes to plot on
    :param figsize: tuple, width, height
    :param title: title for the plot
    :return: matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    mean_tpr = np.mean(tprs, axis=0)
    if max(mean_fpr)>0.5:
        mean_tpr[-1] = 1.0

    # max_fpr = max(mean_fpr)
    # min_area = 0.5 * max_fpr ** 2
    # max_area = max_fpr
    # partial_auc = auc(mean_fpr, mean_tpr)
    # mean_auc =  0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    if label is None:
        label = r'ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    else:
        label = r'%s (%0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc)

    ax.plot(mean_fpr, mean_tpr, label=label,
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, label=label, alpha=.15)

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([min(mean_fpr), max(mean_fpr)])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax = single_label_legend(ax)
    return ax, mean_auc

def plot_recall_cv(recalls, label, ax=None, figsize=(10,10), title="Recall", kargs_plot={}):
    """
        Produces PR based on several CV splits. Thr curves are produced similarly to SKLEARN ROC example.
        Thus the array for plotting needs to be produced beforehand!! e.g. by using the function generate precs
        on rec, prec from sklearn prec rec curve function, aps from each CV should be generated beforehand as well.

        :param precs: precs list generated as described above
        :param aps: aps list generated as described above
        :param label: label for the legend (and for coloring)
        :param ax: axes to plot on
        :param figsize: tuple, width, height
        :param title: title for the plot
        :param kargs: passed to plot function
        :return: matplotlib axes
        """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    mean_thresh = np.linspace(0, 1, 100)
    mean_rec = np.mean(recalls, axis=0)

    if label is None:
        label = 'Recall'
    else:
        label = r'%s Recall' % (label)

    ax.plot(mean_thresh, mean_rec, label=label,
            lw=2, alpha=.8, **kargs_plot)

    std_rec = np.std(recalls, axis=0)
    recs_upper = np.minimum(mean_rec + std_rec, 1)
    recs_lower = np.maximum(mean_rec - std_rec, 0)
    ax.fill_between(mean_thresh, recs_lower, recs_upper, label=label, alpha=.15, **kargs_plot)

    ax.set_xlim([1.0, 0.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Threshold (inverted)')
    ax.set_ylabel('Recall')
    ax.set_title(title)
    ax = single_label_legend(ax)
    return ax

def plot_pr_cv(precs, aps, label, ax=None, figsize=(10, 10), title="ROC"):
    """
    Produces PR based on several CV splits. Thr curves are produced similarly to SKLEARN ROC example.
    Thus the array for plotting needs to be produced beforehand!! e.g. by using the function generate precs
    on rec, prec from sklearn prec rec curve function, aps from each CV should be generated beforehand as well.

    :param precs: precs list generated as described above
    :param aps: aps list generated as described above
    :param label: label for the legend (and for coloring)
    :param ax: axes to plot on
    :param figsize: tuple, width, height
    :param title: title for the plot
    :return: matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    mean_rec = np.linspace(0, 1, 100)
    mean_prec = np.mean(precs, axis=0)

    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    if label is None:
        label = r'PR (%0.2f $\pm$ %0.2f)' % (mean_ap, std_ap)
    else:
        label = r'%s (%0.2f $\pm$ %0.2f)' % (label, mean_ap, std_ap)

    ax.plot(mean_rec, mean_prec, label=label,
            lw=2, alpha=.8)

    std_prec = np.std(precs, axis=0)
    precs_upper = np.minimum(mean_prec + std_prec, 1)
    precs_lower = np.maximum(mean_prec - std_prec, 0)
    ax.fill_between(mean_rec, precs_lower, precs_upper, label=label, alpha=.15)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax = single_label_legend(ax)
    return ax, mean_ap


def plot_roc_single(y_test, probas, target, title="ROC", label=None,
                    ax=None, lw=2, figsize=(10, 10), max_fpr = 1.0):
    """

    :param y_test:
    :param probas:
    :param target:
    :param title:
    :param label:
    :param ax:
    :param lw:
    :param figsize:
    :return:
    """

    if type(y_test) == pd.core.frame.DataFrame:
        y_test = y_test.values

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_auc = roc_auc_score(y_test, probas, max_fpr=max_fpr)
    if label is None:
        ax.plot(fpr, tpr, lw=lw,
                label='{0} (AUC = {1:0.3f})'.format(target, roc_auc))
    else:
        ax.plot(fpr, tpr, lw=lw, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, max_fpr])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    return ax


def plot_PR_single(y_test, probas, target, label=None,
                   title="Precision - Recall",
                   ax=None, lw=2, figsize=(10, 10)):
    """

    :param y_test:
    :param probas:
    :param target:
    :param label:
    :param title:
    :param ax:
    :param lw:
    :param figsize:
    :return:
    """
    if type(y_test) == pd.core.frame.DataFrame:
        y_test = y_test.values

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    prec, rec, _ = precision_recall_curve(y_test, probas)
    AP = average_precision_score(y_test, probas)
    if label is None:
        ax.plot(rec, prec, lw=lw, label='{0} (AP = {1:0.2f})'.format(target, AP))
    else:
        ax.plot(rec, prec, lw=lw, label=label)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    ax.legend()
    return ax


def plot_roc_multilab(y_test, probas, classes,
                      title="ROC",
                      plot_macro=False, plot_micro=True,
                      ax=None,
                      lw=2, figsize=(10, 10)):
    """
    Draws ROC curve for multiclass or multilabel classification

    :param y_test: real labels
    :param probas: probabilities from predict_proba
    :param classes: name of the classes, ordered like probas columns
    :param title: title for the figure (default = "ROC")
    :param plot_macro: bool, whether to draw macro-average (default = F)
    :param plot_micro: bool, whether to draw micro-average (default = T)
    :param ax: matplotlib axes to draw on
    :param lw: linewidth for plots (default = 2)
    :param figsize: size of the figure (default = (10,10))
    :return: axes object
    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if type(y_test) == pd.core.frame.DataFrame:
        y_test = y_test.values

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    for i, c in enumerate(classes):
        fpr[c], tpr[c], _ = roc_curve(y_test[:, i], probas[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])
        ax.plot(fpr[c], tpr[c], lw=lw,
                label='{0} (AUC = {1:0.2f})'.format(c, roc_auc[c]))

    # micro average
    if plot_micro:
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), probas.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        ax.plot(fpr["micro"], tpr["micro"],
                label='micro-average (AUC = {0:0.2f})'.format(
                    roc_auc["micro"]),
                linewidth=lw, color='black', linestyle="--")

    # macro average
    if plot_macro:
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for c in classes:
            mean_tpr += interp(all_fpr, fpr[c], tpr[c])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average (AUC = {0:0.2f})'.format(
                     roc_auc["macro"]),
                 linewidth=lw, color='#777777', linestyle="--")

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def plot_PR_multilab(y_test, probas, classes,
                     title="Precision - Recall",
                     plot_micro=True,
                     ax=None,
                     lw=2, figsize=(10, 10)):
    """
    Draws PR curve for multiclass or multilabel classification

    :param y_test: real labels
    :param probas: probabilities from predict_proba
    :param classes: name of the classes, ordered like probas columns
    :param title: title for the figure (default = "ROC")
    :param plot_micro: bool, whether to draw micro-average (default = T)
    :param ax: matplotlib axes to draw on
    :param lw: linewidth for plots (default = 2)
    :param figsize: size of the figure (default = (10,10))
    :return: axes object
    """
    prec = dict()
    rec = dict()
    AP = dict()

    if type(y_test) == pd.core.frame.DataFrame:
        y_test = y_test.values

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)

    for i, c in enumerate(classes):
        prec[c], rec[c], _ = precision_recall_curve(y_test[:, i], probas[:, i])
        AP[c] = average_precision_score(y_test[:, i], probas[:, i])
        ax.plot(rec[c], prec[c],
                lw=lw,
                label='{0} (AP = {1:0.2f})'.format(c, AP[c]))
    # micro average
    if plot_micro:
        prec["micro"], rec["micro"], _ = precision_recall_curve(
            y_test.ravel(), probas.ravel())
        AP["micro"] = average_precision_score(y_test, probas, average="micro")
        ax.plot(rec["micro"], prec["micro"],
                label='micro-average (AP = {0:0.2f})'.format(AP["micro"]),
                linewidth=lw, color='black', linestyle="--")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(title)
    ax.legend(loc="lower left")
    return ax


def plot_learning_curve(eval_dict, metric, val_names=["Train", "Test"], title="Learning Curve",
                        figsize=(10, 10), lw=2):
    """
    Plots the learning curves based on XGBOOST eval results
    :param eval_dict: evaluation results from XGBOOST
    :param metric: str, the metric tested
    :param val_names: list, names of validation sets
    :param title: Figure title for each - {Title} - {Metric}
    :param figsize: size of the figure to return
    :param lw:
    :return:
    """
    data = {}
    for i, name in enumerate(val_names):
        data[val_names[i]] = eval_dict['validation_{}'.format(i)][metric]
    x = None

    fig = plt.figure(figsize=figsize)

    plt.title(title)
    for k, v in data.items():
        if x is None:
            x = np.arange(len(v))
        plt.plot(x, v, lw=lw, label="{}".format(k))
    plt.xlabel('Estimators')
    plt.ylabel(metric)
    plt.legend(loc="best")

    return fig


def plot_calibration_curve(probas, y_test, title='Calibration plots  (reliability curve)', figsize=(8, 8), bins=20):
    """
    Draws a calibration curve for the estimator
    :param probas: ndarray of probabilities per label
    :param y_test: real binarized labels
    :param title: title of the plots
    :param figsize: figure size
    :param bins: number of bins to use for histogram, default 20
    :return: fig object
    """

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:")
    for i, c in enumerate(y_test.columns):
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test.iloc[:, i], probas[:, i], n_bins=10)
        clf_score = brier_score_loss(
            y_test.iloc[:, i], probas[:, i], pos_label=1)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.2f)" % (c, clf_score))

        ax2.hist(probas[:, i], range=(0, 1), bins=bins, label=c,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(title)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    return fig


def stats_report(pred, probas, y_test, targets, f):
    """
    Writes a status report to file
    :param pred:
    :param probas:
    :param y_test:
    :param targets:
    :param f:
    :return:
    """
    if y_test.shape[1] > 1:
        utils.print_write(classification_report(y_test, pred), f)

    res = pd.DataFrame({"ROC AUC": roc_auc_score(y_test, probas, average=None),
                        "F1": f1_score(y_test, pred, average=None),
                        "Avg. Precision": average_precision_score(y_test, probas, average=None),
                        },
                       index=targets).sort_values(by="F1", ascending=False)

    utils.print_write(res.to_string(), f)
    utils.print_write("\nJaccard sim. {:.3f}\nLogloss: {:.3f}\nlabel ranking AP: {:.3f}".format(
        jaccard_similarity_score(y_test, pred),
        log_loss(
            y_test, probas),
        label_ranking_average_precision_score(
            y_test, probas)), f)

if __name__ == "__main__":
    recalls = [0]*4
    tprs=[0]*4
    aucs=[0]*4
    precs=[0]*4
    aps=[0]*4

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold

    X,y = make_classification()
    kf = KFold(n_splits=4)
    for i,(train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)[:,1]

        prec, rec, thresh = precision_recall_curve(y_test, probas)
        fig_single, ax_single = plt.subplots()
        ax_single.plot(thresh,rec[:-1])
        fig_single.canvas.draw()
        plt.show()

        recalls[i] = generate_recalls(y_test, probas)
        tprs[i], aucs[i] = generate_tprs(y_test, probas)
        precs[i], aps[i] = generate_precs(y_test, probas)

    plot_recall_cv(recalls, "", title="RECALL")
    plt.show()

    plot_pr_cv(precs, aps, "", title="PR")
    plt.show()

    plot_roc_cv(tprs,aucs, "", title="ROC")
    plt.show()

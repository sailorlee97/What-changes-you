import itertools
import random
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from numpy import interp
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          save_name='confusionmatrix',
                          normalize=True):
    """
    :param cm:混淆矩阵
    :param target_names:每个类的名称
    :param title: 图片名称
    :param cmap:
    :param normalize:
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    mpl.use('Agg')  # !IMPORTANT

    plt.figure(figsize=(15, 12))  # (15,12)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45,fontsize=20)
        plt.yticks(tick_marks, target_names,fontsize=20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=20)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./{}.png'.format(save_name),dpi=500)  # dpi分辨率
    # plt.show()


def plot_conf(y_pre, y_val, labels,name):
    """
    绘制混淆矩阵
    :param y_pre:预测标签
    :param y_val: 真实标签
    :param labels: 标签名称
    :param label_value: 映射的标签值
    :return:
    """
    conf_mat = confusion_matrix(y_true=y_val, y_pred=y_pre)
    print(conf_mat)
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix',save_name=name)

def multi_roc(y_label, y_score,labels):
    """
      将多分类的roc图输出到一张图上

      Args:
          y_label: 多个分类的真实标签
          y_score: 多个分类的预测标签
          labels: 多个分类的名称

      """
    # 计算每一类的ROC
    n_classes=len(y_label[0])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    print(y_label.ravel())
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(dpi=500)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})' 
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = list(mcolors.CSS4_COLORS.keys())  # 颜色变化

    for i in range(n_classes):
        colors_i = random.randint(1,148)
        plt.plot(fpr[i], tpr[i], color=mcolors.CSS4_COLORS[colors[colors_i]], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], '--', lw=1, color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('multi-calss ROC', fontsize=25)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('multi_class_roc.png')
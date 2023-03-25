from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from util.Config import Config

import math

import matplotlib.pyplot as plt

device = Config.device


class Evaluation:
    def __init__(self, X, y, model_name):
        self.X = X
        self.y_true = y
        self.k = 0.2
        self.y_pred = torch.softmax(self.X, dim=1).argmax(dim=1).cpu().detach().numpy()
        self.y_pred_prob = self.X.cpu().detach().numpy()[:, 1]
        self.model_name = model_name

    @staticmethod
    def plot_roc_4_three_model(result_list):
        idx = 0
        color_list = ["#2ed573", "#ff4757", "#3742fa"]
        for result in result_list:
            X = result[0]
            y_true = result[1]
            y_pred_prob = X.cpu().detach().numpy()[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            AUC = auc(fpr, tpr)
            AUC = round(AUC, 2)

            if idx == 0:
                label = "bjCnet_pt (area = %0.2f)" % AUC
            elif idx == 1:
                label = "bjCnet_ft (area = %0.2f)" % AUC
            elif idx == 2:
                label = "non_ContNet (area = %0.2f)" % AUC

            plt.plot(
                fpr,
                tpr,
                color=color_list[idx],
                lw=2,
                label=label
            )
            idx = idx + 1

        plt.plot([0, 1], [0, 1], color="#34495e", lw=2, linestyle="--")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curves")
        plt.legend(loc="lower right")
        plt.savefig(Config.roc_path.format("all"))
        plt.close()

    def evaluate_4_nn(self):
        balanced_acc, cls_report = self.__get_eval_indicies()
        recall_top_k = self.__get_recall_top_k()
        mcc = self.__get_mcc()
        effort_top_k = self.__get_effort_top_k()

        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_prob)
        AUC = auc(fpr, tpr)
        AUC = round(AUC, 2)

        print(
            "AUC: {}\nbalanced_acc: {}\nrecall@top20: {}\neffort@recall20: {}\nMCC: {}\nclassification report\n{}".format(
                AUC,
                balanced_acc,
                recall_top_k,
                effort_top_k,
                mcc,
                cls_report))

        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % AUC,
        )

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.savefig(Config.roc_path.format(self.model_name))
        plt.close()

    def __get_mcc(self):
        cfm = confusion_matrix(self.y_true, self.y_pred)
        TN = cfm[0, 0]
        FP = cfm[0, 1]
        FN = cfm[1, 0]
        TP = cfm[1, 1]

        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = round(mcc, 4)
        return mcc

    def __get_eval_indicies(self):
        balanced_acc = balanced_accuracy_score(self.y_true, self.y_pred)

        balanced_acc = round(balanced_acc, 4)

        cls_report = classification_report(self.y_true, self.y_pred)

        return balanced_acc, cls_report

    def __get_effort_top_k(self):
        X = torch.index_select(self.X, 1, torch.LongTensor([1]).to(device)).squeeze(1)
        X = torch.softmax(X, dim=0)
        X = torch.sort(X, descending=True)
        y = self.y_true.cpu().detach().numpy().tolist()

        y_1 = pd.DataFrame(y)
        y_1 = y_1[y_1[0] == 1]
        y_1_cnt = y_1.shape[0]

        indices = X.indices.cpu().detach().numpy().tolist()

        line = 0
        cnt = 0
        for index in indices:
            if cnt == int(y_1_cnt * self.k):
                break

            if y[index] == 1:
                cnt = cnt + 1

            line = line + 1

        effort_top_k = round(line / len(indices), 4)

        return effort_top_k

    def __get_recall_top_k(self):
        X = torch.index_select(self.X, 1, torch.LongTensor([1]).to(device)).squeeze(1)
        X = torch.softmax(X, dim=0)
        X = torch.sort(X, descending=True)
        y = self.y_true.cpu().detach().numpy().tolist()

        indices = X.indices.cpu().detach().numpy().tolist()

        limit = 0
        cnt = 0
        for index in indices:
            if limit == int(len(indices) * self.k):
                break

            if y[index] == 1:
                cnt = cnt + 1

            limit = limit + 1

        recall_top_k = round(cnt / int(len(indices) * 0.2), 2)

        return recall_top_k

    def plot_roc(self, X, y):
        cv = StratifiedKFold(n_splits=10)
        # classifier = RandomForestClassifier()
        # classifier = DecisionTreeClassifier(criterion='entropy')
        classifier = KNeighborsClassifier()

        y = pd.DataFrame(y).values.ravel()

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()

        with tqdm(total=10, desc='(T)') as pbar:
            classification_result = []
            for i, (train, test) in enumerate(cv.split(X, y)):
                classifier.fit(X[train], y[train])
                viz = RocCurveDisplay.from_estimator(
                    classifier,
                    X[test],
                    y[test],
                    name="ROC fold {}".format(i),
                    alpha=0.3,
                    lw=1,
                    ax=ax,
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

                y_pred = classifier.predict(X[test])

                pbar.update()

        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="RoC with 10cv",
        )
        ax.legend(loc="lower right")
        plt.grid()
        plt.savefig(Config.roc_path)
        plt.show()

    @staticmethod
    def plot_loss(epoch_loss_list, model_name):

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('epoch_idx')
        plt.ylabel('loss')

        plt.grid()

        epoch_list = [e for e in range(len(epoch_loss_list))]

        plt.plot(epoch_list, epoch_loss_list, linewidth=1, linestyle="solid", label="train loss", color="#9ebc19")
        plt.legend()
        plt.title('Loss curve')

        plt.savefig(Config.loss_path.format(model_name))
        plt.show()
        plt.clf()

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import plotly
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def plot_missing_data(df, figsize=(20,20)):
    sns.set(rc = {'figure.figsize': figsize})
    sns.heatmap(
        data = np.where(df.isnull() == True,1,0),
        xticklabels= df.columns)


def plot_confusion_matrix(y_true, y_pred, labels):
    plt.figure(figsize=(12,10))
    cm_train = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    ax = sns.heatmap(
        cm_train, 
        cmap="YlGnBu",
        fmt="d",
        annot=True, 
        xticklabels=labels,
        yticklabels=labels)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix')

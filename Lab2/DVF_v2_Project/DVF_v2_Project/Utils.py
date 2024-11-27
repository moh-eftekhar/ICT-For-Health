from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def Accuracy(acts, pred):
    accuracy = accuracy_score(acts, pred)
    return accuracy


def Confusion_matrix(acts, pred):
    conf_matrix = confusion_matrix(acts, pred)
    return conf_matrix

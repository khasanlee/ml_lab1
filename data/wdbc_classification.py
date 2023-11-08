import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from matplotlib.lines import Line2D  # For the custom legend

def load_wdbc_data(filename):
    class WDBCData:
        data = []  # Shape: (569, 30)
        target = []  # Shape: (569, )
        target_names = ['malignant', 'benign']
        feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness',
                         'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
                         'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
                         'compactness error', 'concavity error', 'concave points error', 'symmetry error',
                         'fractal dimension error',
                         'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
                         'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
                         'worst fractal dimension']

    wdbc = WDBCData()
    with open(filename) as f:
        for line in f.readlines():
            items = line.split(',')
            if items[1] == 'M':
                new_target = 0
            else:
                new_target = 1
            wdbc.target.append(new_target)  # Add the true label (0 for M / 1 for others)
            wdbc.data.append([float(i) for i in items[2:]])  # Add 30 attributes (as floating-point numbers)
    wdbc.data = np.array(wdbc.data)
    return wdbc

if __name__ == '__main__':
    # Load a dataset
    wdbc = load_wdbc_data('data/wdbc.data')

    # Train a model
    model = svm.SVC()  
    model.fit(wdbc.data, wdbc.target)

    # Test the model
    predict = model.predict(wdbc.data)
    accuracy = metrics.balanced_accuracy_score(wdbc.target, predict)

    # Visualize the confusion matrix
    cm = metrics.confusion_matrix(wdbc.target, predict)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(wdbc.target_names))
    plt.xticks(tick_marks, wdbc.target_names, rotation=45)
    plt.yticks(tick_marks, wdbc.target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Visualize testing results
    cmap = np.array([(1, 0, 0), (0, 1, 0)])
    clabel = [Line2D([0], [0], marker='o', lw=0, label=wdbc.target_names[i], color=cmap[i]) for i in range(len(cmap))]

    for (x, y) in [(0, 1), (2, 3)]:  
        plt.figure()
        plt.title(f'My Classifier (Accuracy: {accuracy:.3f})')
        plt.scatter(wdbc.data[:, x], wdbc.data[:, y], c=cmap[wdbc.target], edgecolors=cmap[predict])
        plt.xlabel(wdbc.feature_names[x])
        plt.ylabel(wdbc.feature_names[y])
        plt.legend(handles=clabel, framealpha=0.5)

    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot3D(sample, titles=lambda i:str(i), title="", fileName=None,
           size_factor_x=1, size_factor_y=1):
    """
    Display images in a dynamically arranged grid with individual titles and an overall title.:

    Args:
        sample (Iterable): Images to plot (2D arrays).
        titles (list of str, optional): Titles per subplot, length == len(sample).
        title (str, optional): Supertitle for full figure.
        fileName (str, optional): Path to save figure.
        size_factor_x (float, optional): Width size.
        size_factor_y (float, optional): Height size.
    """
    plt.figure()
    n = len(sample)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * size_factor_x,
                                      rows * size_factor_y),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            img = sample[i]
            h, w = img.shape[:2]
            ax.imshow(img, cmap='plasma', aspect='auto', interpolation='nearest')
            ax.axis('off')
            if titles:
                ax.set_title(titles(i), fontsize=8)
        else:
            fig.delaxes(ax)
    if title:
        fig.suptitle(title)
    if fileName:
        fig.savefig(fileName, dpi=300, bbox_inches='tight')

def plot_loss_acc(history, title="", fileNameBase=None):
    addToTtile = ""
    if title:
        addToTtile = " - "+title
    plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss"+addToTtile)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if fileNameBase:
        plt.savefig(fileNameBase+"_loss", dpi=300, bbox_inches='tight')

    plt.figure()
    plt.plot(history['epoch'], history['train_acc'], label='Training Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Validation Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy"+addToTtile)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if fileNameBase:
        plt.savefig(fileNameBase+"_acc", dpi=300, bbox_inches='tight')

def plot_confusion_matrix(y_pred, y_true, labels, title="",fileName=None, accFileName=None):
    plt.figure()
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    if fileName:
        plt.savefig(fileName, dpi=300, bbox_inches='tight')

def write_test_acc(test_acc, filename):
    with open(filename, 'w') as f:
        f.write(f"{test_acc}")

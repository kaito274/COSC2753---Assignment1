import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(train_Y, pred_train_Y, val_Y=None, pred_val_Y=None):
    # Compute confusion matrix for the training set
    cm_train = confusion_matrix(train_Y, pred_train_Y)
    
    # Initialize the plot
    plt.figure(figsize=(16, 6))
    
    # Plot the training set confusion matrix
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.heatmap(cm_train, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', cbar=False)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Training Set Confusion Matrix')
    
    # Check if validation data is provided
    if val_Y is not None and pred_val_Y is not None:
        cm_val = confusion_matrix(val_Y, pred_val_Y)
        # Plot the validation set confusion matrix
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        sns.heatmap(cm_val, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', cbar=False)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Validation Set Confusion Matrix')
    
    # Adjust layout and show plot
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()

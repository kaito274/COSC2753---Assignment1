import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


def threshold_tuning_with_rocauc(train_probabilities, val_probabilities, train_Y, val_Y, thresholds = None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, num=100)  # 100 thresholds spaced evenly from 0 to 1
    train_rocauc_scores = []
    val_rocauc_scores = []
    for thresh in thresholds:
        # Apply threshold to positive class probabilities to create binary predictions
        train_preds = (train_probabilities >= thresh).astype(int)
        val_preds = (val_probabilities >= thresh).astype(int)
        
        # Calculate the F1 score at this threshold for both sets
        train_auc = roc_auc_score(train_Y, train_preds)
        val_auc = roc_auc_score(val_Y, val_preds)
        
        train_rocauc_scores.append(train_auc)
        val_rocauc_scores.append(val_auc)
        
    return train_rocauc_scores, val_rocauc_scores


def threshold_tuning_with_f1(train_probabilities, val_probabilities, train_Y, val_Y, thresholds = None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, num=100)  # 100 thresholds spaced evenly from 0 to 1
    train_f1_scores = []
    val_f1_scores = []
    for thresh in thresholds:
        # Apply threshold to positive class probabilities to create binary predictions
        train_preds = (train_probabilities >= thresh).astype(int)
        val_preds = (val_probabilities >= thresh).astype(int)
        
        # Calculate the F1 score at this threshold for both sets
        train_f1 = f1_score(train_Y, train_preds)
        val_f1 = f1_score(val_Y, val_preds)
        
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

    f1_diff = np.abs(np.array(train_f1_scores) - np.array(val_f1_scores))
    combined_score = np.array(val_f1_scores) - f1_diff  # Example metric to balance both
    optimal_index = np.argmax(combined_score)
    optimal_threshold = thresholds[optimal_index]
    
    print(f"Optimal threshold: {optimal_threshold}")
        
    return train_f1_scores, val_f1_scores, optimal_threshold

def threshold_tuning_with_f1_weighted(train_probabilities, val_probabilities, train_Y, val_Y, thresholds = None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, num=100)  # 100 thresholds spaced evenly from 0 to 1
    train_f1_scores = []
    val_f1_scores = []
    for thresh in thresholds:
        # Apply threshold to positive class probabilities to create binary predictions
        train_preds = (train_probabilities >= thresh).astype(int)
        val_preds = (val_probabilities >= thresh).astype(int)
        
        # Calculate the F1 score at this threshold for both sets
        train_f1 = f1_score(train_Y, train_preds, average='binary')
        val_f1 = f1_score(val_Y, val_preds, average='weighted')
        
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

    f1_diff = np.abs(np.array(train_f1_scores) - np.array(val_f1_scores))
    combined_score = np.array(val_f1_scores) - f1_diff  # Example metric to balance both
    optimal_index = np.argmax(combined_score)
    optimal_threshold = thresholds[optimal_index]
    
    print(f"Optimal threshold: {optimal_threshold}")
        
    return train_f1_scores, val_f1_scores, optimal_threshold

def create_plot_pivot(data2, x_column):
    """ Create a pivot table for satisfaction versus another rating for easy plotting. """
    _df_plot = data2.groupby([x_column, 'Status']).size() \
    .reset_index().pivot(columns='Status', index=x_column, values=0)
    return _df_plot

def get_performance(train_Y, train_pred_Y, train_predprob_Y, val_Y, val_pred_Y, val_predprob_Y):
    # Calculating metrics
    metrics_data = {
        'F1 Score (Binary)': [
            f1_score(train_Y, train_pred_Y, average='binary'),
            f1_score(val_Y, val_pred_Y, average='binary')
        ],
        'F1 Score (Weighted)': [
            None,  # Not applicable for train set
            f1_score(val_Y, val_pred_Y, average='weighted')
        ],
        'ROC-AUC Score': [
            roc_auc_score(train_Y, train_predprob_Y),
            roc_auc_score(val_Y, val_predprob_Y)
        ],
    }
    
    # Creating DataFrame
    metrics_df = pd.DataFrame(metrics_data, index=['Train Set', 'Validation Set'])
    
    # Displaying the table
    print(metrics_df)

def plot_tuning_with_optimal_threshold(label_train, label_val, title, y_label, train_scores, val_scores, optimal_threshold, thresholds = None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, num=100)  # 100 thresholds spaced evenly from 0 to 1
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, train_scores, label=label_train)
    plt.plot(thresholds, val_scores, label=label_val)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.title(title)
    plt.xlabel('Threshold')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_metrics(models, train_X, train_Y, val_X, val_Y):
    # Create an empty dictionary to store the scores
    scores = {}
    print(f"{'Model':22}{'F1 (train sets)':>18}{'F1 (validation sets)':>22}{'ROC_AUC (train sets)':>22}{'ROC_AUC (validation sets)':>28}")
    # Loop over the models
    for clf, label in models:
        # Make a copy of the data to avoid modifying the original
        train_X_temp = train_X.copy()
        train_Y_temp = train_Y.copy()
        val_X_temp = val_X.copy()
        val_Y_temp = val_Y.copy()

        if label in ['logistic2', 'logistic3']:
            # Degree of polynomial features based on the model name
            degree = int(label[-1])  # Extract the degree from the model label
            
            # Polynomial Features
            poly = PolynomialFeatures(degree)
            train_X_temp = poly.fit_transform(train_X_temp)
            val_X_temp = poly.transform(val_X_temp)
            
            # Scale the data after transforming it to polynomial features
            scaler = StandardScaler()
            train_X_temp = scaler.fit_transform(train_X_temp)
            val_X_temp = scaler.transform(val_X_temp)
        
        # Predictions and probabilities for train sets
        train_preds = clf.predict(train_X_temp)
        train_probs = clf.predict_proba(train_X_temp)[:, 1]
        
        # Predictions and probabilities for validation sets
        val_preds = clf.predict(val_X_temp)
        val_probs = clf.predict_proba(val_X_temp)[:, 1]
        
        # F1 and ROC-AUC for train sets
        train_f1 = f1_score(train_Y_temp, train_preds, average='binary')
        train_roc_auc = roc_auc_score(train_Y_temp, train_probs)
        
        # F1 and ROC-AUC for validation sets
        val_f1 = f1_score(val_Y_temp, val_preds, average='binary')
        val_roc_auc = roc_auc_score(val_Y_temp, val_probs)
        
        # Store the scores in the dictionary
        scores[label] = {
            'F1 (train sets)': train_f1,
            'F1 (validation sets)': val_f1,
            'ROC-AUC (train sets)': train_roc_auc,
            'ROC-AUC (validation sets)': val_roc_auc
        }
        print(f"{label:18}{train_f1:>18.6f}{val_f1:>22.6f}{train_roc_auc:>20.6f}{val_roc_auc:>25.6f}")
    
    return scores

def calculate_metrics_after_tuning_threshold(optimal_threshold_list, train_X, train_Y, val_X, val_Y):
    # Create an empty dictionary to store the scores
    scores = {}
    print(f"{'Model':18}{'F1 (train sets)':>15}{'F1 (validation sets)':>22}{'ROC_AUC (train sets)':>22}{'ROC_AUC (validation sets)':>27}", end = ' ' )
    print(f"{'Optimal threshold':30}")
    # Loop over the models
    for clf, optimal_threshold, label in optimal_threshold_list:
        # Make a copy of the data to avoid modifying the original
        train_X_temp = train_X.copy()
        train_Y_temp = train_Y.copy()
        val_X_temp = val_X.copy()
        val_Y_temp = val_Y.copy()

        if label in ['logistic2', 'logistic3']:
            # Degree of polynomial features based on the model name
            degree = int(label[-1])  # Extract the degree from the model label
            
            # Polynomial Features
            poly = PolynomialFeatures(degree)
            train_X_temp = poly.fit_transform(train_X_temp)
            val_X_temp = poly.transform(val_X_temp)
            
            # Scale the data after transforming it to polynomial features
            scaler = StandardScaler()
            train_X_temp = scaler.fit_transform(train_X_temp)
            val_X_temp = scaler.transform(val_X_temp)
        
        # Predictions and probabilities for train sets
        train_probs = clf.predict_proba(train_X_temp)[:, 1]
        train_preds = (train_probs >= optimal_threshold).astype(int)

        
        # Predictions and probabilities for validation sets
        val_probs = clf.predict_proba(val_X_temp)[:, 1]
        val_preds = (val_probs >= optimal_threshold).astype(int)
        
        # F1 and ROC-AUC for train sets
        train_f1 = f1_score(train_Y_temp, train_preds, average='binary')
        train_roc_auc = roc_auc_score(train_Y_temp, train_probs)
        
        # F1 and ROC-AUC for validation sets
        val_f1 = f1_score(val_Y_temp, val_preds, average='binary')
        val_roc_auc = roc_auc_score(val_Y_temp, val_probs)
        
        # Store the scores in the dictionary
        scores[label] = {
            'F1 (train sets)': train_f1,
            'F1 (validation sets)': val_f1,
            'ROC-AUC (train sets)': train_roc_auc,
            'ROC-AUC (validation sets)': val_roc_auc
        }
        print(f"{label:15}{train_f1:>15.6f}{val_f1:>22.6f}{train_roc_auc:>20.6f}{val_roc_auc:>22.6f}{optimal_threshold:25.6f}")
    
    return scores
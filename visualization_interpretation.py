from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

train_cm = confusion_matrix(y_train_tensor.numpy(), train_predictions.numpy())
test_cm = confusion_matrix(y_test_tensor.numpy(), test_predictions.numpy())


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

print("Training Confusion Matrix:")
plot_confusion_matrix(train_cm, "Training Confusion Matrix")

print("Testing Confusion Matrix:")
plot_confusion_matrix(test_cm, "Testing Confusion Matrix")


print("Training Classification Report:")
print(classification_report(y_train_tensor.numpy(), train_predictions.numpy()))

print("Testing Classification Report:")
print(classification_report(y_test_tensor.numpy(), test_predictions.numpy()))

train_probs = train_outputs.numpy()
test_probs = test_outputs.numpy()

train_fpr, train_tpr, _ = roc_curve(y_train_tensor.numpy(), train_probs)
train_auc = auc(train_fpr, train_tpr)

test_fpr, test_tpr, _ = roc_curve(y_test_tensor.numpy(), test_probs)
test_auc = auc(test_fpr, test_tpr)

plt.figure(figsize=(10, 6))
plt.plot(train_fpr, train_tpr, label=f'Training AUC = {train_auc:.2f}')
plt.plot(test_fpr, test_tpr, label=f'Testing AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')  # Reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
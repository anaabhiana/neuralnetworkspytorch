import torch
model_file_name = 'logistic_regression_model.pth'

torch.save(model.state_dict(), model_file_name)
print(f"Model saved as {model_file_name}")

loaded_model = LogisticRegressionModel(input_dim=X_train_tensor.shape[1])

loaded_model.load_state_dict(torch.load(model_file_name))
print(f"Model loaded from {model_file_name}")
loaded_model.eval()
with torch.no_grad():
    test_outputs_loaded = loaded_model(X_test_tensor)
    test_predictions_loaded = (test_outputs_loaded >= 0.5).float()  
    loaded_test_accuracy = (test_predictions_loaded.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item() * 100

print(f"Test Accuracy of the Loaded Model: {loaded_test_accuracy:.2f}%")
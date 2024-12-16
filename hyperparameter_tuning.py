import torch.optim as optim

learning_rates = [0.01, 0.05, 0.1]
num_epochs = 100

test_accuracies = {}

for lr in learning_rates:
    print(f"Training model with learning rate: {lr}")
    model = LogisticRegressionModel(input_dim=X_train_tensor.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs >= 0.5).float() 
        test_accuracy = (test_predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item() * 100
        test_accuracies[lr] = test_accuracy

    print(f"Test Accuracy for learning rate {lr}: {test_accuracy:.2f}%\n")
best_lr = max(test_accuracies, key=test_accuracies.get)
print(f"Best Learning Rate: {best_lr}")
print(f"Test Accuracy with Best Learning Rate: {test_accuracies[best_lr]:.2f}%")

num_epochs = 1000
for epoch in range(1, num_epochs + 1):

    model.train()

    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()


    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")


model.eval() 


with torch.no_grad():

    train_outputs = model(X_train_tensor)
    train_predictions = (train_outputs >= 0.5).float()  # Threshold at 0.5
    train_accuracy = (train_predictions.eq(y_train_tensor).sum() / y_train_tensor.shape[0]).item() * 100
    

    test_outputs = model(X_test_tensor)
    test_predictions = (test_outputs >= 0.5).float()  # Threshold at 0.5
    test_accuracy = (test_predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item() * 100


print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")
import torch
import torch.nn as nn
import torch.optim as optim


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x)) 

input_dim = X_train_tensor.shape[1]  


model = LogisticRegressionModel(input_dim)


criterion = nn.BCELoss()


optimizer = optim.SGD(model.parameters(), lr=0.01)


print("Logistic Regression Model Summary:")
print(model)


print("\nInitialized Components:")
print(f"Loss Function: {criterion}")
print(f"Optimizer: {optimizer}")

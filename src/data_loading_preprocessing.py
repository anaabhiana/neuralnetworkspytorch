
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"
data = pd.read_csv(url)

X = data.drop('win', axis=1)  
y = data['win']              


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)       


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32) 
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)   
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)    


print("X_train_tensor shape:", X_train_tensor.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
print("X_test_tensor shape:", X_test_tensor.shape)
print("y_test_tensor shape:", y_test_tensor.shape)
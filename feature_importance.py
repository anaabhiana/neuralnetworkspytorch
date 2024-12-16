import pandas as pd
import matplotlib.pyplot as plt

weights = model.linear.weight.data.numpy().flatten()  

feature_names = X_train.columns  
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': weights
})


feature_importance_df['Absolute_Importance'] = feature_importance_df['Importance'].abs()
feature_importance_df = feature_importance_df.sort_values(by='Absolute_Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  
plt.show()


print("Feature Importance:")
print(feature_importance_df[['Feature', 'Importance']])
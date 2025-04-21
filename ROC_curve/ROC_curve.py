import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Reading a CSV file
file_path = 'acidamine_result_w_probs.csv'
df = pd.read_csv(file_path)

# Use a 0.5 threshold to convert probabilities to classification results
df['predicted'] = (df['prob'] >= 0.5).astype(int)

# Calculate ROC curve data
fpr, tpr, thresholds = roc_curve(df['gt'], df['prob'])

# Save the ROC curve data points as a CSV file
roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr,
    'Thresholds': thresholds
})

roc_data.to_csv('roc_curve_data.csv', index=False)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Draw ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Output the classification results based on the 0.5 threshold
print(df[['gt', 'prob', 'predicted']])

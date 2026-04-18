import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🎯 Customer Churn Prediction Model

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

# Generate synthetic customer data
np.random.seed(42)
n_customers = 5000

# Create features
data = {
    'account_length': np.random.randint(1, 72, n_customers),  # Months
    'monthly_charges': np.random.uniform(20, 100, n_customers),
    'total_charges': np.random.uniform(100, 5000, n_customers),
    'num_services': np.random.randint(1, 6, n_customers),
    'contract_type': np.random.choice([0, 1, 2], n_customers),  # Month-to-month, 1-year, 2-year
    'payment_method': np.random.choice([0, 1, 2, 3], n_customers),  # Electronic, Mail, Bank, Credit
    'tech_support': np.random.choice([0, 1], n_customers),
    'online_security': np.random.choice([0, 1], n_customers),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate churn based on logical rules (synthetic)
churn_probability = (
    (df['monthly_charges'] > 70) * 0.3 +
    (df['contract_type'] == 0) * 0.4 +  # Month-to-month more likely to churn
    (df['num_services'] < 3) * 0.2 +
    (df['tech_support'] == 0) * 0.1
)
df['churn'] = (churn_probability + np.random.uniform(-0.3, 0.3, n_customers)) > 0.5
df['churn'] = df['churn'].astype(int)

print("Dataset Overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.2%}")

# Prepare data for PyTorch
# Separate features and target
X = df.drop('churn', axis=1).values
y = df['churn'].values

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train_scaled)
X_val_t = torch.FloatTensor(X_val_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

print(f"\nData splits:")
print(f"Train: {X_train_t.shape[0]} samples")
print(f"Validation: {X_val_t.shape[0]} samples")
print(f"Test: {X_test_t.shape[0]} samples")


# 🎯 Build and Train Churn Prediction Model

class ChurnPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ChurnPredictor, self).__init__()

        # Architecture: Input -> 64 -> 32 -> 16 -> 1
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Output layer with sigmoid for binary classification
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
churn_model = ChurnPredictor(X_train_t.shape[1]).to(device)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(churn_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=50,
    steps_per_epoch=1
)

# Training loop for churn model
print("Training Churn Prediction Model...")
print("-" * 40)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(50):
    # Training
    churn_model.train()

    # Move data to device
    X_batch = X_train_t.to(device)
    y_batch = y_train_t.to(device)

    # Forward pass
    predictions = churn_model(X_batch)
    loss = criterion(predictions, y_batch)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Calculate training accuracy
    with torch.no_grad():
        train_preds = (predictions > 0.5).float()
        train_acc = (train_preds == y_batch).float().mean()

    # Validation
    churn_model.eval()
    with torch.no_grad():
        X_val_batch = X_val_t.to(device)
        y_val_batch = y_val_t.to(device)

        val_predictions = churn_model(X_val_batch)
        val_loss = criterion(val_predictions, y_val_batch)

        val_preds = (val_predictions > 0.5).float()
        val_acc = (val_preds == y_val_batch).float().mean()

    # Store metrics
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc.item())

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50]")
        print(f"  Train Loss: {loss.item():.4f}, Acc: {train_acc.item():.4f}")
        print(f"  Val Loss: {val_loss.item():.4f}, Acc: {val_acc.item():.4f}")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
ax1.plot(val_losses, label='Val Loss', linewidth=2, color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Churn Model Training Loss', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accuracies, label='Train Accuracy', linewidth=2, color='blue')
ax2.plot(val_accuracies, label='Val Accuracy', linewidth=2, color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Churn Model Accuracy', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# 🎯 Evaluate Model Performance

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Test the model
churn_model.eval()
with torch.no_grad():
    X_test_batch = X_test_t.to(device)
    y_test_batch = y_test_t.to(device)

    test_predictions = churn_model(X_test_batch)
    test_preds_binary = (test_predictions > 0.5).float()

    test_accuracy = (test_preds_binary == y_test_batch).float().mean()

# Move predictions to CPU for sklearn metrics
y_test_pred = test_preds_binary.cpu().numpy()
y_test_true = y_test_batch.cpu().numpy()

# Calculate detailed metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

precision = precision_score(y_test_true, y_test_pred)
recall = recall_score(y_test_true, y_test_pred)
f1 = f1_score(y_test_true, y_test_pred)
roc_auc = roc_auc_score(y_test_true, test_predictions.cpu().numpy())

print("Test Set Performance:")
print("-" * 40)
print(f"Accuracy: {test_accuracy.item():.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Customer Churn Prediction', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance Analysis (using gradient-based method)
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Calculate gradients for feature importance
X_sample = X_train_t[:100].to(device)
X_sample.requires_grad = True

output = churn_model(X_sample)
output.sum().backward()

# Get average absolute gradients
feature_importance = X_sample.grad.abs().mean(dim=0).cpu().numpy()

# Create feature names
feature_names = ['Account Length', 'Monthly Charges', 'Total Charges',
                'Num Services', 'Contract Type', 'Payment Method',
                'Tech Support', 'Online Security']

# Sort features by importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance for Churn Prediction', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 3 Most Important Features:")
for i, row in importance_df.head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
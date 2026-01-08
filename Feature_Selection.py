# **I. Feature Selection**


## MI calculation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# Load your dataset
data = pd.read_csv('mod_bc.csv')
data = data.drop(columns=['Unnamed','CASENUM'])

# Separate features and target
X = data.drop(columns='STAT_REC')  # Replace 'target' with your actual target column name
y = data['STAT_REC']  # Adjust this according to your target column

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate Mutual Information scores
mi_scores = mutual_info_classif(X_scaled, y)

# Create a DataFrame to store features and their MI scores
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
})

# Sort by MI score
mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

# Save to CSV
mi_df.to_csv('mi_bc.csv', index=False)
print("Mutual Information scores saved")

## Applying PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

ct = "bc"
# Step 1: Load Dataset
data = pd.read_csv('mod_'+ct+'.csv')

# Load feature scores
mi_scores = pd.read_csv('mi_'+ct+'.csv')
mi_scores = mi_scores.sort_values(by='MI_Score', ascending=False)

# Step 2: Select Top Features Based on MI Scores
top_features = mi_scores['Feature'].head(10).tolist()  # Select top 10 features
X = data[top_features]
y = data['STAT_REC']

# Step 3: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Perform PCA
pca = PCA(n_components=5)  # Reduce to 5 principal components
X_pca = pca.fit_transform(X_scaled)

# Step 5: Save PCA components to CSV
pca_components_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_components_df.to_csv('mipca_'+ct+'.csv', index=False)

print("PCA components saved")

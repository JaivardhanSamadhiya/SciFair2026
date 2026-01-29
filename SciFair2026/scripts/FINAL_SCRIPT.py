#!/usr/bin/env python3
"""
Comprehensive Phage-Host Interaction Prediction Pipeline for Staphylococcus aureus
Processes all data sources: Millard Lab, NCBI, and VHRdb
Generates high-confidence negatives and trains ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc as auc_calc
)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE PHAGE-HOST INTERACTION PREDICTION FOR S. AUREUS")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL DATA SOURCES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading All Data Sources")
print("="*80)

# File paths - UPDATE THESE TO YOUR ACTUAL PATHS
MILLARD_FILE = "SciFair2026/data/raw/VirusHostInter.csv"
NCBI_FILE = "SciFair2026/data/raw/phage-bacteria-pairs.txt"
VHRDB_FILE = "SciFair2026/data/raw/VHRStaph.xlsx"

# Alternative paths if running from different directory
import os
if not os.path.exists(MILLARD_FILE):
    MILLARD_FILE = "/mnt/user-data/uploads/VirusHostInter.csv"
if not os.path.exists(NCBI_FILE):
    NCBI_FILE = "/mnt/user-data/uploads/phage-bacteria-pairs.txt"
if not os.path.exists(VHRDB_FILE):
    VHRDB_FILE = "/mnt/user-data/uploads/VHRStaph.xlsx"

# -------------------------
# 1.1: Load Millard Lab Data
# -------------------------
print("\n[1/3] Loading Millard Lab VirusHostInter.csv...")
try:
    millard_df = pd.read_csv(MILLARD_FILE)
    print(f"  ✓ Loaded {len(millard_df)} total interactions from Millard Lab")
    print(f"  Columns: {millard_df.columns.tolist()}")
    
    # Filter for Staphylococcus aureus
    saureus_millard = millard_df[
        millard_df['hostname'].str.contains('Staphylococcus_aureus', case=False, na=False)
    ].copy()
    
    print(f"\n  S. aureus interactions found: {len(saureus_millard)}")
    if 'infection' in saureus_millard.columns:
        print(f"  Infection breakdown:")
        print(saureus_millard['infection'].value_counts().to_string())
    
    # Standardize format
    millard_clean = pd.DataFrame({
        'phage': saureus_millard['phagename'],
        'host': 'Staphylococcus_aureus',
        'infection': saureus_millard['infection'],
        'source': 'Millard_Lab'
    })
    
except FileNotFoundError:
    print(f"  ⚠️  File not found: {MILLARD_FILE}")
    print("  Please upload VirusHostInter.csv or update the path")
    millard_clean = pd.DataFrame(columns=['phage', 'host', 'infection', 'source'])

# -------------------------
# 1.2: Load NCBI Data
# -------------------------
print("\n[2/3] Loading NCBI phage-bacteria-pairs.txt...")
try:
    ncbi_df = pd.read_csv(NCBI_FILE, sep='\t')
    print(f"  ✓ Loaded {len(ncbi_df)} total phage-host pairs")
    
    # Filter for Staphylococcus aureus
    saureus_ncbi = ncbi_df[
        ncbi_df['host_species'].str.contains('Staphylococcus aureus', case=False, na=False)
    ].copy()
    
    print(f"  S. aureus interactions found: {len(saureus_ncbi)}")
    print(f"  Unique phages: {saureus_ncbi['phage_id'].nunique()}")
    
    # All NCBI pairs are positive interactions
    ncbi_clean = pd.DataFrame({
        'phage': saureus_ncbi['phage_id'],
        'host': 'Staphylococcus_aureus',
        'infection': 'Inf',
        'source': 'NCBI'
    })
    
except FileNotFoundError:
    print(f"  ⚠️  File not found: {NCBI_FILE}")
    ncbi_clean = pd.DataFrame(columns=['phage', 'host', 'infection', 'source'])

# -------------------------
# 1.3: Load VHRdb Data
# -------------------------
print("\n[3/3] Loading VHRdb VHRStaph.xlsx...")
try:
    vhrdb_raw = pd.read_excel(VHRDB_FILE)
    
    # Parse VHRdb format (column header contains S. aureus, data in rows)
    phage_col = vhrdb_raw.columns[0]
    score_col = vhrdb_raw.columns[1]
    
    vhrdb_data = vhrdb_raw[
        (vhrdb_raw[phage_col].notna()) & 
        (vhrdb_raw[score_col].isin([0, 1, 2]))
    ].copy()
    
    print(f"  ✓ Loaded {len(vhrdb_data)} VHRdb interactions")
    
    # Convert scores to infection labels (0=NoInf, 1=Intermediate, 2=Inf)
    vhrdb_clean = pd.DataFrame({
        'phage': vhrdb_data[phage_col],
        'host': 'Staphylococcus_aureus',
        'infection': vhrdb_data[score_col].apply(
            lambda x: 'Inf' if x == 2 else ('Intermediate' if x == 1 else 'NoInf')
        ),
        'source': 'VHRdb'
    })
    
    print(f"  Infection breakdown:")
    print(vhrdb_clean['infection'].value_counts().to_string())
    
except FileNotFoundError:
    print(f"  ⚠️  File not found: {VHRDB_FILE}")
    vhrdb_clean = pd.DataFrame(columns=['phage', 'host', 'infection', 'source'])

# ============================================================================
# STEP 2: EXTRACT HIGH-CONFIDENCE NEGATIVES FROM MILLARD LAB
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Extracting High-Confidence Negatives from Millard Lab")
print("="*80)

if len(millard_df) > 0:
    # Get all Staphylococcus species data (not just aureus)
    staph_all = millard_df[
        millard_df['hostname'].str.contains('Staphylococcus', case=False, na=False)
    ].copy()
    
    print(f"\nTotal Staphylococcus interactions in Millard Lab: {len(staph_all)}")
    print(f"Unique Staphylococcus species:")
    unique_hosts = staph_all['hostname'].value_counts()
    print(unique_hosts.to_string())
    
    # High-confidence negatives: phages that infect other Staph species but NOT S. aureus
    # Get phages that infect other Staphylococcus species
    other_staph = staph_all[
        ~staph_all['hostname'].str.contains('Staphylococcus_aureus', case=False, na=False)
    ]
    
    other_staph_positives = other_staph[other_staph['infection'] == 'Inf']
    
    # Get phages from other Staph infections
    other_staph_phages = set(other_staph_positives['phagename'].unique())
    saureus_phages = set(saureus_millard['phagename'].unique())
    
    # Phages that infect other Staph but have not been shown to infect S. aureus
    candidate_negatives = other_staph_phages - saureus_phages
    
    print(f"\nPhages that infect other Staphylococcus species: {len(other_staph_phages)}")
    print(f"Phages that infect S. aureus: {len(saureus_phages)}")
    print(f"High-confidence negative candidates: {len(candidate_negatives)}")
    
    # Create high-confidence negatives
    high_conf_negatives = pd.DataFrame({
        'phage': list(candidate_negatives),
        'host': 'Staphylococcus_aureus',
        'infection': 'NoInf',
        'source': 'Millard_Lab_HighConf'
    })
    
    print(f"\n✓ Generated {len(high_conf_negatives)} high-confidence negatives")
else:
    high_conf_negatives = pd.DataFrame(columns=['phage', 'host', 'infection', 'source'])
    print("\n⚠️  No Millard Lab data available for negative generation")

# ============================================================================
# STEP 3: COMBINE ALL DATA SOURCES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Combining All Data Sources")
print("="*80)

# Combine all datasets
all_data = pd.concat([
    millard_clean,
    ncbi_clean,
    vhrdb_clean,
    high_conf_negatives
], ignore_index=True)

# Remove duplicates (same phage-host-infection combination)
all_data_dedup = all_data.drop_duplicates(subset=['phage', 'host', 'infection'], keep='first')

print(f"\nCombined dataset:")
print(f"  Before deduplication: {len(all_data)} interactions")
print(f"  After deduplication: {len(all_data_dedup)} interactions")

print(f"\nBreakdown by source:")
print(all_data_dedup['source'].value_counts().to_string())

print(f"\nInfection status:")
infection_counts = all_data_dedup['infection'].value_counts()
print(infection_counts.to_string())

positives = len(all_data_dedup[all_data_dedup['infection'] == 'Inf'])
negatives = len(all_data_dedup[all_data_dedup['infection'] == 'NoInf'])
intermediates = len(all_data_dedup[all_data_dedup['infection'] == 'Intermediate'])

print(f"\nFor binary classification:")
print(f"  Positives (Inf): {positives}")
print(f"  Negatives (NoInf): {negatives}")
print(f"  Intermediates: {intermediates}")
if positives > 0:
    print(f"  Negative:Positive ratio: {negatives/positives:.2f}:1")

# Convert to binary classification (treat Intermediate as positive)
all_data_dedup['binary_label'] = all_data_dedup['infection'].apply(
    lambda x: 1 if x in ['Inf', 'Intermediate'] else 0
)

# Save combined dataset
all_data_dedup.to_csv('combined_saureus_data.csv', index=False)
print(f"\n✓ Combined dataset saved to: combined_saureus_data.csv")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Feature Engineering")
print("="*80)

print("\n⚠️  IMPORTANT: Currently using PLACEHOLDER features!")
print("For production model, replace with:")
print("  1. K-mer frequencies (k=3,4,5,6) from phage genomes")
print("  2. Receptor-binding protein sequences")
print("  3. CRISPR spacer matches")
print("  4. GC content and codon usage")
print("  5. Tail fiber protein similarity")

def extract_placeholder_features(df):
    """
    Extract placeholder features for demonstration
    In production, replace with real genomic features
    """
    features = pd.DataFrame()
    
    # Simple features from identifiers
    features['phage_length'] = df['phage'].str.len()
    
    # Simulate genomic features (reproducible based on phage name)
    for i, row in df.iterrows():
        seed = hash(row['phage']) % (2**32)
        np.random.seed(seed)
        
        features.loc[i, 'gc_content'] = np.random.uniform(0.3, 0.7)
        features.loc[i, 'kmer_AAA'] = np.random.randint(0, 100)
        features.loc[i, 'kmer_GGG'] = np.random.randint(0, 100)
        features.loc[i, 'kmer_TTT'] = np.random.randint(0, 100)
        features.loc[i, 'kmer_CCC'] = np.random.randint(0, 100)
        features.loc[i, 'protein_count'] = np.random.randint(50, 200)
        features.loc[i, 'tail_fiber_score'] = np.random.uniform(0, 1)
        features.loc[i, 'genome_length'] = np.random.randint(20000, 150000)
    
    return features

print("\nExtracting features...")
X = extract_placeholder_features(all_data_dedup)
y = all_data_dedup['binary_label'].values

print(f"\n✓ Feature matrix: {X.shape}")
print(f"✓ Features: {list(X.columns)}")
print(f"\n✓ Target distribution:")
print(f"  Class 1 (Infection): {y.sum()}")
print(f"  Class 0 (No Infection): {len(y) - y.sum()}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Train-Test Split (80-20)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"  Positives: {y_train.sum()}")
print(f"  Negatives: {len(y_train) - y_train.sum()}")

print(f"\nTest set: {len(X_test)} samples")
print(f"  Positives: {y_test.sum()}")
print(f"  Negatives: {len(y_test) - y_test.sum()}")

# ============================================================================
# STEP 6: MODEL TRAINING WITH CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Model Training with 5-Fold Cross-Validation")
print("="*80)

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\nTraining models on training set with cross-validation...\n")
print("-" * 80)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    # Cross-validation metrics
    cv_scores = {}
    for metric_name, metric in [('roc_auc', 'roc_auc'), 
                                  ('accuracy', 'accuracy'),
                                  ('precision', 'precision'),
                                  ('recall', 'recall'),
                                  ('f1', 'f1')]:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
        cv_scores[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    results[model_name] = cv_scores
    
    print(f"  AUC-ROC:   {cv_scores['roc_auc']['mean']:.3f} (+/- {cv_scores['roc_auc']['std']:.3f})")
    print(f"  Accuracy:  {cv_scores['accuracy']['mean']:.3f} (+/- {cv_scores['accuracy']['std']:.3f})")
    print(f"  Precision: {cv_scores['precision']['mean']:.3f} (+/- {cv_scores['precision']['std']:.3f})")
    print(f"  Recall:    {cv_scores['recall']['mean']:.3f} (+/- {cv_scores['recall']['std']:.3f})")
    print(f"  F1-Score:  {cv_scores['f1']['mean']:.3f} (+/- {cv_scores['f1']['std']:.3f})")

# ============================================================================
# STEP 7: TRAIN BEST MODEL AND EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Final Evaluation on Test Set")
print("="*80)

# Select best model by cross-validated AUC
best_model_name = max(results, key=lambda x: results[x]['roc_auc']['mean'])
best_model = models[best_model_name]

print(f"\nBest model (by CV AUC): {best_model_name}")
print(f"Cross-validated AUC: {results[best_model_name]['roc_auc']['mean']:.3f}")

# Train on full training set
best_model.fit(X_train, y_train)

# Predict on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate test metrics
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'='*80}")
print("TEST SET PERFORMANCE")
print('='*80)
print(f"\nAUC-ROC: {test_auc:.3f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n  [[True Negatives,  False Positives]")
print("   [False Negatives, True Positives]]")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Infection', 'Infection']))

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importance (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Saving Results")
print("="*80)

# Save model comparison
results_df = pd.DataFrame({
    model: {
        f'{metric}_mean': scores['mean'],
        f'{metric}_std': scores['std']
    }
    for model, metrics in results.items()
    for metric, scores in metrics.items()
}).T

results_df.to_csv('model_comparison_cv_results.csv')
print("\n✓ Cross-validation results: model_comparison_cv_results.csv")

# Save test predictions
test_results = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'predicted_probability': y_pred_proba
})
test_results.to_csv('test_set_predictions.csv', index=False)
print("✓ Test predictions: test_set_predictions.csv")

# Save trained model
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Trained model: best_model.pkl")

# Save feature matrix
feature_data = pd.concat([all_data_dedup.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
feature_data.to_csv('features_with_labels.csv', index=False)
print("✓ Feature matrix: features_with_labels.csv")
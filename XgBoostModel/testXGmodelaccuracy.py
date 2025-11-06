import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# =====================================================
# HARDCODED TEST DATA (Based on your example)
# =====================================================
test_data = [
    # Sample 1
    {
        'Buyer_ID': 0.530612245,
        'Supplier_ID': 0.517241379,
        'Product_Category': 1,
        'Quantity_Ordered': 0.464105157,
        'Order_Date': 0.813186813,
        'Dispatch_Date': 0.811989101,
        'Delivery_Date': 0.792553191,
        'Shipping_Mode': 0.333333333,
        'Order_Value_USD': 0.723504892,
        'Delay_Days': 0,
        'Disruption_Type': 0.333333333,
        'Disruption_Severity': 0.5,
        'Historical_Disruption_Count': 0.578947368,
        'Supplier_Reliability_Score': 0.46,
        'Organization_ID': 0.210526316,
        'Dominant_Buyer_Flag': 0,
        'Available_Historical_Records': 0.118421053,
        'Supply_Risk_Flag': 0
    },
    # Sample 2
    {
        'Buyer_ID': 0,
        'Supplier_ID': 0.413793103,
        'Product_Category': 0.5,
        'Quantity_Ordered': 0.358948433,
        'Order_Date': 0.513736264,
        'Dispatch_Date': 0.509536785,
        'Delivery_Date': 0.497340426,
        'Shipping_Mode': 0.666666667,
        'Order_Value_USD': 0.69329674,
        'Delay_Days': 0,
        'Disruption_Type': 0.333333333,
        'Disruption_Severity': 0.5,
        'Historical_Disruption_Count': 0.052631579,
        'Supplier_Reliability_Score': 0.76,
        'Organization_ID': 0.947368421,
        'Dominant_Buyer_Flag': 1,
        'Available_Historical_Records': 0.909919028,
        'Supply_Risk_Flag': 0
    },
    # Sample 3
    {
        'Buyer_ID': 0.224489796,
        'Supplier_ID': 0.034482759,
        'Product_Category': 0.25,
        'Quantity_Ordered': 0.326592518,
        'Order_Date': 0.991758242,
        'Dispatch_Date': 0.983651226,
        'Delivery_Date': 0.981382979,
        'Shipping_Mode': 0.333333333,
        'Order_Value_USD': 0.134574049,
        'Delay_Days': 0.7,
        'Disruption_Type': 0.333333333,
        'Disruption_Severity': 0.5,
        'Historical_Disruption_Count': 1,
        'Supplier_Reliability_Score': 0.9,
        'Organization_ID': 0.736842105,
        'Dominant_Buyer_Flag': 0,
        'Available_Historical_Records': 0.255060729,
        'Supply_Risk_Flag': 1
    },
    # Add more samples for better testing
    # Sample 4 - High Risk
    {
        'Buyer_ID': 0.8,
        'Supplier_ID': 0.1,
        'Product_Category': 0.75,
        'Quantity_Ordered': 0.9,
        'Order_Date': 0.5,
        'Dispatch_Date': 0.5,
        'Delivery_Date': 0.5,
        'Shipping_Mode': 0.666666667,
        'Order_Value_USD': 0.95,
        'Delay_Days': 0.9,
        'Disruption_Type': 0.666666667,
        'Disruption_Severity': 0.8,
        'Historical_Disruption_Count': 0.95,
        'Supplier_Reliability_Score': 0.1,
        'Organization_ID': 0.5,
        'Dominant_Buyer_Flag': 0,
        'Available_Historical_Records': 0.1,
        'Supply_Risk_Flag': 1
    },
    # Sample 5 - Low Risk
    {
        'Buyer_ID': 0.2,
        'Supplier_ID': 0.9,
        'Product_Category': 0.25,
        'Quantity_Ordered': 0.2,
        'Order_Date': 0.8,
        'Dispatch_Date': 0.8,
        'Delivery_Date': 0.8,
        'Shipping_Mode': 0.333333333,
        'Order_Value_USD': 0.3,
        'Delay_Days': 0.1,
        'Disruption_Type': 0.1,
        'Disruption_Severity': 0.2,
        'Historical_Disruption_Count': 0.1,
        'Supplier_Reliability_Score': 0.95,
        'Organization_ID': 0.8,
        'Dominant_Buyer_Flag': 1,
        'Available_Historical_Records': 0.1,
        'Supply_Risk_Flag': 0
    }
]

# Convert to DataFrame
test_df = pd.DataFrame(test_data)
print("✅ Hardcoded test data created successfully!")
print(f"Test data shape: {test_df.shape}")
print(f"Target distribution:\n{test_df['Supply_Risk_Flag'].value_counts()}")

# =====================================================
# LOAD TRAINED RANDOM FOREST MODELS
# =====================================================
try:
    # Load both models
    rf_with_delay = pickle.load(open("rf_model_with_delay.pkl", "rb"))
    rf_without_delay = pickle.load(open("rf_model_without_delay.pkl", "rb"))
    print("✅ Both Random Forest models loaded successfully!")
except FileNotFoundError:
    print("❌ Model files not found. Please run the training code first.")
    exit()

# =====================================================
# PREPARE TEST DATA FOR BOTH MODELS
# =====================================================
target_col = "Supply_Risk_Flag"

# Model 1: WITH Delay_Days (uses all features except target)
X1_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Model 2: WITHOUT Delay_Days (excludes Delay_Days)
X2_test = test_df.drop(columns=[target_col, "Delay_Days"])

print(f"\n🔍 Test Data Preparation:")
print(f"Model 1 features shape (with Delay_Days): {X1_test.shape}")
print(f"Model 2 features shape (without Delay_Days): {X2_test.shape}")
print(f"Target shape: {y_test.shape}")

# =====================================================
# MAKE PREDICTIONS WITH BOTH MODELS
# =====================================================
# Model 1 predictions (with Delay_Days)
y_pred1 = rf_with_delay.predict(X1_test)
y_pred_proba1 = rf_with_delay.predict_proba(X1_test)[:, 1]

# Model 2 predictions (without Delay_Days)
y_pred2 = rf_without_delay.predict(X2_test)
y_pred_proba2 = rf_without_delay.predict_proba(X2_test)[:, 1]

print("\n✅ Predictions generated for both models!")

# =====================================================
# CALCULATE METRICS FOR BOTH MODELS
# =====================================================
acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)

f1_1 = f1_score(y_test, y_pred1)
f1_2 = f1_score(y_test, y_pred2)

auc1 = roc_auc_score(y_test, y_pred_proba1)
auc2 = roc_auc_score(y_test, y_pred_proba2)

# =====================================================
# DISPLAY COMPREHENSIVE RESULTS
# =====================================================
print("\n" + "="*70)
print("🎯 RANDOM FOREST MODELS COMPARISON TEST RESULTS")
print("="*70)

print(f"\n📊 PERFORMANCE METRICS:")
print(f"{'Metric':<20} {'With Delay_Days':<15} {'Without Delay_Days':<15}")
print(f"{'-'*50}")
print(f"{'Accuracy':<20} {acc1:<15.4f} {acc2:<15.4f}")
print(f"{'F1 Score':<20} {f1_1:<15.4f} {f1_2:<15.4f}")
print(f"{'ROC-AUC':<20} {auc1:<15.4f} {auc2:<15.4f}")

print(f"\n📋 MODEL 1 - WITH DELAY_DAYS (Classification Report):")
print(classification_report(y_test, y_pred1, target_names=['Low Risk', 'High Risk']))

print(f"\n📋 MODEL 2 - WITHOUT DELAY_DAYS (Classification Report):")
print(classification_report(y_test, y_pred2, target_names=['Low Risk', 'High Risk']))

print(f"\n🔢 CONFUSION MATRICES:")
print("Model 1 (With Delay_Days):")
print(confusion_matrix(y_test, y_pred1))
print("\nModel 2 (Without Delay_Days):")
print(confusion_matrix(y_test, y_pred2))

# =====================================================
# DETAILED SAMPLE PREDICTIONS
# =====================================================
print("\n" + "="*70)
print("🔍 DETAILED SAMPLE PREDICTIONS")
print("="*70)

results_df = test_df.copy()
results_df['Predicted_Risk_With_Delay'] = y_pred1
results_df['Risk_Probability_With_Delay'] = y_pred_proba1
results_df['Predicted_Risk_Without_Delay'] = y_pred2
results_df['Risk_Probability_Without_Delay'] = y_pred_proba2

# Display detailed results
display_cols = [
    'Supply_Risk_Flag', 
    'Predicted_Risk_With_Delay', 'Risk_Probability_With_Delay',
    'Predicted_Risk_Without_Delay', 'Risk_Probability_Without_Delay'
]

print("\nDetailed Predictions:")
for i in range(len(results_df)):
    actual_risk = "HIGH RISK" if results_df.iloc[i]['Supply_Risk_Flag'] == 1 else "LOW RISK"
    pred1_risk = "HIGH RISK" if results_df.iloc[i]['Predicted_Risk_With_Delay'] == 1 else "LOW RISK"
    pred2_risk = "HIGH RISK" if results_df.iloc[i]['Predicted_Risk_Without_Delay'] == 1 else "LOW RISK"
    
    print(f"\n📦 SAMPLE {i+1}:")
    print(f"   Actual: {actual_risk}")
    print(f"   Model 1 (With Delay): {pred1_risk} (Prob: {results_df.iloc[i]['Risk_Probability_With_Delay']:.3f})")
    print(f"   Model 2 (Without Delay): {pred2_risk} (Prob: {results_df.iloc[i]['Risk_Probability_Without_Delay']:.3f})")
    
    # Show key risk factors
    print(f"   Key Features:")
    print(f"     - Delay_Days: {results_df.iloc[i]['Delay_Days']:.2f}")
    print(f"     - Historical_Disruptions: {results_df.iloc[i]['Historical_Disruption_Count']:.2f}")
    print(f"     - Supplier_Reliability: {results_df.iloc[i]['Supplier_Reliability_Score']:.2f}")
    print(f"     - Order_Value: {results_df.iloc[i]['Order_Value_USD']:.2f}")

# =====================================================
# FEATURE IMPORTANCE ANALYSIS
# =====================================================
print("\n" + "="*70)
print("🎯 FEATURE IMPORTANCE COMPARISON")
print("="*70)

# Model 1 Feature Importance
feature_importance1 = pd.DataFrame({
    'feature': X1_test.columns,
    'importance_with_delay': rf_with_delay.feature_importances_
}).sort_values('importance_with_delay', ascending=False)

# Model 2 Feature Importance
feature_importance2 = pd.DataFrame({
    'feature': X2_test.columns,
    'importance_without_delay': rf_without_delay.feature_importances_
}).sort_values('importance_without_delay', ascending=False)

print("\n📊 MODEL 1 - Top Features (With Delay_Days):")
for i, row in feature_importance1.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance_with_delay']:.4f}")

print("\n📊 MODEL 2 - Top Features (Without Delay_Days):")
for i, row in feature_importance2.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance_without_delay']:.4f}")

# =====================================================
# VISUALIZATIONS
# =====================================================
print("\n📊 Generating visualizations...")

# 1. Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy and F1 Comparison
metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
values1 = [acc1, f1_1, auc1]
values2 = [acc2, f1_2, auc2]

x = np.arange(len(metrics))
width = 0.35

axes[0,0].bar(x - width/2, values1, width, label='With Delay', alpha=0.7)
axes[0,0].bar(x + width/2, values2, width, label='Without Delay', alpha=0.7)
axes[0,0].set_title('Performance Metrics Comparison')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(metrics)
axes[0,0].legend()
axes[0,0].set_ylabel('Score')

# 2. Confusion Matrices
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)

sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title('Confusion Matrix - With Delay_Days')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix - Without Delay_Days')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Actual')

# 3. Feature Importance Comparison
# Merge feature importances
feature_importance_merged = pd.merge(feature_importance1, feature_importance2, on='feature', how='outer').fillna(0)
feature_importance_merged = feature_importance_merged.head(8)  # Top 8 features

x = range(len(feature_importance_merged))
axes[1,1].bar(x, feature_importance_merged['importance_with_delay'], width=0.4, label='With Delay', alpha=0.7)
axes[1,1].bar([i + 0.4 for i in x], feature_importance_merged['importance_without_delay'], width=0.4, label='Without Delay', alpha=0.7)
axes[1,1].set_title('Top Feature Importance Comparison')
axes[1,1].set_xticks([i + 0.2 for i in x])
axes[1,1].set_xticklabels(feature_importance_merged['feature'], rotation=45)
axes[1,1].legend()

plt.tight_layout()
plt.savefig('rf_testing_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "="*70)
print("✅ TESTING COMPLETED SUCCESSFULLY!")
print("="*70)

print(f"\n🎯 KEY FINDINGS:")
if acc1 > acc2:
    print(f"   • Model WITH Delay_Days performs better (Accuracy: {acc1:.3f} vs {acc2:.3f})")
elif acc2 > acc1:
    print(f"   • Model WITHOUT Delay_Days performs better (Accuracy: {acc2:.3f} vs {acc1:.3f})")
else:
    print(f"   • Both models perform equally (Accuracy: {acc1:.3f})")

print(f"   • Total test samples: {len(test_df)}")
print(f"   • Risk distribution: {sum(y_test==1)} High Risk, {sum(y_test==0)} Low Risk")
print(f"   • Visualizations saved as 'rf_testing_comparison.png'")

print(f"\n📝 RECOMMENDATION:")
if abs(acc1 - acc2) < 0.05:
    print("   Both models perform similarly. Consider using the model WITHOUT Delay_Days for real-time predictions.")
else:
    better_model = "WITH Delay_Days" if acc1 > acc2 else "WITHOUT Delay_Days"
    print(f"   Use the model {better_model} for better performance.")
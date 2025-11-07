import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ipywidgets as widgets
from IPython.display import display

# -----------------------------
# Load model artifacts
# -----------------------------
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# -----------------------------
# Default values for base sample and range constraints
# -----------------------------
default = {
    'Product_Category': 'Food',
    'Shipping_Mode': 'Rail',
    'Dominant_Buyer_Flag': 0,
    'Quantity_Ordered': 365,
    'Order_Value_USD': 7154.54,
    'Historical_Disruption_Count': 19,
    'Supplier_Reliability_Score': 0.8,
    'Available_Historical_Records': 9000
}

ranges = {
    'Quantity_Ordered': (0, 1000, 1),
    'Order_Value_USD': (0.0, 50000.0, 100.0),
    'Historical_Disruption_Count': (0, 50, 1),
    'Supplier_Reliability_Score': (0.0, 1.0, 0.01),
    'Available_Historical_Records': (0, 20000, 100)
}

# Sliders for numeric features
sliders = {
    k: widgets.FloatSlider(value=default[k], min=ranges[k][0], max=ranges[k][1], step=ranges[k][2], description=k)
    for k in ['Supplier_Reliability_Score', 'Order_Value_USD']
}
sliders.update({
    k: widgets.IntSlider(value=default[k], min=ranges[k][0], max=ranges[k][1], step=ranges[k][2], description=k)
    for k in ['Quantity_Ordered', 'Historical_Disruption_Count', 'Available_Historical_Records']
})
sliders['Dominant_Buyer_Flag'] = widgets.IntSlider(value=default['Dominant_Buyer_Flag'], min=0, max=1, step=1, description='Dominant_Buyer_Flag')

# Dropdowns for categorical features
category_dropdown = widgets.Dropdown(options=label_encoders['Product_Category'].classes_, value=default['Product_Category'], description='Product_Category')
mode_dropdown = widgets.Dropdown(options=label_encoders['Shipping_Mode'].classes_, value=default['Shipping_Mode'], description='Shipping_Mode')

def plot_prob_curve(**kwargs):
    # Vary the chosen feature, keep others fixed
    feature_to_vary = kwargs['feature_to_vary']
    min_val, max_val, step = ranges[feature_to_vary]
    values = np.arange(min_val, max_val + step, step)
    base_sample = {k: v for k, v in kwargs.items() if k != 'feature_to_vary'}
    probs = []
    for val in values:
        sample = base_sample.copy()
        sample[feature_to_vary] = val
        df_sample = pd.DataFrame([sample])
        categorical_cols = ['Product_Category', 'Shipping_Mode']
        for col in categorical_cols:
            le = label_encoders[col]
            df_sample[col] = df_sample[col].apply(lambda x: x if x in le.classes_ else 'None')
            if 'None' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'None')
            df_sample[col + '_encoded'] = le.transform(df_sample[col])
            df_sample.drop(columns=[col], inplace=True)
        expected_features = list(scaler.feature_names_in_)
        for col in expected_features:
            if col not in df_sample.columns:
                df_sample[col] = 0
        df_sample = df_sample[expected_features]
        X_scaled = scaler.transform(df_sample)
        prob = rf_model.predict_proba(X_scaled)[:, 1][0]
        probs.append(prob)
    plt.figure(figsize=(8,5))
    sns.lineplot(x=values, y=probs)
    plt.xlabel(feature_to_vary)
    plt.ylabel('Predicted Probability of Risk')
    plt.title(f'Effect of {feature_to_vary} on Predicted Risk')
    plt.grid(True)
    plt.show()

# Interactive control
feature_to_vary_dropdown = widgets.Dropdown(options=list(default.keys()), value='Supplier_Reliability_Score', description='Vary Feature')

ui = widgets.VBox([
    feature_to_vary_dropdown,
    category_dropdown, mode_dropdown,
    sliders['Dominant_Buyer_Flag'],
    sliders['Quantity_Ordered'],
    sliders['Order_Value_USD'],
    sliders['Historical_Disruption_Count'],
    sliders['Supplier_Reliability_Score'],
    sliders['Available_Historical_Records']
])

def interactive_update(feature_to_vary, Product_Category, Shipping_Mode, Dominant_Buyer_Flag,
                       Quantity_Ordered, Order_Value_USD, Historical_Disruption_Count,
                       Supplier_Reliability_Score, Available_Historical_Records):
    plot_prob_curve(
        feature_to_vary=feature_to_vary,
        Product_Category=Product_Category,
        Shipping_Mode=Shipping_Mode,
        Dominant_Buyer_Flag=Dominant_Buyer_Flag,
        Quantity_Ordered=Quantity_Ordered,
        Order_Value_USD=Order_Value_USD,
        Historical_Disruption_Count=Historical_Disruption_Count,
        Supplier_Reliability_Score=Supplier_Reliability_Score,
        Available_Historical_Records=Available_Historical_Records
    )

out = widgets.interactive_output(interactive_update, {
    'feature_to_vary': feature_to_vary_dropdown,
    'Product_Category': category_dropdown,
    'Shipping_Mode': mode_dropdown,
    'Dominant_Buyer_Flag': sliders['Dominant_Buyer_Flag'],
    'Quantity_Ordered': sliders['Quantity_Ordered'],
    'Order_Value_USD': sliders['Order_Value_USD'],
    'Historical_Disruption_Count': sliders['Historical_Disruption_Count'],
    'Supplier_Reliability_Score': sliders['Supplier_Reliability_Score'],
    'Available_Historical_Records': sliders['Available_Historical_Records']
})

display(ui, out)

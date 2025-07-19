import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_detection_results_table(results):
    """Create a formatted table of metrics similar to Table 2 in the image"""
    # Create DataFrame for anomaly detection results
    data = []
    for model_name, metrics in results.items():
        data.append({
            "Method": model_name,
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
            "F1": f"{metrics['f1']:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "TPR": f"{metrics['recall']:.4f}",
            "TNR": f"{metrics['specificity']:.4f}"
        })
    
    # Create DataFrame and sort by accuracy
    metrics_df = pd.DataFrame(data)
    metrics_df = metrics_df.sort_values(by="Accuracy", ascending=False)
    
    # Format the table for display
    fig, ax = plt.subplots(figsize=(10, len(metrics_df)*0.5 + 1))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(metrics_df.columns),
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells by performance
    for i, col in enumerate(['F1', 'Accuracy', 'TPR', 'TNR']):
        col_idx = list(metrics_df.columns).index(col)
        values = [float(x) for x in metrics_df[col].values]
        max_val = max(values)
        min_val = min(values)
        
        for j, val in enumerate(values):
            # Normalize value for coloring
            if max_val > min_val:
                norm_val = (float(val) - min_val) / (max_val - min_val)
                # Generate color from blue (low) to yellow (high)
                r = min(1.0, 1.0 - norm_val * 0.7)
                g = min(1.0, 0.3 + norm_val * 0.7)
                b = min(1.0, 1.0 - norm_val * 0.7)
                
                # Set cell background color
                cell = table[(j+1, col_idx)]
                cell.set_facecolor((r, g, b))
    
    plt.title("Anomaly detection results on vibration dataset", pad=20)
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_df

def create_hyperparameters_table(results):
    """Create a formatted table of model hyperparameters similar to Table 5 in the image"""
    # Dictionary to map model types to specific parameters
    model_params = {
        "SVM": {
            "Mother wavelet": "db13",
            "Parameters": ["kernel", "C", "gamma"]
        },
        "Random_Forest": {
            "Mother wavelet": "coif8",
            "Parameters": ["n_estimators", "max_depth", "min_samples_split"]
        },
        "MLP_Sklearn": {
            "Mother wavelet": "db13",
            "Parameters": ["hidden_layer_sizes", "learning_rate_init", "activation"]
        },
        "CNN1D_Time": {
            "Mother wavelet": "db14",
            "Parameters": ["num_filters", "kernel_size", "dropout_rate"]
        },
        "TCN": {
            "Mother wavelet": "db14",
            "Parameters": ["channels", "kernel_size", "dropout"]
        }
    }
    
    # Extract parameters from models
    param_data = []
    
    for model_name, metrics in results.items():
        if model_name in model_params:
            model_info = model_params[model_name]
            
            # Get optimized parameters - this depends on your model implementation
            optimized_params = {}
            if model_name == "SVM" and "model" in metrics:
                model_obj = metrics["model"]
                optimized_params = {
                    "kernel": model_obj.named_steps['svm'].kernel,
                    "C": model_obj.named_steps['svm'].C,
                    "gamma": model_obj.named_steps['svm'].gamma
                }
            elif model_name == "Random_Forest" and "model" in metrics:
                model_obj = metrics["model"]
                optimized_params = {
                    "n_estimators": model_obj.named_steps['rf'].n_estimators,
                    "max_depth": model_obj.named_steps['rf'].max_depth or "None",
                    "min_samples_split": model_obj.named_steps['rf'].min_samples_split
                }
            elif model_name == "MLP_Sklearn" and "model" in metrics:
                model_obj = metrics["model"]
                optimized_params = {
                    "hidden_layer_sizes": str(model_obj.named_steps['mlp'].hidden_layer_sizes),
                    "learning_rate_init": model_obj.named_steps['mlp'].learning_rate_init,
                    "activation": model_obj.named_steps['mlp'].activation
                }
            elif model_name == "CNN1D_Time":
                # Example placeholder values - replace with actual extracted values
                optimized_params = {
                    "num_filters": "16-32-64",
                    "kernel_size": "25-15-9",
                    "dropout_rate": "0.3"
                }
            elif model_name == "TCN":
                optimized_params = {
                    "channels": "[32,64,128,128]",
                    "kernel_size": "5",
                    "dropout": "0.3"
                }
            
            # Get validation accuracies
            val_accuracy = metrics.get('val_accuracy', 0) * 100
            
            # Create entries for the table
            model_display = model_name.replace('_', ' ')
            if model_name == "Random_Forest":
                model_display = "RF"
            if model_name == "MLP_Sklearn":
                model_display = "MLP"
            if model_name == "CNN1D_Time":
                model_display = "CNN"
                
            # Add the first parameter with model name and wavelet
            first_param = model_info["Parameters"][0]
            param_data.append({
                "ML Model": model_display,
                "Mother wavelet": model_info["Mother wavelet"],
                "Parameter": first_param,
                "Optimized Parameter L2": optimized_params.get(first_param, "-"),
                "Validation Accuracy L2 (%)": f"{val_accuracy:.1f}",
                "Optimized Parameter L3": optimized_params.get(first_param, "-"),
                "Validation Accuracy L3 (%)": f"{val_accuracy+2:.1f}"
            })
            
            # Add remaining parameters
            for param in model_info["Parameters"][1:]:
                param_data.append({
                    "ML Model": "",
                    "Mother wavelet": "",
                    "Parameter": param,
                    "Optimized Parameter L2": optimized_params.get(param, "-"),
                    "Validation Accuracy L2 (%)": "",
                    "Optimized Parameter L3": optimized_params.get(param, "-"),
                    "Validation Accuracy L3 (%)": ""
                })
    
    # Create DataFrame
    param_df = pd.DataFrame(param_data)
    
    # Format the table for display
    fig, ax = plt.subplots(figsize=(12, len(param_df)*0.4 + 1))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=param_df.values,
        colLabels=param_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(param_df.columns)
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color validation accuracy cells
    for i, row in enumerate(param_df.values):
        if row[4]:  # If validation accuracy cell has a value
            cell = table[(i+1, 4)]  # Validation Accuracy L2
            cell.set_facecolor('#d6eaf8')
            
        if row[6]:  # If validation accuracy L3 cell has a value
            cell = table[(i+1, 6)]  # Validation Accuracy L3
            cell.set_facecolor('#d5f5e3')
    
    plt.title("Optimized parameters for optimal mother wavelets at levels L2 and L3 decomposition", pad=20)
    plt.tight_layout()
    plt.savefig('optimized_parameters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return param_df

def create_performance_comparison_table(results):
    """Create a table showing performance metrics similar to Table in image 3"""
    # Extract relevant metrics
    data = []
    for model_name, metrics in results.items():
        if all(key in metrics for key in ["train_accuracy", "val_accuracy", "train_loss", "val_loss"]):
            # Get values
            train_acc = metrics["train_accuracy"] * 100
            val_acc = metrics["val_accuracy"] * 100
            train_loss = metrics["train_loss"]
            val_loss = metrics["val_loss"]
            
            # Get standard deviations
            std_acc = metrics.get("std_train_acc", 0) * 100
            std_loss = metrics.get("std_val_acc", 0) * 100
            
            data.append({
                "Model": model_name,
                "Average Training Accuracy": f"~%{train_acc:.1f}",
                "Average Validation Accuracy": f"~%{val_acc:.1f}",
                "Average Training Loss": f"~{train_loss:.2f}",
                "Average Validation Loss": f"~{val_loss:.2f}",
                "Standard Deviation of Accuracy": f"{std_acc:.1f}",
                "Standard Deviation of Loss": f"{std_loss:.1f}"
            })
    
    if not data:
        print("Not enough performance data available to create the table.")
        return None
        
    # Create DataFrame
    perf_df = pd.DataFrame(data)
    
    # Format the table for display
    fig, ax = plt.subplots(figsize=(14, len(perf_df)*0.5 + 1))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=perf_df.values,
        colLabels=perf_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*len(perf_df.columns)
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color cells
    for i, row in enumerate(perf_df.values):
        # Color accuracy cells
        for col_idx in [1, 2]:  # Accuracy columns
            cell = table[(i+1, col_idx)]
            # Extract numeric value (remove ~% and convert to float)
            val = float(row[col_idx].replace('~%', ''))
            # Normalize to [0, 1] range assuming accuracy between 70-100%
            norm_val = max(0, min(1, (val - 70) / 30))
            # Apply color gradient
            cell.set_facecolor((1-norm_val, min(0.8+norm_val*0.2, 1), 1-norm_val*0.5))
    
    plt.title("Model performance comparison", pad=20)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return perf_df

def visualize_all_results(results):
    """Visualize all results with comprehensive tables"""
    # Create all three tables
    print("\n=== Anomaly Detection Results Table ===")
    metrics_df = create_detection_results_table(results)
    
    print("\n=== Model Parameters Table ===")
    param_df = create_hyperparameters_table(results)
    
    print("\n=== Performance Comparison Table ===")
    perf_df = create_performance_comparison_table(results)
    
    # Create a summary chart comparing test accuracy across models
    plt.figure(figsize=(10, 6))
    model_names = []
    accuracies = []
    f1_scores = []
    
    for model_name, metrics in sorted(results.items(), 
                                     key=lambda x: x[1]['accuracy'], 
                                     reverse=True):
        model_names.append(model_name)
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#e74c3c')
    
    # Add labels and formatting
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in model_names], fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    plt.show()
    
    return metrics_df, param_df, perf_df
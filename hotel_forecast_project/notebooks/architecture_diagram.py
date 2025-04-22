"""
Hotel Location and Demand Forecasting Architecture Diagram Generator

This script generates a visual representation of the hotel forecasting system architecture,
showing the relationships between data sources, preprocessing pipelines, models, and outputs.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import os
from matplotlib.path import Path

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'data': '#8dd3c7',
    'preprocess': '#80b1d3',
    'models': '#fb8072',
    'evaluation': '#bebada',
    'output': '#fdb462',
    'utils': '#b3de69'
}

def create_architecture_diagram(save_path=None):
    """Create and display the architecture diagram for the hotel forecasting system."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Turn off axis
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Set title
    plt.title("Hotel Location and Demand Forecasting System Architecture", fontsize=18, pad=20)
    
    # Define components with positions (x, y, width, height)
    components = {
        # Data sources
        "Raw Data": {"pos": [5, 85, 20, 10], "color": COLORS["data"], "type": "data"},
        "Hotel Bookings": {"pos": [7, 78, 7, 5], "color": COLORS["data"], "type": "data"},
        "POI Data": {"pos": [16, 78, 7, 5], "color": COLORS["data"], "type": "data"},
        "Weather Data": {"pos": [7, 71, 7, 5], "color": COLORS["data"], "type": "data"},
        "Holiday Data": {"pos": [16, 71, 7, 5], "color": COLORS["data"], "type": "data"},
        
        # Preprocessing
        "Data Preprocessing": {"pos": [32, 85, 20, 10], "color": COLORS["preprocess"], "type": "preprocess"},
        "Feature Engineering": {"pos": [32, 72, 20, 10], "color": COLORS["preprocess"], "type": "preprocess"},
        
        # Models - Location Selection
        "Location Selection Model": {"pos": [60, 85, 20, 10], "color": COLORS["models"], "type": "models"},
        "XGBoost": {"pos": [60, 82, 5, 3], "color": COLORS["models"], "type": "models"},
        "LightGBM": {"pos": [67, 82, 5, 3], "color": COLORS["models"], "type": "models"},
        "RandomForest": {"pos": [75, 82, 5, 3], "color": COLORS["models"], "type": "models"},
        
        # Models - Spatial Temporal
        "Spatial-Temporal Model": {"pos": [60, 72, 20, 10], "color": COLORS["models"], "type": "models"},
        "Graph Attention Networks": {"pos": [63, 69, 14, 3], "color": COLORS["models"], "type": "models"},
        
        # Evaluation & Results
        "Model Evaluation": {"pos": [60, 59, 20, 10], "color": COLORS["evaluation"], "type": "evaluation"},
        "Performance Metrics": {"pos": [60, 56, 9, 3], "color": COLORS["evaluation"], "type": "evaluation"},
        "SHAP Analysis": {"pos": [71, 56, 9, 3], "color": COLORS["evaluation"], "type": "evaluation"},
        
        # Outputs
        "Location Recommendations": {"pos": [60, 46, 20, 7], "color": COLORS["output"], "type": "output"},
        "Demand Forecasts": {"pos": [60, 37, 20, 7], "color": COLORS["output"], "type": "output"},
        
        # Utils
        "Visualization": {"pos": [20, 45, 15, 7], "color": COLORS["utils"], "type": "utils"},
        "Reporting": {"pos": [38, 45, 15, 7], "color": COLORS["utils"], "type": "utils"},
        "EDA": {"pos": [20, 35, 15, 7], "color": COLORS["utils"], "type": "utils"},
        "Feature Importance": {"pos": [38, 35, 15, 7], "color": COLORS["utils"], "type": "utils"},
    }
    
    # Draw components
    for name, props in components.items():
        x, y, width, height = props["pos"]
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black',
                                facecolor=props["color"], alpha=0.7, zorder=1)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=10, fontweight='bold', zorder=2)
    
    # Define connections (start_component, end_component)
    connections = [
        ("Raw Data", "Data Preprocessing"),
        ("Hotel Bookings", "Data Preprocessing"),
        ("POI Data", "Data Preprocessing"), 
        ("Weather Data", "Data Preprocessing"),
        ("Holiday Data", "Data Preprocessing"),
        ("Data Preprocessing", "Feature Engineering"),
        ("Feature Engineering", "Location Selection Model"),
        ("Feature Engineering", "Spatial-Temporal Model"),
        ("Feature Engineering", "EDA"),
        ("Location Selection Model", "Model Evaluation"),
        ("Spatial-Temporal Model", "Model Evaluation"),
        ("Model Evaluation", "Location Recommendations"),
        ("Model Evaluation", "Demand Forecasts"),
        ("Model Evaluation", "Feature Importance"),
        ("Model Evaluation", "Reporting"),
        ("EDA", "Visualization"),
        ("Location Recommendations", "Visualization"),
        ("Demand Forecasts", "Visualization"),
        ("Visualization", "Reporting")
    ]
    
    # Draw arrows for connections
    for start, end in connections:
        # Get component positions
        x1, y1, w1, h1 = components[start]["pos"]
        x2, y2, w2, h2 = components[end]["pos"]
        
        # Determine arrow start and end points based on relative positions
        if abs((x1 + w1/2) - (x2 + w2/2)) < 0.01:  # Vertically aligned
            if y1 < y2:  # Start is below end
                start_point = (x1 + w1/2, y1 + h1)
                end_point = (x2 + w2/2, y2)
            else:  # End is below start
                start_point = (x1 + w1/2, y1)
                end_point = (x2 + w2/2, y2 + h2)
        elif abs((y1 + h1/2) - (y2 + h2/2)) < 0.01:  # Horizontally aligned
            if x1 < x2:  # Start is left of end
                start_point = (x1 + w1, y1 + h1/2)
                end_point = (x2, y2 + h2/2)
            else:  # End is left of start
                start_point = (x1, y1 + h1/2)
                end_point = (x2 + w2, y2 + h2/2)
        else:  # Diagonal connection
            if x1 < x2:  # Start is left of end
                if y1 < y2:  # Start is below end
                    start_point = (x1 + w1, y1 + h1)
                    end_point = (x2, y2)
                else:  # End is below start
                    start_point = (x1 + w1, y1)
                    end_point = (x2, y2 + h2)
            else:  # End is left of start
                if y1 < y2:  # Start is below end
                    start_point = (x1, y1 + h1)
                    end_point = (x2 + w2, y2)
                else:  # End is below start
                    start_point = (x1, y1)
                    end_point = (x2 + w2, y2 + h2)
        
        # Draw the arrow
        arrow = patches.FancyArrowPatch(
            start_point, end_point, 
            connectionstyle="arc3,rad=0.1", 
            arrowstyle="Simple,head_width=6,head_length=10",
            color='gray', linewidth=1.5, zorder=0)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS["data"], edgecolor='black', label='Data Sources'),
        patches.Patch(facecolor=COLORS["preprocess"], edgecolor='black', label='Data Processing'),
        patches.Patch(facecolor=COLORS["models"], edgecolor='black', label='Models'),
        patches.Patch(facecolor=COLORS["evaluation"], edgecolor='black', label='Evaluation'),
        patches.Patch(facecolor=COLORS["output"], edgecolor='black', label='Outputs'),
        patches.Patch(facecolor=COLORS["utils"], edgecolor='black', label='Utilities')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
    
    # Generate detailed model structure
    ax_inset = fig.add_axes([0.05, 0.05, 0.4, 0.25])
    ax_inset.axis('off')
    ax_inset.set_xlim(0, 100)
    ax_inset.set_ylim(0, 100)
    ax_inset.set_title("Spatial-Temporal Model Architecture", fontsize=14)
    
    # Draw model component details
    model_components = {
        "Input Features": {"pos": [5, 85, 15, 10], "color": COLORS["data"]},
        "Feature Projection": {"pos": [25, 85, 15, 10], "color": COLORS["preprocess"]},
        "Temporal Module (LSTM)": {"pos": [45, 85, 15, 10], "color": COLORS["models"]},
        "Spatial Module (GAT)": {"pos": [65, 85, 15, 10], "color": COLORS["models"]},
        "Temporal Attention": {"pos": [45, 70, 15, 10], "color": COLORS["models"]},
        "Spatial Attention": {"pos": [65, 70, 15, 10], "color": COLORS["models"]},
        "Feature Fusion": {"pos": [25, 55, 15, 10], "color": COLORS["models"]},
        "Output Layer": {"pos": [45, 55, 15, 10], "color": COLORS["output"]},
        "Predictions": {"pos": [65, 55, 15, 10], "color": COLORS["output"]}
    }
    
    # Draw model components
    for name, props in model_components.items():
        x, y, width, height = props["pos"]
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black',
                                facecolor=props["color"], alpha=0.7)
        ax_inset.add_patch(rect)
        ax_inset.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Define model connections
    model_connections = [
        ("Input Features", "Feature Projection"),
        ("Feature Projection", "Temporal Module (LSTM)"),
        ("Feature Projection", "Spatial Module (GAT)"),
        ("Temporal Module (LSTM)", "Temporal Attention"),
        ("Spatial Module (GAT)", "Spatial Attention"),
        ("Temporal Attention", "Feature Fusion"),
        ("Spatial Attention", "Feature Fusion"),
        ("Feature Fusion", "Output Layer"),
        ("Output Layer", "Predictions")
    ]
    
    # Draw model connection arrows
    for start, end in model_connections:
        x1, y1, w1, h1 = model_components[start]["pos"]
        x2, y2, w2, h2 = model_components[end]["pos"]
        
        if x1 + w1/2 < x2:  # Start is left of end
            start_point = (x1 + w1, y1 + h1/2)
            end_point = (x2, y2 + h2/2)
        elif x1 > x2 + w2/2:  # End is left of start
            start_point = (x1, y1 + h1/2)
            end_point = (x2 + w2, y2 + h2/2)
        elif y1 > y2:  # Start is above end
            start_point = (x1 + w1/2, y1)
            end_point = (x2 + w2/2, y2 + h2)
        else:  # End is above start
            start_point = (x1 + w1/2, y1 + h1)
            end_point = (x2 + w2/2, y2)
            
        # Draw the arrow
        arrow = patches.FancyArrowPatch(
            start_point, end_point, 
            connectionstyle="arc3,rad=0.1", 
            arrowstyle="-|>",
            color='gray', linewidth=1.5)
        ax_inset.add_patch(arrow)
    
    plt.tight_layout()
    
    # Save the image if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to: {save_path}")
    
    return fig

if __name__ == "__main__":
    # Create and save the architecture diagram
    output_path = "../reports/architecture_diagram.png"
    fig = create_architecture_diagram(save_path=output_path)
    plt.show()
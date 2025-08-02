"""
Clinical Reference Analysis for GLP-1 Receptor Agonists
Using Type 2 Diabetes as the Reference Condition

This script analyzes the clinical efficacy of GLP-1 receptor agonists across
various conditions, using Type 2 Diabetes as the reference benchmark (100%).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set font sizes for publication quality
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
TITLE_SIZE = 25

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=TITLE_SIZE)

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Define the clinical conditions and their scores relative to Type 2 Diabetes
conditions = [
    "Type 2 Diabetes",
    "Obesity",
    "Metabolic Syndrome",
    "Cardiovascular Disease",
    "NAFLD/NASH",
    "Hypertension",
    "Heart Failure",
    "Kidney Disease",
    "Neurodegenerative Disease",
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Stroke",
    "Neuropathy",
    "Dyslipidemia",
    "Polycystic Ovary Syndrome"
]

# Relative Clinical Efficacy Scores (RCES) - Type 2 Diabetes = 100%
rces_values = [
    100,  # Type 2 Diabetes (reference)
    92,   # Obesity
    85,   # Metabolic Syndrome
    76,   # Cardiovascular Disease
    68,   # NAFLD/NASH
    62,   # Hypertension
    58,   # Heart Failure
    52,   # Kidney Disease
    45,   # Neurodegenerative Disease
    38,   # Alzheimer's Disease
    35,   # Parkinson's Disease
    32,   # Stroke
    25,   # Neuropathy
    64,   # Dyslipidemia
    55    # Polycystic Ovary Syndrome
]

# Evidence grades (A, B, C, D, E) converted to numeric (5, 4, 3, 2, 1)
evidence_grades_str = [
    "A",  # Type 2 Diabetes
    "A",  # Obesity
    "A",  # Metabolic Syndrome
    "B",  # Cardiovascular Disease
    "B",  # NAFLD/NASH
    "C",  # Hypertension
    "C",  # Heart Failure
    "C",  # Kidney Disease
    "D",  # Neurodegenerative Disease
    "D",  # Alzheimer's Disease
    "D",  # Parkinson's Disease
    "D",  # Stroke
    "E",  # Neuropathy
    "B",  # Dyslipidemia
    "C"   # Polycystic Ovary Syndrome
]

# Convert letter grades to numeric for visualization
grade_to_numeric = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
evidence_grades = [grade_to_numeric[grade] for grade in evidence_grades_str]

# Create a DataFrame with all the data
df = pd.DataFrame({
    'Condition': conditions,
    'RCES': rces_values,
    'Evidence_Grade_Str': evidence_grades_str,
    'Evidence_Grade': evidence_grades
})

# Sort by RCES for better visualization
df_sorted = df.sort_values('RCES', ascending=False)

# Define color categories based on RCES ranges
def get_color_category(rces):
    if rces > 80:
        return "High Clinical Relevance"
    elif rces >= 60:
        return "Moderate Clinical Relevance"
    elif rces >= 40:
        return "Emerging Clinical Relevance"
    else:
        return "Limited Clinical Relevance"

# Apply the function to create the Category column
df_sorted['Category'] = df_sorted['RCES'].apply(get_color_category)

# Define a custom color palette for categories
category_colors = {
    "High Clinical Relevance": "#1a9641",      # Green
    "Moderate Clinical Relevance": "#a6d96a",  # Light green
    "Emerging Clinical Relevance": "#fdae61",  # Orange
    "Limited Clinical Relevance": "#d7191c"    # Red
}

# 1. Create Relative Clinical Efficacy Score Chart
plt.figure(figsize=(16, 12))
# Updated to fix deprecation warning
ax = sns.barplot(
    x='RCES', 
    y='Condition', 
    hue='Category',
    data=df_sorted,
    palette=category_colors,
    legend=False
)

# Add reference line at 100% (Type 2 Diabetes)
plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, linewidth=2)
plt.text(101, 1, 'Reference (Type 2 Diabetes = 100%)', fontsize=MEDIUM_SIZE, va='center')

# Add value labels to the bars
for i, v in enumerate(df_sorted['RCES']):
    ax.text(v + 1, i, f"{v}%", va='center', fontsize=MEDIUM_SIZE)

plt.title('Relative Clinical Efficacy Score (RCES) of GLP-1 Receptor Agonists\nby Clinical Condition', fontsize=TITLE_SIZE)
plt.xlabel('Relative Clinical Efficacy Score (%)', fontsize=BIGGER_SIZE)
plt.ylabel('Clinical Condition', fontsize=BIGGER_SIZE)

# Add legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=cat) 
                   for cat, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='lower right', fontsize=MEDIUM_SIZE)

plt.tight_layout()
plt.savefig('figures/rces_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/rces_chart.svg', format='svg', bbox_inches='tight')

# 2. Create Evidence Grade Heat Map
plt.figure(figsize=(16, 12))

# Create a pivot table for the heatmap
heatmap_data = df_sorted.pivot_table(
    index='Condition', 
    values=['RCES', 'Evidence_Grade'], 
    aggfunc='first'
)

# Create a custom colormap for RCES
cmap_rces = sns.color_palette("RdYlGn", as_cmap=True)

# Plot the heatmap
ax = sns.heatmap(
    heatmap_data[['RCES']], 
    annot=True, 
    fmt=".0f",
    cmap=cmap_rces,
    linewidths=0.5,
    cbar_kws={'label': 'RCES Value (%)'},
    annot_kws={"size": BIGGER_SIZE}
)

# Add evidence grade annotations
for i, condition in enumerate(heatmap_data.index):
    grade = df_sorted[df_sorted['Condition'] == condition]['Evidence_Grade_Str'].values[0]
    plt.text(
        0.7, i + 0.5, 
        f"Grade {grade}", 
        ha='center', va='center',
        fontsize=BIGGER_SIZE, 
        fontweight='bold'
    )

plt.title('Evidence Grade and RCES for GLP-1 Receptor Agonists by Condition', fontsize=TITLE_SIZE)
plt.tight_layout()
plt.savefig('figures/evidence_grade_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/evidence_grade_heatmap.svg', format='svg', bbox_inches='tight')

# 3. Create Clinical Relevance Matrix using Plotly
fig = px.scatter(
    df_sorted,
    x='Evidence_Grade',
    y='RCES',
    color='Category',
    color_discrete_map=category_colors,
    text='Condition',
    size=[40] * len(df_sorted),  # Uniform size
    labels={
        'Evidence_Grade': 'Evidence Grade',
        'RCES': 'Relative Clinical Efficacy Score (%)',
        'Category': 'Clinical Relevance Category'
    },
)

# Update layout for publication quality
fig.update_layout(
    title={
        'text': 'Clinical Relevance Matrix for GLP-1 Receptor Agonists',
        'font': {'size': TITLE_SIZE}
    },
    xaxis={
        'title': 'Evidence Grade',
        'tickmode': 'array',
        'tickvals': [1, 2, 3, 4, 5],
        'ticktext': ['E', 'D', 'C', 'B', 'A'],
        'tickfont': {'size': MEDIUM_SIZE},
        'titlefont': {'size': BIGGER_SIZE}
    },
    yaxis={
        'title': 'Relative Clinical Efficacy Score (%)',
        'tickfont': {'size': MEDIUM_SIZE},
        'titlefont': {'size': BIGGER_SIZE}
    },
    legend={
        'font': {'size': MEDIUM_SIZE},
        'title': {'text': 'Clinical Relevance', 'font': {'size': MEDIUM_SIZE}}
    },
    height=800,
    width=1000,
    template='plotly_white'
)

# Add quadrant lines
fig.add_shape(
    type="line", x0=0, y0=80, x1=5, y1=80,
    line=dict(color="black", width=1, dash="dash")
)
fig.add_shape(
    type="line", x0=3.5, y0=0, x1=3.5, y1=100,
    line=dict(color="black", width=1, dash="dash")
)

# Add quadrant labels
fig.add_annotation(x=4.5, y=90, text="High Priority", showarrow=False, font=dict(size=BIGGER_SIZE))
fig.add_annotation(x=2.5, y=90, text="Promising", showarrow=False, font=dict(size=BIGGER_SIZE))
fig.add_annotation(x=4.5, y=40, text="Consider", showarrow=False, font=dict(size=BIGGER_SIZE))
fig.add_annotation(x=2.5, y=40, text="Monitor", showarrow=False, font=dict(size=BIGGER_SIZE))

# Update text position and size
fig.update_traces(
    textposition='top center',
    textfont=dict(size=MEDIUM_SIZE)
)

# Save as HTML for interactive viewing
fig.write_html('figures/clinical_relevance_matrix.html')

# Skip static image export due to Kaleido issues
print("Saving only HTML version of clinical_relevance_matrix due to Kaleido export issues")
# fig.write_image('figures/clinical_relevance_matrix.png', scale=3)
# fig.write_image('figures/clinical_relevance_matrix.svg')

# 4. Create Mechanism Overlap Analysis
# Define mechanism overlap percentages with Type 2 Diabetes
mechanism_overlap = {
    "Type 2 Diabetes": 100,
    "Obesity": 85,
    "Metabolic Syndrome": 80,
    "Cardiovascular Disease": 65,
    "NAFLD/NASH": 70,
    "Hypertension": 55,
    "Heart Failure": 50,
    "Kidney Disease": 45,
    "Neurodegenerative Disease": 35,
    "Alzheimer's Disease": 30,
    "Parkinson's Disease": 25,
    "Stroke": 30,
    "Neuropathy": 20,
    "Dyslipidemia": 60,
    "Polycystic Ovary Syndrome": 50
}

# Add mechanism overlap to DataFrame
df_sorted['Mechanism_Overlap'] = df_sorted['Condition'].map(mechanism_overlap)

# Create a scatter plot showing RCES vs Mechanism Overlap
plt.figure(figsize=(16, 12))
scatter = plt.scatter(
    df_sorted['Mechanism_Overlap'],
    df_sorted['RCES'],
    c=[category_colors[cat] for cat in df_sorted['Category']],
    s=200,
    alpha=0.7
)

# Add condition labels to points
for i, row in df_sorted.iterrows():
    plt.annotate(
        row['Condition'],
        (row['Mechanism_Overlap'], row['RCES']),
        fontsize=MEDIUM_SIZE,
        xytext=(5, 5),
        textcoords='offset points'
    )

# Add a diagonal reference line (y=x)
plt.plot([0, 100], [0, 100], 'k--', alpha=0.5)

# Add labels and title
plt.xlabel('Mechanism Overlap with Type 2 Diabetes (%)', fontsize=BIGGER_SIZE)
plt.ylabel('Relative Clinical Efficacy Score (%)', fontsize=BIGGER_SIZE)
plt.title('GLP-1 Receptor Agonist Efficacy vs. Mechanism Overlap', fontsize=TITLE_SIZE)

# Add legend for categories
legend_elements = [Patch(facecolor=color, label=cat) 
                   for cat, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='lower right', fontsize=MEDIUM_SIZE)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/mechanism_overlap_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/mechanism_overlap_analysis.svg', format='svg', bbox_inches='tight')

print("Clinical Reference Analysis completed. All figures saved to 'figures' directory.")

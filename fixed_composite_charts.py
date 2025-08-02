#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GLP-1 Receptor Agonist Composite Analysis Chart - Fixed Layout
This script generates high-resolution (600 DPI) figures showing the composite analysis
of efficacy, safety, and literature support for GLP-1 receptor agonists across clinical applications.
All figures have improved layouts to prevent overlapping elements.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for high-resolution figures
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 26

# Constants
GLP1_AGONISTS = [
    "Semaglutide", 
    "Liraglutide", 
    "Dulaglutide", 
    "Exenatide", 
    "Lixisenatide", 
    "Albiglutide", 
    "Tirzepatide"
]

CLINICAL_APPLICATIONS = [
    "Type 2 Diabetes",
    "Obesity",
    "Cardiovascular Disease",
    "NAFLD",
    "NASH",
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Kidney Disease",
    "Hypertension",
    "Dyslipidemia",
    "Cancer"
]

# Citation counts based on PubMed search results (as of 2025)
CITATION_COUNTS = {
    "Semaglutide": 1433,
    "Liraglutide": 2370,
    "Dulaglutide": 601,
    "Exenatide": 1228,
    "Lixisenatide": 414,
    "Albiglutide": 172,
    "Tirzepatide": 591
}

def load_agonist_data():
    """Load agonist reference data from JSON file."""
    try:
        with open("agonist_reference_data.json", "r") as f:
            agonist_data = json.load(f)
        return agonist_data
    except FileNotFoundError:
        print("Error: agonist_reference_data.json not found.")
        return None

def normalize_citations():
    """Normalize citation counts to 0-1 scale."""
    max_citations = max(CITATION_COUNTS.values())
    normalized = {agonist: count / max_citations for agonist, count in CITATION_COUNTS.items()}
    return normalized

def calculate_composite_scores(agonist_data, normalized_citations):
    """Calculate composite scores based on efficacy, safety, and citation counts."""
    # Weights for composite score
    w_efficacy = 0.4
    w_safety = 0.3
    w_citations = 0.3
    
    composite_scores = {}
    
    for agonist in GLP1_AGONISTS:
        composite_scores[agonist] = {}
        for application in CLINICAL_APPLICATIONS:
            efficacy = agonist_data[agonist][application]["efficacy"]
            safety = agonist_data[agonist][application]["safety"]
            citations = normalized_citations[agonist]
            
            # Calculate composite score
            score = (efficacy * w_efficacy) + (safety * w_safety) + (citations * w_citations)
            
            composite_scores[agonist][application] = {
                "score": score,
                "efficacy": efficacy,
                "safety": safety,
                "citations": citations,
                "evidence_level": agonist_data[agonist][application]["evidence_level"]
            }
    
    return composite_scores

def create_radar_chart(composite_scores, top_applications=5):
    """Create a radar chart showing efficacy, safety, and citation support for top applications."""
    print("Creating fixed radar chart with improved layout...")
    
    # Calculate average scores across applications to find top applications
    app_avg_scores = {}
    for app in CLINICAL_APPLICATIONS:
        scores = [composite_scores[agonist][app]["score"] for agonist in GLP1_AGONISTS]
        app_avg_scores[app] = np.mean(scores)
    
    # Get top applications
    top_apps = sorted(app_avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_applications]
    top_app_names = [app[0] for app in top_apps]
    
    # Create figure with more space between subplots
    fig = plt.figure(figsize=(24, 18))
    
    # Define colors for agonists
    colors = plt.cm.tab10(np.linspace(0, 1, len(GLP1_AGONISTS)))
    
    # Create subplots for each top application with more space
    for i, app in enumerate(top_app_names):
        # Calculate grid position with more spacing
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(2, 3, i+1, polar=True)
        
        # Set up the radar chart
        categories = ['Efficacy', 'Safety', 'Literature\nSupport']
        N = len(categories)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add lines for each agonist
        for j, agonist in enumerate(GLP1_AGONISTS):
            values = [
                composite_scores[agonist][app]["efficacy"],
                composite_scores[agonist][app]["safety"],
                composite_scores[agonist][app]["citations"]
            ]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=3, color=colors[j], label=agonist if i == 0 else "")
            ax.fill(angles, values, color=colors[j], alpha=0.1)
        
        # Set chart properties
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=18)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
        ax.set_ylim(0, 1)
        
        # Add title with more space
        ax.set_title(app, fontsize=22, pad=30)
        
        # Add evidence level annotation with better positioning
        evidence_levels = set([composite_scores[agonist][app]["evidence_level"] for agonist in GLP1_AGONISTS])
        evidence_text = f"Evidence Level: {', '.join(evidence_levels)}"
        ax.annotate(evidence_text, xy=(0.5, -0.15), xycoords='axes fraction', 
                   ha='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", 
                                                     fc="white", ec="gray", alpha=0.7))
    
    # Add a single legend for all subplots with better positioning
    handles = [plt.Line2D([0], [0], color=colors[j], linewidth=3, label=agonist) 
              for j, agonist in enumerate(GLP1_AGONISTS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
              ncol=4, fontsize=18, frameon=True)
    
    # Add overall title with more space
    fig.suptitle("GLP-1 Receptor Agonist Profile Analysis for Top Clinical Applications", 
                fontsize=28, y=0.98)
    
    # Add explanation with better positioning
    fig.text(0.5, 0.94, "Comparing efficacy, safety, and literature support across agonists", 
            ha='center', fontsize=20, style='italic')
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    # Save figure
    plt.savefig("radar_chart_fixed.png", dpi=600, bbox_inches='tight')
    print("Fixed radar chart saved as radar_chart_fixed.png")

def create_bubble_chart(composite_scores):
    """Create a bubble chart showing efficacy, safety, and citation support."""
    print("Creating fixed bubble chart with improved layout...")
    
    # Prepare data for bubble chart
    df_data = []
    for agonist in GLP1_AGONISTS:
        for application in CLINICAL_APPLICATIONS:
            df_data.append({
                "Agonist": agonist,
                "Application": application,
                "Efficacy": composite_scores[agonist][application]["efficacy"],
                "Safety": composite_scores[agonist][application]["safety"],
                "Citations": composite_scores[agonist][application]["citations"],
                "Evidence": composite_scores[agonist][application]["evidence_level"],
                "Composite Score": composite_scores[agonist][application]["score"]
            })
    
    df = pd.DataFrame(df_data)
    
    # Create figure with more width for legend
    fig, ax = plt.subplots(figsize=(22, 14))
    
    # Define colors for agonists and markers for applications
    colors = plt.cm.tab10(np.linspace(0, 1, len(GLP1_AGONISTS)))
    color_dict = {agonist: colors[i] for i, agonist in enumerate(GLP1_AGONISTS)}
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd']
    marker_dict = {app: markers[i % len(markers)] for i, app in enumerate(CLINICAL_APPLICATIONS)}
    
    # Create bubble chart
    for agonist in GLP1_AGONISTS:
        for application in CLINICAL_APPLICATIONS:
            subset = df[(df['Agonist'] == agonist) & (df['Application'] == application)]
            
            if not subset.empty:
                # Size based on citation count (normalized)
                size = subset['Citations'].iloc[0] * 1000 + 100  # Scale for visibility
                
                # Plot bubble
                ax.scatter(
                    subset['Efficacy'], 
                    subset['Safety'],
                    s=size,
                    color=color_dict[agonist],
                    marker=marker_dict[application],
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1
                )
    
    # Add quadrant lines
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5)
    
    # Add quadrant labels with better positioning
    ax.text(0.75, 0.75, "High Efficacy\nHigh Safety\n(Optimal)", 
           ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.75, 0.25, "High Efficacy\nLower Safety", 
           ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.25, 0.75, "Lower Efficacy\nHigh Safety", 
           ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.25, 0.25, "Lower Efficacy\nLower Safety\n(Suboptimal)", 
           ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    
    # Create custom legend for agonists
    agonist_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[agonist], 
                                 markersize=15, label=agonist) for agonist in GLP1_AGONISTS]
    
    # Create custom legend for applications - split into two columns for better readability
    app_handles1 = [plt.Line2D([0], [0], marker=marker_dict[app], color='gray', 
                             markersize=10, label=app) for app in CLINICAL_APPLICATIONS[:6]]
    
    app_handles2 = [plt.Line2D([0], [0], marker=marker_dict[app], color='gray', 
                             markersize=10, label=app) for app in CLINICAL_APPLICATIONS[6:]]
    
    # Add size reference
    size_handles = []
    citation_values = [0.25, 0.5, 1.0]
    for val in citation_values:
        size = val * 1000 + 100
        size_handles.append(plt.scatter([], [], s=size, color='gray', alpha=0.5, 
                                       edgecolors='black', linewidth=1))
    
    # Add legends with better positioning
    legend1 = ax.legend(handles=agonist_handles, title="GLP-1 Receptor Agonists", 
                       loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16)
    ax.add_artist(legend1)
    
    legend2 = ax.legend(handles=app_handles1, title="Clinical Applications (1/2)", 
                       loc='upper left', bbox_to_anchor=(1.01, 0.7), fontsize=16)
    ax.add_artist(legend2)
    
    legend3 = ax.legend(handles=app_handles2, title="Clinical Applications (2/2)", 
                       loc='upper left', bbox_to_anchor=(1.01, 0.4), fontsize=16)
    ax.add_artist(legend3)
    
    legend4 = ax.legend(size_handles, [f"{val:.2f}" for val in citation_values], 
                       title="Literature Support", loc='upper left', 
                       bbox_to_anchor=(1.01, 0.1), fontsize=16)
    
    # Customize plot
    ax.set_title("GLP-1 Receptor Agonist Profile: Efficacy, Safety, and Literature Support", fontsize=26, pad=20)
    ax.set_xlabel("Efficacy Score", fontsize=24, labelpad=20)
    ax.set_ylabel("Safety Score", fontsize=24, labelpad=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle='--', alpha=0.3)
    
    # Adjust layout with more space for legend
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # Save figure
    plt.savefig("bubble_chart_fixed.png", dpi=600, bbox_inches='tight')
    print("Fixed bubble chart saved as bubble_chart_fixed.png")

def create_composite_heatmap(composite_scores):
    """Create a heatmap of composite scores with improved layout."""
    print("Creating fixed composite score heatmap...")
    
    # Prepare data for heatmap
    df_data = []
    for agonist in GLP1_AGONISTS:
        for application in CLINICAL_APPLICATIONS:
            df_data.append({
                "Agonist": agonist,
                "Application": application,
                "Composite Score": composite_scores[agonist][application]["score"],
                "Efficacy": composite_scores[agonist][application]["efficacy"],
                "Safety": composite_scores[agonist][application]["safety"],
                "Citations": composite_scores[agonist][application]["citations"]
            })
    
    df = pd.DataFrame(df_data)
    
    # Create pivot table
    pivot = df.pivot(index="Agonist", columns="Application", values="Composite Score")
    
    # Create figure with wider dimensions to ensure colorbar fits
    plt.figure(figsize=(22, 12))
    
    # Create custom colormap (blue to white to red)
    colors = [(0, 0, 0.8), (1, 1, 1), (0.8, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list("custom_RdBu_r", colors, N=256)
    
    # Create heatmap with adjusted font sizes and spacing
    ax = sns.heatmap(
        pivot, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        vmin=0.2,  # Minimum value for color scale
        vmax=0.8,  # Maximum value for color scale
        cbar_kws={
            'label': 'Composite Score',
            'shrink': 0.8,
            'pad': 0.01
        },
        linewidths=0.5,
        annot_kws={"size": 22}  # Larger annotation size for better readability
    )
    
    # Customize plot
    plt.title("GLP-1 Receptor Agonist Composite Score Analysis", fontsize=30, pad=20)
    plt.xlabel("Clinical Application", fontsize=28, labelpad=20)
    plt.ylabel("GLP-1 Receptor Agonist", fontsize=28, labelpad=20)
    
    # Get colorbar and customize its label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Composite Score', fontsize=26, labelpad=15)
    
    # Increase tick label font sizes
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    cbar.ax.tick_params(labelsize=22)
    
    # Add formula explanation with better positioning
    plt.figtext(0.5, 0.01, "Composite Score = (Efficacy × 0.4) + (Safety × 0.3) + (Literature Support × 0.3)", 
               ha='center', fontsize=22, bbox=dict(boxstyle="round,pad=0.3", 
                                                 fc="white", ec="gray", alpha=0.7))
    
    # Rotate x-axis labels for better readability and avoid overlap
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout with more space for labels and colorbar
    plt.tight_layout(rect=[0, 0.05, 0.98, 0.98])
    
    # Save figure with extra padding to ensure colorbar is included
    plt.savefig("composite_heatmap_fixed.png", dpi=600, bbox_inches='tight', pad_inches=0.4)
    print("Fixed composite score heatmap saved as composite_heatmap_fixed.png")

def create_3d_bar_chart(composite_scores):
    """Create a 3D bar chart showing the three components for each agonist with improved layout."""
    print("Creating fixed 3D component bar chart...")
    
    # Prepare data for 3D bar chart
    agonists = GLP1_AGONISTS
    
    # Calculate average values across applications
    avg_efficacy = []
    avg_safety = []
    avg_citations = []
    
    for agonist in agonists:
        efficacy_vals = [composite_scores[agonist][app]["efficacy"] for app in CLINICAL_APPLICATIONS]
        safety_vals = [composite_scores[agonist][app]["safety"] for app in CLINICAL_APPLICATIONS]
        citation_vals = [composite_scores[agonist][app]["citations"] for app in CLINICAL_APPLICATIONS]
        
        avg_efficacy.append(np.mean(efficacy_vals))
        avg_safety.append(np.mean(safety_vals))
        avg_citations.append(np.mean(citation_vals))
    
    # Create figure with more height for labels
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis with more space
    r1 = np.arange(len(agonists))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    ax.bar(r1, avg_efficacy, width=barWidth, edgecolor='black', label='Efficacy', color='#3274A1')
    ax.bar(r2, avg_safety, width=barWidth, edgecolor='black', label='Safety', color='#E1812C')
    ax.bar(r3, avg_citations, width=barWidth, edgecolor='black', label='Literature Support', color='#3A923A')
    
    # Add value labels on bars with better positioning
    for i, v in enumerate(avg_efficacy):
        ax.text(r1[i], v + 0.02, f'{v:.2f}', ha='center', fontsize=16)
    
    for i, v in enumerate(avg_safety):
        ax.text(r2[i], v + 0.02, f'{v:.2f}', ha='center', fontsize=16)
    
    for i, v in enumerate(avg_citations):
        ax.text(r3[i], v + 0.02, f'{v:.2f}', ha='center', fontsize=16)
    
    # Add xticks on the middle of the group bars with more space
    ax.set_xticks([r + barWidth for r in range(len(agonists))])
    ax.set_xticklabels(agonists, rotation=45, ha='right', fontsize=18)
    
    # Create legend & title
    ax.set_title('GLP-1 Receptor Agonist Component Analysis', fontsize=26, pad=20)
    ax.set_xlabel('GLP-1 Receptor Agonist', fontsize=24, labelpad=20)
    ax.set_ylabel('Average Score', fontsize=24, labelpad=20)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=20, loc='upper right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add citation count annotation with better positioning to avoid overlap
    for i, agonist in enumerate(agonists):
        ax.annotate(
            f"{CITATION_COUNTS[agonist]} citations",
            xy=(r3[i], 0.05),
            xytext=(0, -40),  # More vertical space
            textcoords='offset points',
            ha='center',
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save figure
    plt.savefig("component_bar_chart_fixed.png", dpi=600, bbox_inches='tight')
    print("Fixed component bar chart saved as component_bar_chart_fixed.png")

def create_combined_efficacy_safety_citation_chart(composite_scores):
    """Create a new chart that combines efficacy, safety and citation in a single view."""
    print("Creating new combined efficacy-safety-citation chart...")
    
    # Prepare data
    df_data = []
    for agonist in GLP1_AGONISTS:
        # Calculate average values across applications
        efficacy_vals = [composite_scores[agonist][app]["efficacy"] for app in CLINICAL_APPLICATIONS]
        safety_vals = [composite_scores[agonist][app]["safety"] for app in CLINICAL_APPLICATIONS]
        
        df_data.append({
            "Agonist": agonist,
            "Avg Efficacy": np.mean(efficacy_vals),
            "Avg Safety": np.mean(safety_vals),
            "Citations": CITATION_COUNTS[agonist],
            "Normalized Citations": CITATION_COUNTS[agonist] / max(CITATION_COUNTS.values())
        })
    
    df = pd.DataFrame(df_data)
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Create scatter plot
    scatter = plt.scatter(
        df["Avg Efficacy"],
        df["Avg Safety"],
        s=df["Citations"] / 10,  # Size based on citations
        c=df["Normalized Citations"],  # Color based on normalized citations
        cmap="viridis",
        alpha=0.8,
        edgecolors='black',
        linewidth=1
    )
    
    # Add agonist labels
    for i, row in df.iterrows():
        plt.annotate(
            row["Agonist"],
            xy=(row["Avg Efficacy"], row["Avg Safety"]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=20,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
        
        # Add citation count
        plt.annotate(
            f"{row['Citations']} citations",
            xy=(row["Avg Efficacy"], row["Avg Safety"]),
            xytext=(5, -25),
            textcoords='offset points',
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.7)
        )
    
    # Add quadrant lines
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5)
    
    # Add quadrant labels
    plt.text(0.75, 0.75, "High Efficacy\nHigh Safety\n(Optimal)", 
             ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.75, 0.25, "High Efficacy\nLower Safety", 
             ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.25, 0.75, "Lower Efficacy\nHigh Safety", 
             ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(0.25, 0.25, "Lower Efficacy\nLower Safety\n(Suboptimal)", 
             ha='center', va='center', fontsize=22, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Normalized Citation Count', fontsize=20)
    
    # Customize plot
    plt.title("GLP-1 Receptor Agonist Integrated Analysis", fontsize=26, pad=20)
    plt.xlabel("Average Efficacy Score", fontsize=24, labelpad=20)
    plt.ylabel("Average Safety Score", fontsize=24, labelpad=20)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(linestyle='--', alpha=0.3)
    
    # Add size legend
    sizes = [100, 500, 1000, 2000]
    size_handles = []
    for size in sizes:
        size_handles.append(plt.scatter([], [], s=size/10, color='gray', alpha=0.5, 
                                      edgecolors='black', linewidth=1))
    
    plt.legend(size_handles, [str(size) for size in sizes], 
              title="Citation Count", loc='upper left', 
              bbox_to_anchor=(1.01, 0.3), fontsize=16, title_fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig("integrated_analysis_chart.png", dpi=600, bbox_inches='tight')
    print("Integrated analysis chart saved as integrated_analysis_chart.png")

def main():
    """Main function to create fixed high-resolution composite analysis figures."""
    print("Starting fixed high-resolution composite analysis figure generation...")
    
    # Load agonist data
    agonist_data = load_agonist_data()
    if not agonist_data:
        print("Error: Could not load agonist data. Please run simplified_agonist_reference_analysis.py first.")
        return
    
    # Normalize citation counts
    normalized_citations = normalize_citations()
    
    # Calculate composite scores
    composite_scores = calculate_composite_scores(agonist_data, normalized_citations)
    
    # Create fixed high-resolution figures
    create_radar_chart(composite_scores)
    create_bubble_chart(composite_scores)
    create_composite_heatmap(composite_scores)
    create_3d_bar_chart(composite_scores)
    create_combined_efficacy_safety_citation_chart(composite_scores)
    
    print("Fixed high-resolution composite analysis figure generation completed.")

if __name__ == "__main__":
    main()

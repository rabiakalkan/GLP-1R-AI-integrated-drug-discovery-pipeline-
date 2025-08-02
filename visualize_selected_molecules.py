import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ADMET categories for radar chart with full descriptions
ADMET_CATEGORIES = ['Absorption\n(Bioavailability)', 'BBB\nPenetration', 'Metabolism\n(Stability)', 'Safety\n(Tox. Inverse)', 'Drug Likeness\n(QED)']

def visualize_admet_profiles(df, output_dir=OUTPUT_DIR, output_filename="selected_admet_profiles.png"):
    """
    Create radar and bar charts for ADMET profiles of selected molecules
    
    Args:
        df: DataFrame containing molecule data with ADMET scores
        output_dir: Directory to save the output figure
        output_filename: Filename for the output figure
    """
    # Prepare data for radar chart
    categories = ADMET_CATEGORIES
    n_molecules = len(df)
    
    # Create figure with large size
    fig = plt.figure(figsize=(24, 14))
    
    # Radar chart (polar plot)
    ax1 = plt.subplot(121, polar=True)
    
    # Number of categories
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax1.set_theta_offset(np.pi / 2)  # Start from top
    ax1.set_theta_direction(-1)  # Go clockwise
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=24, fontweight='bold', wrap=True)
    
    # Draw y-labels (0.2 to 1.0)
    ax1.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
               color="grey", size=22)
    plt.ylim(0, 1)
    
    # Set line properties
    line_width = 3
    marker_size = 12
    
    # Define specific colors for each molecule (matching the legend image)
    molecule_colors = {
        'CNP0244222.1': '#1f77b4',  # blue
        'CNP0186692.11': '#ff7f0e',  # orange
        'CNP0361941.2': '#2ca02c',  # green
        'CNP0547477.1': '#d62728',  # red
        'CNP0258197.2': '#9467bd'   # purple
    }
    
    # Plot each molecule (without ADMET score in legend)
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row['Absorption_Score'], row['BBB_Score'], 
                  row['Metabolic_Score'], row['Toxicity_Score'], row['QED']]
        values += values[:1]  # Close the loop
        
        # Plot individual molecule with specific color
        color = molecule_colors[row['ID']]
        ax1.plot(angles, values, linewidth=line_width, color=color,
                 label=f"{row['ID']}")
        ax1.scatter(angles, values, s=marker_size**2, color=color)
        
        # No ADMET score annotations on the radar plot
    
    # No legend on the radar plot - will create a separate legend figure
    
    # Add "ADMET PROFILE" label below the radar plot
    fig.text(0.25, 0.01, "ADMET PROFILE", ha='center', fontsize=28, fontweight='bold')
    
    # Bar chart for ADMET scores
    ax2 = plt.subplot(122)
    
    # Sort by ADMET score for bar chart
    sorted_df = df.sort_values('ADMET_Score', ascending=False)
    
    # Create bar positions
    bar_positions = np.arange(len(sorted_df))
    
    # Create colormap based on QED values
    cmap = plt.cm.YlGn
    norm = plt.Normalize(vmin=sorted_df['QED'].min(), vmax=sorted_df['QED'].max())
    
    # Create bars with colors based on QED values
    bars = ax2.bar(bar_positions, sorted_df['ADMET_Score'], width=0.6, 
                  color=[cmap(norm(qed)) for qed in sorted_df['QED']])
    
    # Add score values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sorted_df["ADMET_Score"].iloc[i]:.3f}',
                ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # Add molecule IDs as x-tick labels
    plt.xticks(bar_positions, sorted_df['ID'], rotation=45, ha='right', fontsize=22)
    
    # Add y-axis label and title
    plt.ylabel('ADMET Score', fontsize=24, fontweight='bold')
    plt.title('ADMET Scores of Best Molecules', fontsize=25, fontweight='bold')
    
    # Add colorbar for QED values
    cmap = plt.cm.YlGn  # Define colormap for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=sorted_df['QED'].min(), vmax=sorted_df['QED'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('QED Value', fontsize=22, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)
    
    # Adjust y-axis
    plt.ylim(0, 1)
    plt.yticks(fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with high DPI
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    print(f"ADMET profiles saved: {output_path}")
    
    return df

def create_legend_figure(df, output_dir=OUTPUT_DIR, output_filename="molecules_legend.png"):
    """
    Create a separate figure containing just the legend for molecules
    
    Args:
        df: DataFrame containing molecule data
        output_dir: Directory to save the output figure
        output_filename: Filename for the output figure
    """
    # Create figure for legend only
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Hide axes
    ax.axis('off')
    
    # Create dummy lines for legend (without ADMET score)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([0], [0], linewidth=4, label=f"{row['ID']}")
    
    # Create legend with large font size
    legend = ax.legend(loc='center', fontsize=26, title='Molecules', frameon=True)
    plt.setp(legend.get_title(), fontsize=30, fontweight='bold')
    
    # Save figure
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    print(f"Legend figure saved: {output_path}")

def visualize_molecular_structures(df, output_dir=OUTPUT_DIR, output_filename="selected_molecular_structures.png"):
    """
    Create a figure with molecular structures for selected molecules
    
    Args:
        df: DataFrame containing molecule data with SMILES
        output_dir: Directory to save the output figure
        output_filename: Filename for the output figure
    """
    n_molecules = len(df)
    
    # Determine grid layout (approximately square)
    n_cols = int(np.ceil(np.sqrt(n_molecules)))
    n_rows = int(np.ceil(n_molecules / n_cols))
    
    # Create figure with large size
    fig = plt.figure(figsize=(20, 16))
    
    # Sort by ADMET score
    sorted_df = df.sort_values('ADMET_Score', ascending=False)
    
    # Process each molecule
    for idx, (_, row) in enumerate(sorted_df.iterrows()):
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(row['SMILES'])
        
        if mol is not None:
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            ax.axis('off')
            
            # Draw molecule with RDKit
            drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
            drawer.SetFontSize(18)
            
            # Set drawing options
            opts = drawer.drawOptions()
            opts.bondLineWidth = 3
            
            # Draw molecule
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Convert to image
            png_data = drawer.GetDrawingText()
            mol_img = Image.open(io.BytesIO(png_data))
            
            # Display image
            ax.imshow(mol_img)
            
            # Add molecule ID and scores as text with larger font size
            ax.text(0.5, -0.05, f"{row['ID']}", 
                    transform=ax.transAxes, ha='center', fontsize=26, fontweight='bold')
            ax.text(0.5, -0.12, f"ADMET: {row['ADMET_Score']:.3f}, QED: {row['QED']:.3f}", 
                    transform=ax.transAxes, ha='center', fontsize=24)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with high DPI
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()
    
    print(f"Molecular structures saved: {output_path}")

if __name__ == "__main__":
    # Load the selected molecules from Excel file
    excel_path = os.path.join(OUTPUT_DIR, "ADMET selected molecules.xlsx")
    selected_molecules = pd.read_excel(excel_path)
    
    print(f"Loaded {len(selected_molecules)} selected molecules from {excel_path}")
    
    # Create visualizations
    print("\nCreating visualizations for selected molecules...")
    visualize_admet_profiles(selected_molecules)
    visualize_molecular_structures(selected_molecules)
    create_legend_figure(selected_molecules)
    
    print("\nVisualization completed!")

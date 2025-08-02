# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Draw, QED
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import RepeatedKFold
import json
import psutil
import time
from scipy.stats import randint, uniform
import os

# Global constants
MAX_MEMORY = 8000  # 8 GB memory limit
# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output directory created at: {}".format(OUTPUT_DIR))

def create_output_dir(base_dir='.'):
    """Create output directory"""
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Visualization of ADMET analysis results"""
    # Style settings
    plt.style.use('default')
    
    # 1. Molecular Property Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'QED']
    for idx, col in enumerate(cols):
        sns.histplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title('{} Dağılımı'.format(col))
    
    plt.tight_layout()
    output_path = "{}/admet_distributions.png".format(output_dir)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print("Distributions saved: {}".format(os.path.abspath(output_path)))
    plt.close()
    
    # 2. ADMET Scores Correlation Matrix
    admet_scores = ['Absorption_Score', 'BBB_Score', 'Metabolic_Score', 'Toxicity_Score', 'ADMET_Score']
    sns.heatmap(df[admet_scores].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('ADMET Scores Correlation Matrix')
    plt.tight_layout()
    output_path = "{}/admet_correlations.png".format(output_dir)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print("Correlation matrix saved: {}".format(os.path.abspath(output_path)))
    plt.close()
    
    # 3. Drug-Likeliness vs ADMET Scores
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    scores = ['Absorption_Score', 'BBB_Score', 'Metabolic_Score', 'Toxicity_Score']
    for idx, score in enumerate(scores):
        sns.scatterplot(data=df, x='QED', y=score, ax=axes[idx])
        axes[idx].set_title('QED vs {}'.format(score))
    
    plt.tight_layout()
    output_path = "{}/qed_vs_admet.png".format(output_dir)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print("QED vs ADMET graphs saved: {}".format(os.path.abspath(output_path)))
    plt.close()

def visualize_2d_structures(df, n=5, output_dir=OUTPUT_DIR):
    """Visualize the 2D structure of the best n molecules"""
    # Select the best n molecules
    top_mols = df.nlargest(n, 'ADMET_Score')
    
    # Calculate grid dimensions
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    
    # Create subplot grid
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for idx, (_, row) in enumerate(top_mols.iterrows(), 1):
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            plt.subplot(n_rows, n_cols, idx)
            img = Draw.MolToImage(mol)
            plt.imshow(img)
            plt.axis('off')
            plt.title("ID: {}\nADMET Score: {:.3f}\nQED: {:.3f}".format(row['ID'], row['ADMET_Score'], row['QED']))
    
    plt.tight_layout()
    output_path = "{}/top_molecules_2d.png".format(output_dir)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print("2D structures saved: {}".format(os.path.abspath(output_path)))
    plt.close()
    


def visualize_admet_profiles(df, n_molecules=5, output_dir=OUTPUT_DIR, specific_ids=None):
    """Visualize ADMET profiles of top molecules (radar and bar charts only)"""
    # Use specific molecules if provided, otherwise get top n molecules
    if specific_ids is not None:
        top_molecules = df[df['ID'].isin(specific_ids)]
        if len(top_molecules) < len(specific_ids):
            print("Warning: Some specified IDs were not found in the dataset")
            print("Found IDs: {}".format(list(top_molecules['ID'])))
    else:
        top_molecules = df.nlargest(n_molecules, 'ADMET_Score')
    
    # Create figure for ADMET profiles only (radar and bar charts)
    fig = plt.figure(figsize=(24, 14))
    
    # Radar plot (left side)
    ax1 = plt.subplot(121, polar=True)
    
    # Categories for radar plot
    categories = ['Absorption', 'BBB', 'Metabolic', 'Toxicity']
    N = len(categories)
    
    # Angles for radar plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up radar plot
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(0)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.25, 0.5, 0.75])
    ax1.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=22)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=24, fontweight='bold', wrap=True)  
    
    # Plot data on radar chart
    for _, row in top_molecules.iterrows():
        values = [row['Absorption_Score'], row['BBB_Score'], row['Metabolic_Score'], 1 - row['Toxicity_Score']]
        values += values[:1]  # Close the loop
        ax1.plot(angles, values, linewidth=2, linestyle='solid', label=row['ID'])
        ax1.fill(angles, values, alpha=0.1)
    
    # Add legend
    ax1.legend(loc='upper right', fontsize=22)
    ax1.set_title('ADMET Profiles', fontsize=25, fontweight='bold')
    
    # Bar chart (right side)
    ax2 = plt.subplot(122)
    
    # Create bar chart
    bar_positions = np.arange(len(top_molecules))
    bars = ax2.bar(bar_positions, top_molecules['ADMET_Score'], 
                  color=plt.cm.viridis(top_molecules['QED']))
    
    # Add colorbar for QED
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(top_molecules['QED'])
    cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
    cbar.set_label('QED Score', rotation=270, labelpad=20, fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=22)
    
    # Customize bar chart
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels([str(id_) for id_ in top_molecules['ID']], rotation=45, ha='right', fontsize=22)
    ax2.set_ylabel('ADMET Score', fontsize=24, fontweight='bold')  
    ax2.set_title('ADMET Scores of Best Molecules', fontsize=25, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=22)
    
    plt.tight_layout()
    output_path = "{}/admet_profiles.png".format(output_dir)
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    print("ADMET profiles saved: {}".format(os.path.abspath(output_path)))
    plt.close()
    
    # Return the top molecules for use in other visualizations
    return top_molecules


def visualize_molecular_structures(df, n_molecules=5, output_dir=OUTPUT_DIR, specific_ids=None):
    """Visualize molecular structures of top molecules"""
    # Use specific molecules if provided, otherwise get top n molecules
    if specific_ids is not None:
        top_molecules = df[df['ID'].isin(specific_ids)]
        if len(top_molecules) < len(specific_ids):
            print("Warning: Some specified IDs were not found in the dataset")
            print("Found IDs: {}".format(list(top_molecules['ID'])))
    else:
        top_molecules = df.nlargest(n_molecules, 'ADMET_Score')
    
    # Create figure for molecular structures only
    fig = plt.figure(figsize=(20, 16))
    
    # Calculate layout for molecules
    n_mols = len(top_molecules)
    n_cols = min(3, n_mols)  # Maximum 3 molecules per row
    n_rows = (n_mols + n_cols - 1) // n_cols  # Ceiling division
    
    # Draw molecular structures in a grid layout
    for idx, (_, row) in enumerate(top_molecules.iterrows()):
        # Calculate grid position
        col = idx % n_cols
        row_idx = idx // n_cols
        
        # Create subplot for this molecule
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.axis('off')  # Turn off axis
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            # Prepare molecule for drawing - add atom indices and improve rendering
            for atom in mol.GetAtoms():
                atom.SetProp("atomNote", "")
            
            # Draw molecule with better options
            drawer = Draw.MolDraw2DCairo(600, 600)  # Larger size for better quality
            drawer.SetFontSize(18)  # Larger font for atoms
            opts = drawer.drawOptions()
            opts.addAtomIndices = False
            opts.additionalAtomLabelPadding = 0.15  # Add padding around atom labels
            opts.bondLineWidth = 3  # Thicker bonds
            
            # Draw the molecule
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png = drawer.GetDrawingText()
            
            # Convert PNG data to image
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(png))
            
            # Convert to array for matplotlib
            img_array = np.array(img)
            
            # Display the molecule
            ax.imshow(img_array)
            
            # Add molecule info below the structure
            ax.set_title("ID: {}\nADMET Score: {:.3f}\nQED: {:.3f}".format(
                row['ID'], row['ADMET_Score'], row['QED']),
                fontsize=22)
    
    plt.tight_layout()
    output_path = "{}/molecular_structures.png".format(output_dir)
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    print("Molecular structures saved: {}".format(os.path.abspath(output_path)))
    plt.close()

def calculate_molecular_descriptors(smiles):
    """Calculate molecular properties from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'MW': Descriptors.ExactMolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'NumRings': Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'QED': QED.default(mol)
        }
        
        return descriptors
    except Exception as e:
        print("Error: Properties could not be calculated for {} - {}".format(smiles, str(e)))
        return None

def save_results_to_excel(df, output_dir):
    """Save ADMET analysis results to Excel file"""
    print("\nSaving ADMET results to Excel...")
    
    # Create Excel writer
    output_file = os.path.join(output_dir, "admet_analysis_results.xlsx")
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    # Tüm sonuçlar (ADMET skoruna göre sıralı)
    all_results = df.sort_values('ADMET_Score', ascending=False)
    all_results.to_excel(writer, sheet_name='Tüm Sonuçlar', index=False)
    
    # En iyi 50 molekül
    top_50 = df.nlargest(50, 'ADMET_Score')
    top_50.to_excel(writer, sheet_name='En İyi 50', index=False)
    
    # Yüksek QED skoruna sahip moleküller (QED > 0.7)
    high_qed = df[df['QED'] > 0.7].sort_values('ADMET_Score', ascending=False)
    high_qed.to_excel(writer, sheet_name='Yüksek QED (>0.7)', index=False)
    
    # Yüksek BBB skoruna sahip moleküller (BBB > 0.6)
    high_bbb = df[df['BBB_Score'] > 0.6].sort_values('BBB_Score', ascending=False)
    high_bbb.to_excel(writer, sheet_name='Yüksek BBB (>0.6)', index=False)
    
    # Yüksek emilim skoruna sahip moleküller (Absorption > 0.7)
    high_abs = df[df['Absorption_Score'] > 0.7].sort_values('Absorption_Score', ascending=False)
    high_abs.to_excel(writer, sheet_name='Yüksek Emilim (>0.7)', index=False)
    
    # Düşük toksisite skoruna sahip moleküller (Toxicity < 0.3)
    low_tox = df[df['Toxicity_Score'] < 0.3].sort_values('ADMET_Score', ascending=False)
    low_tox.to_excel(writer, sheet_name='Düşük Toksisite (<0.3)', index=False)
    
    # İyi metabolik profile sahip moleküller (Metabolic > 0.6)
    good_met = df[df['Metabolic_Score'] > 0.6].sort_values('Metabolic_Score', ascending=False)
    good_met.to_excel(writer, sheet_name='İyi Metabolizma (>0.6)', index=False)
    
    # İlaç benzeri moleküller (ADMET > 0.6 ve QED > 0.6)
    drug_like = df[(df['ADMET_Score'] > 0.6) & (df['QED'] > 0.6)].sort_values('ADMET_Score', ascending=False)
    drug_like.to_excel(writer, sheet_name='İlaç Benzeri', index=False)
    
    # Her sayfada sütun genişliklerini ayarla
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_len = max(
                series.astype(str).apply(len).max(),  # uzunluk
                len(str(series.name))  # başlık uzunluğu
            ) + 1
            worksheet.set_column(idx, idx, max_len)
    
    # Save Excel file
    writer.close()
    print("Excel file saved: {}".format(output_file))
    
    # Show summary information
    print("\nSummary Information:")
    print("Total number of molecules: {}".format(len(df)))
    print("Number of drug-like molecules (ADMET > 0.6 & QED > 0.6): {}".format(len(drug_like)))
    print("Number of molecules with high QED score (> 0.7): {}".format(len(high_qed)))
    print("Number of molecules with low toxicity (< 0.3): {}".format(len(low_tox)))
    print("Number of molecules with high BBB permeability (> 0.6): {}".format(len(high_bbb)))

def train_and_validate_model(X, y, random_state=42):
    """
    Comprehensive function for model training and validation
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    random_state : int
        Random seed value
        
    Returns:
    --------
    dict
        Model performance metrics and best model
    """
    # Split the dataset (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost hyperparameter search space
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5)
    }
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=-1
    )
    
    # RandomizedSearchCV with 5-fold cross validation
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Model optimization
    search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = search.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(
        best_model, 
        X_train_scaled, 
        y_train, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores)
    
    # Predictions on test set
    y_pred = best_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save model details
    model_details = {
        'best_params': best_model.get_params(),
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'feature_importance': feature_importance,
        'model': best_model,
        'scaler': scaler
    }
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'best_model.joblib')
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model_details

def plot_model_results(model_details, output_dir=OUTPUT_DIR):
    """
    Visualization of model results
    """
    # 1. Feature importance graph
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=model_details['feature_importance'].head(10),
        x='importance',
        y='feature'
    )
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=600)
    plt.close()
    
    # Model performance summary
    print("\nModel Performance Summary:")
    print("Cross-Validation RMSE: {:.4f} ± {:.4f}".format(model_details['cv_rmse_mean'], model_details['cv_rmse_std']))
    print("Test Set RMSE: {:.4f}".format(model_details['test_rmse']))
    print("Test Set R²: {:.4f}".format(model_details['test_r2']))
    print("\nBest Model Parameters:")
    for param, value in model_details['best_params'].items():
        print("{}: {}".format(param, value))

def save_model_report(model_details, output_dir=OUTPUT_DIR):
    """
    Save model details and results as a report
    """
    report = {
        'Model Performance': {
            'Cross-Validation RMSE (mean)': float(model_details['cv_rmse_mean']),
            'Cross-Validation RMSE (std)': float(model_details['cv_rmse_std']),
            'Test Set RMSE': float(model_details['test_rmse']),
            'Test Set R²': float(model_details['test_r2'])
        },
        'Best Parameters': model_details['best_params'],
        'Feature Importance': model_details['feature_importance'].to_dict()
    }
    
    report_path = os.path.join(output_dir, 'model_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print("\nModel report saved: {}".format(report_path))

def calculate_admet_scores(df):
    """Calculate ADMET scores"""
    df['Absorption_Score'] = 0.7 * df['QED'] + 0.3 * (1 - np.minimum(df['TPSA'] / 140, 1))
    df['BBB_Score'] = (0.4 * (1 - np.minimum(df['TPSA'] / 90, 1)) + 
                     0.3 * (1 - np.minimum(df['MW'] / 400, 1)) + 
                     0.3 * (np.minimum(df['LogP'], 4) / 4))
    df['Metabolic_Score'] = (0.4 * df['FractionCSP3'] + 
                           0.3 * (1 - np.minimum(df['AromaticRings'] / 3, 1)) + 
                           0.3 * (1 - np.minimum(abs(df['LogP']), 5) / 5))
    df['Toxicity_Score'] = (0.4 * (np.minimum(df['AromaticRings'], 3) / 3) + 
                          0.3 * (np.minimum(df['LogP'], 5) / 5) + 
                          0.3 * (np.minimum(df['MW'], 500) / 500))
    df['ADMET_Score'] = (df['Absorption_Score'] + df['BBB_Score'] + 
                        df['Metabolic_Score'] + (1 - df['Toxicity_Score'])) / 4
    return df

def train_model(df):
    # Calculate ADMET scores
    print("\nADMET scores are being calculated...")
    df['Absorption_Score'] = 0.7 * df['QED'] + 0.3 * (1 - np.minimum(df['TPSA'] / 140, 1))
    df['BBB_Score'] = (0.4 * (1 - np.minimum(df['TPSA'] / 90, 1)) + 
                     0.3 * (1 - np.minimum(df['MW'] / 400, 1)) + 
                     0.3 * (np.minimum(df['LogP'], 4) / 4))
    df['Metabolic_Score'] = (0.4 * df['FractionCSP3'] + 
                           0.3 * (1 - np.minimum(df['AromaticRings'] / 3, 1)) + 
                           0.3 * (1 - np.minimum(abs(df['LogP']), 5) / 5))
    df['Toxicity_Score'] = (0.4 * (np.minimum(df['AromaticRings'], 3) / 3) + 
                          0.3 * (np.minimum(df['LogP'], 5) / 5) + 
                          0.3 * (np.minimum(df['MW'], 500) / 500))
    df['ADMET_Score'] = (df['Absorption_Score'] + df['BBB_Score'] + 
                        df['Metabolic_Score'] + (1 - df['Toxicity_Score'])) / 4
    
    # Prepare data for model training
    print("\nPreparing data for model training...")
    feature_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'NumRings', 
                   'AromaticRings', 'RotBonds', 'FractionCSP3', 'QED']
    X = df[feature_cols]
    y = df['ADMET_Score']
    
    # Model training and evaluation
    print("\nModel training and evaluation...")
    model_details = train_and_validate_model(X, y, random_state=42)
    
    # Plot model results
    plot_model_results(model_details)
    save_model_report(model_details)
    return df

# Main execution code
if __name__ == "__main__":
    try:
        # Read data
        input_file = "Common-Molecules-Scores-With-SMILES.csv"
        df = pd.read_csv(input_file)
        print("Data successfully read: {}".format(input_file))
        print("\nAvailable columns:")
        print(df.columns.tolist())
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Check SMILES column
        if 'SMILES' not in df.columns:
            raise ValueError("SMILES column not found!")
            
        # Calculate molecular properties
        print("\nCalculating molecular properties...")
        mol_props = []
        for smiles in tqdm(df['SMILES']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                props = {
                    'MW': Descriptors.ExactMolWt(mol),
                    'LogP': Crippen.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'NumRings': Descriptors.RingCount(mol),
                    'AromaticRings': len([ring for ring in Chem.GetSymmSSSR(mol) if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]),
                    'RotBonds': Descriptors.NumRotatableBonds(mol),
                    'FractionCSP3': Descriptors.FractionCSP3(mol),
                    'QED': QED.default(mol)
                }
                mol_props.append(props)
            else:
                mol_props.append({k: 0 for k in ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'NumRings', 'AromaticRings', 'RotBonds', 'FractionCSP3', 'QED']})
                
        # Add properties to DataFrame
        mol_df = pd.concat([df, pd.DataFrame(mol_props)], axis=1)
        
        # Create output directory
        output_dir = create_output_dir()
        
        # Calculate ADMET scores
        print("\nCalculating ADMET scores...")
        mol_df = calculate_admet_scores(mol_df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(mol_df, output_dir)
        visualize_2d_structures(mol_df, n=5, output_dir=output_dir)
        # Create separate visualizations for ADMET profiles and molecular structures
        # First create ADMET profiles visualization (radar/bar charts)
        top_molecules = visualize_admet_profiles(mol_df, n_molecules=5, output_dir=output_dir)
        # Then create molecular structures visualization
        visualize_molecular_structures(mol_df, n_molecules=5, output_dir=output_dir)
        
        # Model training and evaluation
        print("\nStarting model training and evaluation...")
        feature_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'NumRings', 
                       'AromaticRings', 'RotBonds', 'FractionCSP3', 'QED']
        X = mol_df[feature_cols]
        y = mol_df['ADMET_Score']
        
        model_details = train_and_validate_model(X, y, random_state=42)
        plot_model_results(model_details)
        save_model_report(model_details)
        
        # Show the best 5 molecules
        print("\nAnalysis completed! Top 5 molecules:")
        top_mols = mol_df.nlargest(5, 'ADMET_Score')[['ID', 'SMILES', 'ADMET_Score', 'QED', 'MW', 'LogP']]
        print("\nTop 5 molecule properties:")
        print(top_mols.to_string(index=False))
        
        print("\nAll results and visualizations saved in directory: {}".format(os.path.abspath(output_dir)))
        
    except Exception as e:
        print("Error occurred: {}".format(str(e)))
    finally:
        print("\nProgram completed.")
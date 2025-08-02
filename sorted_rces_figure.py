import matplotlib.pyplot as plt
import numpy as np

# Data for the RCES and Evidence Grade
conditions = [
    "Type 2 Diabetes", "Obesity", "Metabolic Syndrome", "Cardiovascular Disease",
    "NAFLD/NASH", "Dyslipidemia", "Hypertension", "Heart Failure",
    "Polycystic Ovary Syndrome", "Kidney Disease", "Neurodegenerative Disease",
    "Alzheimer's Disease", "Parkinson's Disease", "Stroke", "Neuropathy"
]

rces_values = [
    100, 92, 85, 76, 68, 64, 62, 58, 55, 52, 45, 38, 35, 32, 25
]

grades = [
    "Grade A", "Grade A", "Grade A", "Grade B", "Grade B", "Grade B", "Grade C",
    "Grade C", "Grade C", "Grade C", "Grade D", "Grade D", "Grade D", "Grade D", "Grade E"
]

# Combine the data
data = list(zip(conditions, rces_values, grades))

# Sort by RCES values (ascending)
data.sort(key=lambda x: x[1])

# Unpack the sorted data
sorted_conditions, sorted_rces, sorted_grades = zip(*data)

# Create a custom colormap based on RCES values
min_rces = min(sorted_rces)
max_rces = max(sorted_rces)
norm = plt.Normalize(min_rces, max_rces)
colors = plt.cm.RdYlGn(norm(sorted_rces))  # Red-Yellow-Green colormap

# Set up the figure with a larger size
plt.figure(figsize=(14, 12))

# Create the horizontal bar chart
bars = plt.barh(sorted_conditions, sorted_rces, color=colors)

# Add RCES values and grades as text
for i, (rces, grade) in enumerate(zip(sorted_rces, sorted_grades)):
    plt.text(rces + 1, i, str(rces), va='center', fontsize=16, fontweight='bold')
    plt.text(rces + 10, i, grade, va='center', fontsize=16, fontweight='bold')

# Add a vertical reference line for Type 2 Diabetes (100%)
plt.axvline(x=100, color='black', linestyle='--', alpha=0.7)
plt.text(101, len(data) - 1, "Reference (Type 2 Diabetes = 100%)", va='center', fontsize=14)

# Add a colorbar to show the RCES scale
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, orientation='vertical', pad=0.01)
cbar.set_label('RCES Value (%)', fontsize=18)
cbar.ax.tick_params(labelsize=16)

# Set the title and labels with larger font sizes
plt.title('Evidence Grade and RCES for GLP-1 Receptor Agonists by Condition', fontsize=22, pad=20)
plt.xlabel('RCES', fontsize=18)
plt.ylabel('Condition', fontsize=18)

# Increase tick label font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adjust layout
plt.tight_layout()

# Save the figure with high resolution
plt.savefig('sorted_rces_figure.png', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import seaborn as sns

df_latest = pd.read_csv('financity_ux_optimized_dataset.csv')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# New Graph 1: Distribution with New Professional Names
plt.figure()
sns.countplot(data=df_latest, x='Age', hue='Financial_Path', palette='magma')
plt.title('Latest Distribution of Balanced Financial Paths Across Age Groups', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('User Count', fontsize=12)
plt.legend(title='Balanced Path', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('latest_age_vs_financial_path.png', dpi=300)
plt.show()

print("📊 New graphs generated and saved as 'latest_age_vs_financial_path.png'!")
plt.figure()
sns.countplot(data=df_latest, x='Qualification', hue='Financial_Path', palette='magma')
plt.title('Latest Distribution of Balanced Financial Paths Across qualification Groups', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('User Count', fontsize=12)
plt.legend(title='Balanced Path', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('latest_qualification_vs_financial_path.png', dpi=300)
plt.show()

print("📊 New graphs generated and saved as 'latest_qualification_vs_financial_path.png'!")
plt.figure()
sns.countplot(data=df_latest, x='Financial_Goal', hue='Financial_Path', palette='magma')
plt.title('Latest Distribution of Balanced Financial Paths Across financial goal Groups', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('User Count', fontsize=12)
plt.legend(title='Balanced Path', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('financial goal_vs_financial_path.png', dpi=300)
plt.show()

print("📊 New graphs generated and saved as 'latest_goal_vs_financial_path.png'!")


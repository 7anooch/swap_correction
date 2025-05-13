import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data(file_path):
    """
    Load data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_statistics(df):
    """
    Calculate mean and standard deviation for each condition
    """
    stats_df = df.groupby('condition').agg({
        'response_time': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std']
    }).round(3)
    
    return stats_df

def create_visualizations(df):
    """
    Create visualizations for response time and accuracy
    """
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Response Time Box Plot
    sns.boxplot(data=df, x='condition', y='response_time', ax=ax1)
    ax1.set_title('Response Time by Condition')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Response Time (ms)')
    
    # Accuracy Bar Plot
    sns.barplot(data=df, x='condition', y='accuracy', ax=ax2)
    ax2.set_title('Accuracy by Condition')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('swap_correction_analysis.png')
    plt.close()

def perform_statistical_analysis(df):
    """
    Perform statistical tests between conditions
    """
    conditions = df['condition'].unique()
    results = {}
    
    # Perform t-tests for response time
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            cond1 = conditions[i]
            cond2 = conditions[j]
            
            # Response time t-test
            t_stat, p_val = stats.ttest_ind(
                df[df['condition'] == cond1]['response_time'],
                df[df['condition'] == cond2]['response_time']
            )
            
            results[f'{cond1}_vs_{cond2}_rt'] = {
                't_statistic': t_stat,
                'p_value': p_val
            }
            
            # Accuracy t-test
            t_stat, p_val = stats.ttest_ind(
                df[df['condition'] == cond1]['accuracy'],
                df[df['condition'] == cond2]['accuracy']
            )
            
            results[f'{cond1}_vs_{cond2}_acc'] = {
                't_statistic': t_stat,
                'p_value': p_val
            }
    
    return results

def main():
    # Load data
    file_path = 'data/swap_correction_data.csv'
    df = load_data(file_path)
    
    if df is None:
        print("Please make sure the CSV file is in the correct location: data/swap_correction_data.csv")
        return
    
    # Calculate statistics
    stats_df = calculate_statistics(df)
    print("\nDescriptive Statistics:")
    print(stats_df)
    
    # Create visualizations
    create_visualizations(df)
    print("\nVisualizations have been saved as 'swap_correction_analysis.png'")
    
    # Perform statistical analysis
    results = perform_statistical_analysis(df)
    print("\nStatistical Analysis Results:")
    for test, values in results.items():
        print(f"\n{test}:")
        print(f"t-statistic: {values['t_statistic']:.3f}")
        print(f"p-value: {values['p_value']:.3f}")

if __name__ == "__main__":
    main() 
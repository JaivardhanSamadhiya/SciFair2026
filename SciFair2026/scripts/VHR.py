#!/usr/bin/env python3
"""
Filter experimental phage-host infection data for Staphylococcus aureus
This dataset contains both positive (Inf) and negative (NoInf) interactions
"""

import pandas as pd
import sys

def filter_saureus_experimental_data(input_file):
    """
    Filter experimental infection data for S. aureus
    
    Args:
        input_file: Path to the CSV file with infection data
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded file: {input_file}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"\nTotal interactions in dataset: {len(df)}")
    print(f"Column names: {df.columns.tolist()}")
    
    # Show unique hosts
    if 'hostname' in df.columns:
        print(f"\nUnique hosts in dataset: {df['hostname'].nunique()}")
        print(f"First 10 hosts: {df['hostname'].unique()[:10].tolist()}")
    
    # Filter for S. aureus
    saureus_mask = df['hostname'].str.contains(
        'Staphylococcus_aureus', 
        case=False, 
        na=False
    )
    
    saureus_df = df[saureus_mask]
    
    print(f"\n{'='*70}")
    print(f"STAPHYLOCOCCUS AUREUS RESULTS:")
    print(f"{'='*70}")
    
    print(f"\nTotal S. aureus interactions: {len(saureus_df)}")
    
    # Count positive and negative interactions
    if 'infection' in saureus_df.columns:
        infection_counts = saureus_df['infection'].value_counts()
        print(f"\nInfection breakdown:")
        for infection_type, count in infection_counts.items():
            print(f"  {infection_type}: {count}")
        
        # Calculate percentages
        total = len(saureus_df)
        if 'Inf' in infection_counts:
            pos_pct = (infection_counts['Inf'] / total) * 100
            print(f"\nPositive interactions (Inf): {infection_counts.get('Inf', 0)} ({pos_pct:.1f}%)")
        if 'NoInf' in infection_counts:
            neg_pct = (infection_counts['NoInf'] / total) * 100
            print(f"Negative interactions (NoInf): {infection_counts.get('NoInf', 0)} ({neg_pct:.1f}%)")
    
    # Count unique phages
    if 'phagename' in saureus_df.columns:
        unique_phages = saureus_df['phagename'].nunique()
        print(f"\nUnique phages tested against S. aureus: {unique_phages}")
    
    # Show sample of the data
    print(f"\n{'='*70}")
    print("Sample S. aureus interactions:")
    print(f"{'='*70}")
    print(saureus_df.head(10).to_string(index=False))
    
    # Save filtered results
    output_file = 'saureus_experimental_interactions.csv'
    saureus_df.to_csv(output_file, index=False)
    print(f"\n✓ Filtered S. aureus data saved to: {output_file}")
    
    # Additional analysis
    print(f"\n{'='*70}")
    print("DATASET STATISTICS:")
    print(f"{'='*70}")
    
    if 'infection' in saureus_df.columns and 'phagename' in saureus_df.columns:
        # Show which phages infect S. aureus
        infecting_phages = saureus_df[saureus_df['infection'] == 'Inf']['phagename'].unique()
        print(f"\nPhages that infect S. aureus ({len(infecting_phages)}):")
        for i, phage in enumerate(infecting_phages[:20], 1):
            print(f"  {i}. {phage}")
        if len(infecting_phages) > 20:
            print(f"  ... and {len(infecting_phages) - 20} more")
        
        # Show which phages don't infect
        non_infecting_phages = saureus_df[saureus_df['infection'] == 'NoInf']['phagename'].unique()
        print(f"\nPhages that DON'T infect S. aureus ({len(non_infecting_phages)}):")
        for i, phage in enumerate(non_infecting_phages[:10], 1):
            print(f"  {i}. {phage}")
        if len(non_infecting_phages) > 10:
            print(f"  ... and {len(non_infecting_phages) - 10} more")
    
    return saureus_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default filename - adjust as needed
        input_file = "SciFair2026/data/raw/VirusHostInter.csv"
    
    print(f"Processing file: {input_file}\n")
    
    try:
        result = filter_saureus_experimental_data(input_file)
        
        if result is not None and len(result) > 0:
            print(f"\n{'='*70}")
            print("SUMMARY FOR MODEL BUILDING:")
            print(f"{'='*70}")
            if 'infection' in result.columns:
                pos = len(result[result['infection'] == 'Inf'])
                neg = len(result[result['infection'] == 'NoInf'])
                print(f"Positive examples (infections): {pos}")
                print(f"Negative examples (no infection): {neg}")
                print(f"Total S. aureus examples: {pos + neg}")
                if pos > 0:
                    print(f"Negative:Positive ratio: {neg/pos:.2f}:1")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print(f"\nUsage: python {sys.argv[0]} <path_to_csv_file>")
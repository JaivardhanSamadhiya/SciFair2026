#!/usr/bin/env python3
"""
Filter phage-host interaction data for Staphylococcus aureus
"""

import pandas as pd
import sys

def filter_saureus_interactions(input_file):
    """
    Filter phage-host pairs for S. aureus interactions
    
    Args:
        input_file: Path to the phage-bacteria-pairs.txt file
    """
    
    # Try reading the file - common formats are tab-separated or comma-separated
    try:
        # First, try to detect the format by reading first few lines
        with open(input_file, 'r') as f:
            first_line = f.readline()
            
        # Determine separator
        if '\t' in first_line:
            df = pd.read_csv(input_file, sep='\t')
        elif ',' in first_line:
            df = pd.read_csv(input_file, sep=',')
        else:
            # Try whitespace
            df = pd.read_csv(input_file, sep='\s+', engine='python')
            
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Trying alternative reading method...")
        df = pd.read_csv(input_file, sep=None, engine='python')
    
    print(f"Total phage-host pairs in dataset: {len(df)}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Find the column that contains host information
    # Common column names: 'host', 'bacteria', 'Host', 'Bacteria', 'host_name', etc.
    host_column = None
    for col in df.columns:
        if any(term in col.lower() for term in ['host', 'bacteria', 'organism']):
            host_column = col
            break
    
    if host_column is None:
        print("\n⚠️  Could not automatically detect host column.")
        print("Available columns:", df.columns.tolist())
        print("\nPlease specify which column contains the host name:")
        return None
    
    print(f"\n✓ Using '{host_column}' as the host column")
    
    # Filter for S. aureus - check multiple naming variations
    saureus_patterns = [
        'Staphylococcus aureus',
        'S. aureus',
        'S.aureus',
        'Staph aureus',
        'STAPHYLOCOCCUS AUREUS'
    ]
    
    # Create case-insensitive filter
    saureus_mask = df[host_column].astype(str).str.contains(
        'staphylococcus aureus', 
        case=False, 
        na=False
    )
    
    saureus_df = df[saureus_mask]
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Total S. aureus phage-host interactions: {len(saureus_df)}")
    print(f"Percentage of total dataset: {len(saureus_df)/len(df)*100:.2f}%")
    
    # Show some examples
    if len(saureus_df) > 0:
        print(f"\nFirst 10 S. aureus interactions:")
        print(saureus_df.head(10))
        
        # Save filtered results
        output_file = 'saureus_phage_interactions.csv'
        saureus_df.to_csv(output_file, index=False)
        print(f"\n✓ Filtered data saved to: {output_file}")
        
        # Additional statistics
        if 'phage' in df.columns or 'Phage' in df.columns:
            phage_col = 'phage' if 'phage' in df.columns else 'Phage'
            unique_phages = saureus_df[phage_col].nunique()
            print(f"\nUnique phages targeting S. aureus: {unique_phages}")
    else:
        print("\n⚠️  No S. aureus interactions found!")
        print("\nUnique host values (first 20):")
        print(df[host_column].unique()[:20])
    
    return saureus_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default filename
        input_file = "SciFair2026/data/raw/phage-bacteria-pairs.txt"
    
    print(f"Reading file: {input_file}\n")
    
    try:
        result = filter_saureus_interactions(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print(f"\nUsage: python {sys.argv[0]} <path_to_phage-bacteria-pairs.txt>")
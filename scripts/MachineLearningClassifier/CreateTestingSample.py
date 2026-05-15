# import os
# import glob
# import pandas as pd
# RANDOM_STATE = 12

# def create_representative_sample():

#     # load in all data

#     folder_path = '../../local_data/AutomaticProcessingResults'
#     all_files = glob.glob(os.path.join(folder_path, "*.csv"))

#     df_list = []
#     print(f"reading in {len(all_files)} files")
#     for f in all_files:
#         try:
#             df = pd.read_csv(f)
#             if not df.empty:
#                 df_list.append(df)
#         except pd.errors.EmptyDataError:
#             continue
#         except Exception as e:
#             print(f"Error reading {f}: {e}")

#     if not df_list:
#         print("No data found in directory")
#         return
    
#     # create dataframe and clean

#     full_df = pd.concat(df_list, ignore_index=True)
#     print(f"loaded {len(full_df)} total rows")

#     frac_cols = ['sea_ice_frac', 'melt_frac', 'water_frac', 'thin_ice_frac']
#     se_cols   = ['sea_ice_se',   'melt_se',   'water_se',   'thin_ice_se'  ]

#     clean_df_unfiltered = full_df.dropna(subset=frac_cols + se_cols).reset_index(drop=True)
#     selected = []

#     # filter to just rows where
#     # cloud pixels are below 100 pixels
#     # and total pixels are above 600,000 pixels

#     clean_df = clean_df_unfiltered.loc[clean_df_unfiltered['cloud_qa_pixels'] < 100]
#     clean_df = clean_df.loc[clean_df['total_pixels'] > 600000].reset_index(drop=True)
#     clean_df = clean_df.loc[clean_df['coverage_frac'] >= 0.95].reset_index(drop=True)
#     print(f"filtered to {len(clean_df)} rows with 95% coverage, cloud pixels < 100 and total pixels > 600,000")

#     # for each surface cover select 2 random rows with fraction > 50% or top 2 if none

#     print(f"\nfraction > 50% selections")
#     for col in frac_cols:
#         above_50 = clean_df[clean_df[col] > 0.5]
#         print(f"  {col}: {len(above_50)} rows with {col} > 0.5")
        
#         if len(above_50) >= 2:
#             candidates = above_50.sample(n=2, random_state=RANDOM_STATE)
#             print(f"  {col}: randomly sampled from {len(above_50)} rows above 0.5 -> {candidates[col].values.round(3)}")
#         else:
#             candidates = clean_df.nlargest(2, col)
#             print(f"  Warning: only {len(above_50)} row(s) with {col} > 0.5, falling back to top 2: {candidates[col].values.round(3)}")
        
#         selected.append(candidates)

#     # random row from the top and bottom 10th percentile of each SE metric

#     print(f"\nSE top/bottom 10% selections")
#     for col in se_cols:
#         p10 = clean_df[col].quantile(0.10)
#         p90 = clean_df[col].quantile(0.90)

#         print(f"  {col}: 10th percentile={p10:.4f}, 90th percentile={p90:.4f}")

#         low_se  = clean_df[clean_df[col] <= p10].sample(n=1, random_state=RANDOM_STATE)
#         high_se = clean_df[clean_df[col] >= p90].sample(n=1, random_state=RANDOM_STATE)

#         print(f"  {col}: low={low_se[col].values[0]:.4f}")
#         print(f"  {col}: high={high_se[col].values[0]:.4f}")
#         selected.extend([low_se, high_se])

#     # combine and drop duplicates

#     sampled_df = (pd.concat(selected)
#                     .drop_duplicates()
#                     .reset_index(drop=True))

#     print(f"\nfinal sample {len(sampled_df)} / 16 expected rows")

#     out_file = 'sampled_16_points.csv'
#     sampled_df.to_csv(out_file, index=False)
#     print(f"saved to '{out_file}'")

# if __name__ == '__main__':
#     create_representative_sample()

import os
import glob
import pandas as pd
RANDOM_STATE = 12

def create_representative_sample():

    # 1. READ IN VALIDATION DATAFRAMES INSTEAD OF RAW RESULTS
    folder_path = '../../local_data/DataFrames'
    all_files = glob.glob(os.path.join(folder_path, "validation_*.csv"))

    df_list = []
    print(f"Reading in {len(all_files)} validation files")
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df_list.append(df)
        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        print("No validation data found in directory")
        return
    
    # create dataframe and clean
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(full_df)} total matched points")

    # 2. CALCULATE PMW ERRORS
    # Positive error = Visual found more ice than PMW
    # Negative error = PMW found more ice than Visual
    full_df['team_error'] = full_df['visual_ice'] - full_df['team_icecon']
    full_df['bootstrap_error'] = full_df['visual_ice'] - full_df['bootstrap_icecon']

    # 'sea_ice_frac' is now 'visual_ice' from your validation pipeline
    frac_cols  = ['visual_ice', 'melt_frac', 'water_frac', 'thin_ice_frac']
    se_cols    = ['sea_ice_se', 'melt_se',   'water_se',   'thin_ice_se']
    error_cols = ['team_error', 'bootstrap_error']

    # Drop NaNs just to be safe
    clean_df_unfiltered = full_df.dropna(subset=frac_cols + se_cols + error_cols).reset_index(drop=True)
    selected = []

    # Filter to high-quality pixels
    clean_df = clean_df_unfiltered.loc[clean_df_unfiltered['cloud_qa_pixels'] < 100]
    clean_df = clean_df.loc[clean_df['total_pixels'] > 600000].reset_index(drop=True)
    clean_df = clean_df.loc[clean_df['coverage_frac'] >= 0.95].reset_index(drop=True)
    print(f"Filtered to {len(clean_df)} high-quality rows (95% coverage, <100 cloud px, >600k total px)")

    # --- SAMPLE BY FRACTION ---
    print(f"\n--- Fraction > 50% Selections ---")
    for col in frac_cols:
        above_50 = clean_df[clean_df[col] > 0.5]
        print(f"  {col}: {len(above_50)} rows with > 0.5")
        
        if len(above_50) >= 2:
            candidates = above_50.sample(n=2, random_state=RANDOM_STATE).copy()
            print(f"    Randomly sampled 2 rows -> {candidates[col].values.round(3)}")
        else:
            candidates = clean_df.nlargest(2, col).copy()
            print(f"    Warning: Only {len(above_50)} row(s) > 0.5, falling back to top 2 -> {candidates[col].values.round(3)}")
        
        # Add tag: e.g. "high_visual_ice", "high_melt", etc.
        tag_name = f"high_{col.replace('_frac', '')}"
        candidates['edge_case'] = tag_name
        selected.append(candidates)

    # --- SAMPLE BY STANDARD ERROR ---
    print(f"\n--- SE Top/Bottom 10% Selections ---")
    for col in se_cols:
        p10 = clean_df[col].quantile(0.10)
        p90 = clean_df[col].quantile(0.90)

        low_se  = clean_df[clean_df[col] <= p10].sample(n=1, random_state=RANDOM_STATE).copy()
        high_se = clean_df[clean_df[col] >= p90].sample(n=1, random_state=RANDOM_STATE).copy()

        # Add tags: e.g. "low_melt_se", "high_water_se"
        low_se['edge_case'] = f"low_{col}"
        high_se['edge_case'] = f"high_{col}"

        print(f"  {col}: low={low_se[col].values[0]:.4f}, high={high_se[col].values[0]:.4f}")
        selected.extend([low_se, high_se])

    # --- SAMPLE BY PMW ERROR ---
    print(f"\n--- PMW Error Top/Bottom 10% Selections ---")
    for col in error_cols:
        # p10 = Most negative error (PMW highly overestimated compared to visual)
        # p90 = Most positive error (PMW highly underestimated compared to visual)
        p10 = clean_df[col].quantile(0.10)
        p90 = clean_df[col].quantile(0.90)

        low_err  = clean_df[clean_df[col] <= p10].sample(n=1, random_state=RANDOM_STATE).copy()
        high_err = clean_df[clean_df[col] >= p90].sample(n=1, random_state=RANDOM_STATE).copy()

        # Add tags: e.g. "team_overest", "boot_underest"
        prefix = col.split('_')[0] # 'team' or 'bootstrap'
        prefix = 'boot' if prefix == 'bootstrap' else prefix # Shorten bootstrap to boot
        
        low_err['edge_case'] = f"{prefix}_overest"
        high_err['edge_case'] = f"{prefix}_underest"

        print(f"  {col}: PMW Overestimated (low)={low_err[col].values[0]:.4f}, PMW Underestimated (high)={high_err[col].values[0]:.4f}")
        selected.extend([low_err, high_err])

    # combine and drop duplicates based on coordinates/time so we don't get the same pixel twice
    sampled_df = pd.concat(selected).drop_duplicates(subset=['time', 'row', 'col'], keep='first').reset_index(drop=True)

    # Bring edge_case column to the very front so it's the first thing you see in the CSV
    cols = ['edge_case'] + [c for c in sampled_df.columns if c != 'edge_case']
    sampled_df = sampled_df[cols]

    print(f"\nFinal sample size: {len(sampled_df)} unique rows")

    out_file = 'sampled_error_points.csv'
    sampled_df.to_csv(out_file, index=False)
    print(f"Saved to '{out_file}'")

if __name__ == '__main__':
    create_representative_sample()
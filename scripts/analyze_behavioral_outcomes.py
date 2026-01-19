#!/usr/bin/env python
"""Behavioral outcome variables analysis for L2L model data.

This script analyzes behavioral outcomes (score, accept_first, walk_away)
in relation to personality traits for L2L model negotiations.

Usage:
    python scripts/analyze_behavioral_outcomes.py \\
        --input data/simulations/gpt-4o-mini_irp.json \\
        --output-dir output/regression

For KODIS data analysis, use the processed CSV:
    python scripts/analyze_behavioral_outcomes.py \\
        --input data/KODIS/KODIS_H2H_processed.csv \\
        --output-dir output/regression
"""

import os
import sys
import argparse
import json
import pandas as pd
import statsmodels.api as sm
from collections import defaultdict
from typing import Dict


def load_json(file_path: str) -> dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_personality_score(personality: dict) -> dict:
    """Calculate the personality score based on the intensity of adjectives.

    Args:
        personality: Dictionary with personality traits and intensities

    Returns:
        Dictionary of personality scores
    """
    score_dict = {}
    for trait, intensity in personality.items():
        score = 3
        marks = list(intensity.keys())[0]
        adjective = list(intensity.values())[0][0]

        if marks == "High":
            if 'very' in adjective:
                score += 3
            elif 'a bit' in adjective:
                score += 1
            else:
                score += 2
        elif marks == "Low":
            if 'very' in adjective:
                score -= 2
            elif 'a bit' in adjective:
                pass  # score stays 3
            else:
                score -= 1

        score_dict[trait] = score

    return score_dict


def process_l2l_data(data: dict) -> pd.DataFrame:
    """Process L2L model data for regression analysis.

    Args:
        data: L2L simulation data dictionary

    Returns:
        DataFrame with behavioral outcome variables and personality scores
    """
    final = defaultdict(list)

    for idx in range(len(data.get('terminated', []))):
        # Personality scores
        agent1_per, agent2_per = data['personality'][idx]
        agent1_per_score = get_personality_score(agent1_per)
        agent2_per_score = get_personality_score(agent2_per)

        # Position variables (use explicit labels)
        agent1_position = 'seller'
        agent2_position = 'buyer'

        # Scores
        agent1_final_score = data['agent1_final_score'][idx]
        agent2_final_score = data['agent2_final_score'][idx]

        # Walk away first
        walked_away_first = data.get('walked_away_first', [None] * len(data['terminated']))[idx]
        agent1_walk_away_first = walked_away_first == 'Agent1' if walked_away_first else False
        agent2_walk_away_first = walked_away_first == 'Agent2' if walked_away_first else False

        # Accept first
        case = data['case'][idx]
        accept_first = "Agent1" if "Agent1-ACCEPT-DEAL" in case else \
                      "Agent2" if "Agent2-ACCEPT-DEAL" in case else None

        # Case classification
        case_type = "agreement" if "ACCEPT" in case else \
                    "walk-away" if "WALK" in case else "max"

        # Case 1: Self = Agent 1 (Seller)
        final['session_idx'].append(idx)
        final['self_ext'].append(agent1_per_score['EXT'])
        final['self_agr'].append(agent1_per_score['AGR'])
        final['self_con'].append(agent1_per_score['CON'])
        final['self_neu'].append(agent1_per_score['NEU'])
        final['self_ope'].append(agent1_per_score['OPE'])
        final['partner_ext'].append(agent2_per_score['EXT'])
        final['partner_agr'].append(agent2_per_score['AGR'])
        final['partner_con'].append(agent2_per_score['CON'])
        final['partner_neu'].append(agent2_per_score['NEU'])
        final['partner_ope'].append(agent2_per_score['OPE'])
        final['score'].append(agent1_final_score)
        final['case'].append(case_type)
        final['walk_away_first'].append(agent1_walk_away_first)
        final['position'].append(agent1_position)
        final['accept_first'].append('Agent1' == accept_first)

        # Case 2: Self = Agent 2 (Buyer)
        final['session_idx'].append(idx)
        final['self_ext'].append(agent2_per_score['EXT'])
        final['self_agr'].append(agent2_per_score['AGR'])
        final['self_con'].append(agent2_per_score['CON'])
        final['self_neu'].append(agent2_per_score['NEU'])
        final['self_ope'].append(agent2_per_score['OPE'])
        final['partner_ext'].append(agent1_per_score['EXT'])
        final['partner_agr'].append(agent1_per_score['AGR'])
        final['partner_con'].append(agent1_per_score['CON'])
        final['partner_neu'].append(agent1_per_score['NEU'])
        final['partner_ope'].append(agent1_per_score['OPE'])
        final['score'].append(agent2_final_score)
        final['case'].append(case_type)
        final['walk_away_first'].append(agent2_walk_away_first)
        final['position'].append(agent2_position)
        final['accept_first'].append('Agent2' == accept_first)

    # Convert to DataFrame
    df = pd.DataFrame(dict(final))

    # Reverse code walk_away_first to not_walkaway
    df['not_walkaway'] = df['walk_away_first'].astype(int).apply(lambda x: 1 - x)
    df['do_accept_first'] = df['accept_first'].astype(int)
    df['do_agreement'] = df['case'].apply(lambda x: 1 if x == 'agreement' else 0)

    return df


def load_processed_csv(csv_path: str) -> pd.DataFrame:
    """Load processed KODIS CSV file.

    Args:
        csv_path: Path to processed CSV file

    Returns:
        DataFrame with behavioral outcome variables and personality scores
    """
    df = pd.read_csv(csv_path)

    # Rename columns to match L2L format
    column_mapping = {
        'self_extraversion_score': 'self_ext',
        'self_agreeableness_score': 'self_agr',
        'self_conscientiousness_score': 'self_con',
        'self_neuroticism_score': 'self_neu',
        'self_openness_score': 'self_ope',
        'partner_extraversion_score': 'partner_ext',
        'partner_agreeableness_score': 'partner_agr',
        'partner_conscientiousness_score': 'partner_con',
        'partner_neuroticism_score': 'partner_neu',
        'partner_openness_score': 'partner_ope',
    }

    df = df.rename(columns=column_mapping)

    # Add case column for consistency
    df['case'] = df['do_agreement'].apply(lambda x: 'agreement' if x == 1 else 'walk-away')

    return df


def star_sig(p: float) -> str:
    """Return significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def run_regression_analysis(df: pd.DataFrame, model_name: str,
                           output_dir: str, data_type: str = 'model') -> None:
    """Run regression analysis on behavioral outcome variables.

    Args:
        df: Input DataFrame with personality and outcome variables
        model_name: Name of the model for output files
        output_dir: Directory to save output files
        data_type: 'model' or 'kodis'
    """
    os.makedirs(output_dir, exist_ok=True)

    new_df = df.copy()

    # Define personality columns
    personality_cols = [col for col in new_df.columns if col in
                       ['self_ext', 'self_agr', 'self_con', 'self_neu', 'self_ope',
                        'partner_ext', 'partner_agr', 'partner_con', 'partner_neu', 'partner_ope']]

    my_personality = ['self_ext', 'self_agr', 'self_con', 'self_neu', 'self_ope']
    opponent_personality = ['partner_ext', 'partner_agr', 'partner_con', 'partner_neu', 'partner_ope']

    # Center personality variables
    new_df[personality_cols] = new_df[personality_cols].apply(lambda x: x - x.mean())

    # Effective coding for position: buyer → -1, seller → 1
    # This makes the coefficient represent the effect of being seller vs buyer
    position_mapping = {'buyer': -1, 'seller': 1, 0: -1, 1: 1}  # Handle both string and numeric
    new_df['cont_position'] = new_df['position'].map(position_mapping)

    # Define dependent variables
    dv0 = 'score'
    dv1 = 'do_accept_first'
    dv2 = 'not_walkaway'

    # Define experiment sets
    experiment_set = [
        (dv0, ['only_agreement']),
        (dv1, ['only_agreement']),
        (dv2, ['only_walkaway']),
    ]

    # Define models
    models = ['Full model wo/ Interaction']
    model_mapping = {
        'Full model wo/ Interaction': my_personality + opponent_personality + ['cont_position']
    }

    results_list = []
    results_dict = {}

    # Run regression analysis
    for dv, data_cases in experiment_set:
        print("=" * 80)
        print(f"Dependent Variable: {dv}")

        for data in data_cases:
            print(f"Data: {data}")

            for m in models:
                print(f"Model: {m}")

                # Set filtering flags
                filter_agreement_cases = (data == 'only_agreement')
                filter_walkaway_cases = (data == 'only_walkaway')

                # Get independent variables
                independent_vars = model_mapping[m]

                # Drop missing values
                new_df_filtered = new_df.dropna(subset=independent_vars + [dv])
                drop_cnt = len(new_df) - len(new_df_filtered)
                print(f'Dropped {drop_cnt} rows with missing values')

                # Apply case filtering
                if filter_agreement_cases:
                    new_df_filtered = new_df_filtered[new_df_filtered['case'] == 'agreement']
                    print(f'Filtering only agreement cases: {len(new_df_filtered)}')
                elif filter_walkaway_cases:
                    new_df_filtered = new_df_filtered[new_df_filtered['case'] == 'walk-away']
                    print(f'Filtering only walkaway cases: {len(new_df_filtered)}')

                if len(new_df_filtered) == 0:
                    print("No data left after filtering. Skipping.")
                    continue

                # Determine regression model type
                if new_df_filtered[dv].values[0] in [True, False, 0, 1]:
                    new_df_filtered[dv] = new_df_filtered[dv].astype(int)
                    regmodel = sm.Logit
                    reg_type = 'Logistic regression'
                    print("[Regression model] Logistic regression")
                else:
                    regmodel = sm.OLS
                    reg_type = 'Linear regression'
                    print("[Regression model] Linear regression")

                # Add constant and fit model
                X = sm.add_constant(new_df_filtered[independent_vars])
                y = new_df_filtered[dv]

                try:
                    model = regmodel(y, X)
                    results = model.fit()

                    print(results.summary())

                    # Extract summary
                    summary_df = results.summary2().tables[1]
                    p_col = 'P>|z|' if 'P>|z|' in summary_df.columns else 'P>|t|'

                    # Format results
                    summary_df2 = summary_df[["Coef.", p_col]].copy()
                    summary_df2.columns = ["coef", "pvalue"]
                    summary_df2["coef (p-value)"] = summary_df2.apply(
                        lambda row: f"{row['coef']:.2f}{star_sig(row['pvalue'])} ({row['pvalue']:.3f})", axis=1
                    )

                    result_col = summary_df2[["coef (p-value)"]].copy()
                    result_col.columns = [f"{dv}"]
                    result_col.index.name = "variable"
                    results_dict[dv] = result_col

                    # Extract significant variables
                    filtered_df = summary_df[[p_col, 'Coef.']]
                    filtered_df = filtered_df[filtered_df[p_col] < 0.05]

                    print("Significant variables (p < 0.05):")
                    significant_vars = []
                    for var, row in filtered_df.iterrows():
                        if var == 'const' or var == 'cont_position':
                            continue
                        save_str = f"{var}(B={row['Coef.']:.2f}{star_sig(row[p_col])}({row[p_col]:.3f}))"
                        significant_vars.append(save_str)

                    final_vars = " | ".join(significant_vars) if significant_vars else ""
                    results_list.append([dv, data, m, reg_type, final_vars])

                except Exception as e:
                    print(f"Error during regression: {e}")
                    continue

    # Save results
    results_df = pd.DataFrame(results_list, columns=['DVs', 'Data', 'Model', 'Reg_Type', 'Significant_Vars'])

    results_df['Significant_Vars'] = results_df['Significant_Vars'].str.upper()
    results_df['Significant_Vars'] = results_df['Significant_Vars'].apply(lambda x: x.replace('CONT_', ' '))
    results_df['DVs'] = results_df['DVs'].str.upper()
    results_df['DVs'] = results_df['DVs'].apply(lambda x: x.replace('DO_', ' '))

    output_file = os.path.join(output_dir, f'regression_{model_name}_summary.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nRegression summary saved to: {output_file}")

    # Save full results
    if results_dict:
        merged_results = pd.concat(results_dict.values(), axis=1).reset_index()
        merged_results['variable'] = merged_results['variable'].str.upper()
        merged_results['variable'] = merged_results['variable'].apply(lambda x: x.replace('CONT_', ' '))

        merged_results.columns = ['VARIABLE', 'SCORE', 'ACCEPT_FIRST', 'NOT_WALKAWAY']

        full_output_file = os.path.join(output_dir, f'regression_{model_name}_full_results.csv')
        merged_results.to_csv(full_output_file, index=False)
        print(f"Full results saved to: {full_output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze behavioral outcome variables'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON (L2L) or CSV (KODIS) file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/regression',
        help='Directory to save output files'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for output files (auto-detected from input if not specified)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Detect data type and load data
    if args.input.endswith('.csv'):
        print("Loading processed KODIS CSV data...")
        df = load_processed_csv(args.input)
        data_type = 'kodis'
    else:
        print("Loading L2L model JSON data...")
        data = load_json(args.input)
        df = process_l2l_data(data)
        data_type = 'model'

    # Auto-detect model name from input path
    import re
    if not args.model_name:
        # Extract basename and remove suffixes
        basename = os.path.basename(args.input)
        # Remove _irp, _processed, .json, .csv suffixes
        model_name = basename.replace('_irp.json', '').replace('_processed.csv', '')
        model_name = model_name.replace('.json', '').replace('.csv', '')
    else:
        model_name = args.model_name

    print("=" * 80)
    print("Behavioral Outcome Variables Analysis")
    print("=" * 80)
    print(f"Data type: {data_type}")
    print(f"Input file: {args.input}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Observations: {len(df)}")
    print("=" * 80)

    try:
        # Run regression analysis
        print("\nRunning regression analysis...")
        run_regression_analysis(df, model_name, args.output_dir, data_type)

        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

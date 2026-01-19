#!/usr/bin/env python
"""Prepare KODIS H2H data for behavioral outcome analysis.

This script processes the raw KODIS CSV file and creates a clean, analysis-ready dataset.

Usage:
    python scripts/prepare_kodis_data.py \\
        --input data/KODIS/KODIS-human-human-subset.csv \\
        --output data/KODIS/KODIS_H2H_processed.csv
"""

import os
import sys
import argparse
import pandas as pd
from typing import List, Tuple, Optional


def process_chat_data(chat_column: pd.Series, role: str = 'buyer') -> Tuple[List, List, List]:
    """Process KODIS H2H chat data to extract negotiation outcomes.

    Args:
        chat_column: pandas Series containing full chat conversations
        role: 'buyer' or 'seller' to specify perspective

    Returns:
        Tuple of (do_accept_first_list, do_walkaway_list, do_agreement_list)
    """
    do_accept_first_list = []
    do_walkaway_list = []
    do_agreement_list = []

    # Define role identifiers
    if role == 'buyer':
        my_role = 'you'
        other_role = 'other'
    else:  # seller
        my_role = 'other'
        other_role = 'you'

    for conversation in chat_column:
        # Handle missing data
        if isinstance(conversation, (int, float)) or conversation is None:
            do_accept_first_list.append(None)
            do_walkaway_list.append(None)
            do_agreement_list.append(None)
            continue

        if len(str(conversation)) < 2:
            do_accept_first_list.append(None)
            do_walkaway_list.append(None)
            do_agreement_list.append(None)
            continue

        # Parse conversation
        texts = str(conversation).replace("[Other]", "\n[Other]").replace("[You]", "\n[You]").strip()
        sessions = texts.split("\n")

        # Initialize tracking variables
        who_walking_away = None
        who_accept = None
        agreement = None

        # Process each utterance
        for utter in sessions:
            if len(utter) < 1:
                continue

            # Track deal acceptance
            if "Accept Deal" in utter:
                agreement = 1
                if "You" in utter:
                    who_accept = 'you'
                elif "Other" in utter:
                    who_accept = 'other'

            # Track walk away decisions
            if 'I Walk Away.' in utter:
                agreement = 0
                if "You" in utter:
                    who_walking_away = 'you'
                elif "Other" in utter:
                    who_walking_away = 'other'

        # Determine outcomes
        do_accept_first = (who_accept == my_role) if who_accept is not None else None
        do_walkaway = (who_walking_away == my_role) if who_walking_away is not None else None

        do_accept_first_list.append(do_accept_first)
        do_walkaway_list.append(do_walkaway)
        do_agreement_list.append(agreement)

    return do_accept_first_list, do_walkaway_list, do_agreement_list


def process_kodis_data(input_csv: str, output_csv: str) -> None:
    """Process KODIS H2H data and save to analysis-ready format.

    Args:
        input_csv: Path to raw KODIS CSV file
        output_csv: Path to save processed CSV file
    """
    print("=" * 80)
    print("KODIS H2H Data Preparation")
    print("=" * 80)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print("=" * 80)

    # Load raw data
    print("\nLoading raw KODIS data...")
    h2h_data = pd.read_csv(input_csv)
    print(f"Loaded {len(h2h_data)} dyads")

    # Process negotiation outcomes from chat data
    print("\nProcessing buyer chat data...")
    b_accept_first, b_walkaway, b_agreement = process_chat_data(
        h2h_data['b_fullChat'], role='buyer'
    )

    print("Processing seller chat data...")
    s_accept_first, s_walkaway, s_agreement = process_chat_data(
        h2h_data['s_fullChat'], role='seller'
    )

    # Add outcome variables
    h2h_data['b_do_accept_first'] = b_accept_first
    h2h_data['b_do_walkaway'] = b_walkaway
    h2h_data['b_do_agreement'] = b_agreement
    h2h_data['s_do_accept_first'] = s_accept_first
    h2h_data['s_do_walkaway'] = s_walkaway
    h2h_data['s_do_agreement'] = s_agreement

    # Create personality scores from Personality_1 ~ Personality_10
    # Big Five mapping (each trait has 2 items)
    # 1,2: Extraversion, 3,4: Agreeableness, 5,6: Conscientiousness
    # 7,8: Neuroticism, 9,10: Openness

    print("\nComputing personality scores...")

    for prefix in ['b', 's']:
        h2h_data[f'{prefix}_extraversion_score'] = h2h_data[[f'{prefix}_Personality_1', f'{prefix}_Personality_2']].mean(axis=1)
        h2h_data[f'{prefix}_agreeableness_score'] = h2h_data[[f'{prefix}_Personality_3', f'{prefix}_Personality_4']].mean(axis=1)
        h2h_data[f'{prefix}_conscientiousness_score'] = h2h_data[[f'{prefix}_Personality_5', f'{prefix}_Personality_6']].mean(axis=1)
        h2h_data[f'{prefix}_neuroticism_score'] = h2h_data[[f'{prefix}_Personality_7', f'{prefix}_Personality_8']].mean(axis=1)
        h2h_data[f'{prefix}_openness_score'] = h2h_data[[f'{prefix}_Personality_9', f'{prefix}_Personality_10']].mean(axis=1)

    # Create individual participant format (dyadic -> individual)
    print("\nRestructuring data from dyadic to individual format...")

    personality_cols = ['extraversion_score', 'agreeableness_score',
                       'conscientiousness_score', 'neuroticism_score', 'openness_score']
    behavioral_cols = ['do_accept_first', 'do_walkaway', 'do_agreement']

    # Buyer perspective
    b_cols = ([f'b_{col}' for col in personality_cols] +
              [f's_{col}' for col in personality_cols] +
              ['b_points_binary_apol'] +
              [f'b_{col}' for col in behavioral_cols])

    b_data = h2h_data[b_cols].copy()
    b_rename = {f'b_{col}': f'self_{col}' for col in personality_cols + behavioral_cols}
    b_rename.update({f's_{col}': f'partner_{col}' for col in personality_cols})
    b_rename['b_points_binary_apol'] = 'score'
    for col in behavioral_cols:
        b_rename[f'b_{col}'] = col
    b_data = b_data.rename(columns=b_rename)
    b_data['position'] = 'buyer'  # More explicit

    # Seller perspective
    s_cols = ([f's_{col}' for col in personality_cols] +
              [f'b_{col}' for col in personality_cols] +
              ['s_points_binary_apol'] +
              [f's_{col}' for col in behavioral_cols])

    s_data = h2h_data[s_cols].copy()
    s_rename = {f's_{col}': f'self_{col}' for col in personality_cols + behavioral_cols}
    s_rename.update({f'b_{col}': f'partner_{col}' for col in personality_cols})
    s_rename['s_points_binary_apol'] = 'score'
    for col in behavioral_cols:
        s_rename[f's_{col}'] = col
    s_data = s_data.rename(columns=s_rename)
    s_data['position'] = 'seller'  # More explicit

    # Combine perspectives
    restructured_data = pd.concat([b_data, s_data], ignore_index=True)

    # Reverse code do_walkaway to not_walkaway
    restructured_data['not_walkaway'] = (~restructured_data['do_walkaway'].astype(bool)).astype(int)
    restructured_data = restructured_data.drop(columns=['do_walkaway'])

    # Select and order final columns
    final_cols = (
        ['self_extraversion_score', 'self_agreeableness_score', 'self_conscientiousness_score',
         'self_neuroticism_score', 'self_openness_score'] +
        ['partner_extraversion_score', 'partner_agreeableness_score', 'partner_conscientiousness_score',
         'partner_neuroticism_score', 'partner_openness_score'] +
        ['position', 'score', 'do_accept_first', 'do_agreement', 'not_walkaway']
    )

    final_data = restructured_data[final_cols].copy()

    # Save processed data
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_data.to_csv(output_csv, index=False)

    print(f"\nProcessed data saved to: {output_csv}")
    print(f"Final shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total participant observations: {len(final_data)}")
    print(f"Buyers: {sum(final_data['position'] == 0)}")
    print(f"Sellers: {sum(final_data['position'] == 1)}")
    print(f"\nAgreement rate: {final_data['do_agreement'].mean():.2%}")
    print(f"Accept first rate: {final_data['do_accept_first'].mean():.2%}")
    print(f"Not walkaway rate: {final_data['not_walkaway'].mean():.2%}")
    print("=" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare KODIS H2H data for behavioral outcome analysis'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/KODIS/KODIS-human-human-subset.csv',
        help='Path to raw KODIS CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/KODIS/KODIS_H2H_processed.csv',
        help='Path to save processed CSV file'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        process_kodis_data(args.input, args.output)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

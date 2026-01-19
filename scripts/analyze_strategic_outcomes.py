#!/usr/bin/env python
"""Strategic outcome variables analysis for L2L model data.

This script analyzes IRP strategic outcomes (ratio, reciprocity, escalation/descalation)
in relation to personality traits for L2L model negotiations.

Usage:
    python scripts/analyze_strategic_outcomes.py \\
        --input data/simulations/gpt-4o-mini_irp.json \\
        --output-dir output/regression
"""

import os
import sys
import re
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.othermod.betareg as bg
from itertools import product
from typing import Dict, List, Set, Tuple


def get_personality_score(personality: dict) -> dict:
    """Calculate the personality score (1-6 scale) from personality dictionary.

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
                score = 6
            elif 'a bit' in adjective:
                score = 4
            else:
                score = 5
        elif marks == "Low":
            if 'very' in adjective:
                score = 1
            elif 'a bit' in adjective:
                score = 3
            else:
                score = 2

        score_dict[trait] = score

    return score_dict


def calculate_irp_metrics_from_irp2(irp_2: List[Dict], buyer: str = "Agent2", seller: str = "Agent1") -> Dict:
    """Calculate IRP metrics from irp_2 data.

    Args:
        irp_2: List of IRP-annotated utterances
        buyer: Buyer role identifier (e.g., "Agent2", "Speaker1")
        seller: Seller role identifier (e.g., "Agent1", "Speaker2")

    Returns:
        Dictionary with IRP metrics
    """
    # Group utterances by speaker
    speaker1_strategies = []  # seller
    speaker2_strategies = []  # buyer

    current_speaker = None
    current_strategies = []

    for utter in irp_2:
        # Handle different key names for speaker/role
        speaker = utter.get('role') or utter.get('speaker')

        if speaker != current_speaker:
            if current_speaker == buyer:
                speaker2_strategies.append(set(current_strategies))
            elif current_speaker == seller:
                speaker1_strategies.append(set(current_strategies))
            current_speaker = speaker
            current_strategies = []

        # Handle different formats for strategy (string or list)
        strategy_raw = utter['strategy']
        if isinstance(strategy_raw, str):
            # KODIS format: single strategy string
            strategies = [strategy_raw]
        else:
            # L2L format: list of strategies
            strategies = strategy_raw

        # Normalize and add all strategies
        for strategy in strategies:
            normalized = normalize_strategy(strategy)
            current_strategies.append(normalized)

    # Add last speaker's strategies
    if current_speaker == buyer:
        speaker2_strategies.append(set(current_strategies))
    elif current_speaker == seller:
        speaker1_strategies.append(set(current_strategies))

    # Initialize counters
    # Total counts for ratio calculation
    total_coop_buyer, total_comp_buyer, total_neu_buyer, total_res_buyer = 0, 0, 0, 0
    total_coop_seller, total_comp_seller, total_neu_seller, total_res_seller = 0, 0, 0, 0

    # Specific strategy counts for buyer (speaker2)
    total_pos_buyer, total_prop_buyer, total_con_buyer, total_interest_buyer = 0, 0, 0, 0
    total_facts_buyer, total_proc_buyer, total_pow_buyer, total_rights_buyer, total_res_buyer = 0, 0, 0, 0, 0

    # Specific strategy counts for seller (speaker1)
    total_pos_seller, total_prop_seller, total_con_seller, total_interest_seller = 0, 0, 0, 0
    total_facts_seller, total_proc_seller, total_pow_seller, total_rights_seller, total_res_seller = 0, 0, 0, 0, 0

    # Reciprocity and escalation/descalation counters
    reciprocity_coop_buyer, total_coop_buyer = 0, 0
    reciprocity_comp_buyer, total_comp_buyer = 0, 0
    reciprocity_neu_buyer, total_neu_buyer = 0, 0
    reciprocity_coop_seller, total_coop_seller = 0, 0
    reciprocity_comp_seller, total_comp_seller = 0, 0
    reciprocity_neu_seller, total_neu_seller = 0, 0

    escalation_buyer, total_escalation_buyer = 0, 0
    escalation_seller, total_escalation_seller = 0, 0
    descalation_buyer, total_descalation_buyer = 0, 0
    descalation_seller, total_descalation_seller = 0, 0

    # Process buyer utterances (speaker2)
    for i in range(len(speaker2_strategies)):
        strategies = speaker2_strategies[i]
        strategy_type = get_strategy_type(strategies)

        # Count by type
        if "COOPERATIVE" in strategy_type:
            total_coop_buyer += 1
        elif "COMPETITIVE" in strategy_type:
            total_comp_buyer += 1
        elif "NEUTRAL" in strategy_type:
            total_neu_buyer += 1
        elif "RESIDUAL" in strategy_type:
            total_res_buyer += 1

        # Count specific strategies
        if "POS" in strategies:
            total_pos_buyer += 1
        if "PROP" in strategies:
            total_prop_buyer += 1
        if "CON" in strategies:
            total_con_buyer += 1
        if "INTEREST" in strategies:
            total_interest_buyer += 1
        if "FACTS" in strategies:
            total_facts_buyer += 1
        if "PROC" in strategies:
            total_proc_buyer += 1
        if "POWER" in strategies:
            total_pow_buyer += 1
        if "RIGHTS" in strategies:
            total_rights_buyer += 1
        if "RES" in strategies:
            total_res_buyer += 1

    # Process seller utterances (speaker1)
    for i in range(len(speaker1_strategies)):
        strategies = speaker1_strategies[i]
        strategy_type = get_strategy_type(strategies)

        if "COOPERATIVE" in strategy_type:
            total_coop_seller += 1
        elif "COMPETITIVE" in strategy_type:
            total_comp_seller += 1
        elif "NEUTRAL" in strategy_type:
            total_neu_seller += 1
        elif "RESIDUAL" in strategy_type:
            total_res_seller += 1

        if "POS" in strategies:
            total_pos_seller += 1
        if "PROP" in strategies:
            total_prop_seller += 1
        if "CON" in strategies:
            total_con_seller += 1
        if "INTEREST" in strategies:
            total_interest_seller += 1
        if "FACTS" in strategies:
            total_facts_seller += 1
        if "PROC" in strategies:
            total_proc_seller += 1
        if "POWER" in strategies:
            total_pow_seller += 1
        if "RIGHTS" in strategies:
            total_rights_seller += 1
        if "RES" in strategies:
            total_res_seller += 1

    # Calculate reciprocity and escalation/descalation
    # Process buyer <-> seller pairs
    min_len = min(len(speaker1_strategies), len(speaker2_strategies))

    for i in range(min_len):
        # Buyer (speaker2) -> Seller (speaker1)
        anchor = get_strategy_type(speaker2_strategies[i])
        response = get_strategy_type(speaker1_strategies[i])

        anchor_raw = speaker2_strategies[i]
        response_raw = speaker1_strategies[i]

        # Skip NULL utterances
        if anchor == {"NULL"} or response == {"NULL"}:
            continue

        # Count seller's strategies
        if "COOPERATIVE" in anchor:
            total_coop_seller += 1
        if "COMPETITIVE" in anchor:
            total_comp_seller += 1
        if "NEUTRAL" in anchor:
            total_neu_seller += 1

        # Escalation: cooperative/neutral/residual -> competitive
        if "COOPERATIVE" in anchor or "NEUTRAL" in anchor or "RESIDUAL" in anchor:
            total_escalation_seller += 1
            if "COMPETITIVE" in response:
                escalation_buyer += 1

        # Descalation: competitive -> cooperative/neutral/residual
        if "COMPETITIVE" in anchor:
            total_descalation_seller += 1
            if "COOPERATIVE" in response or "NEUTRAL" in response or "RESIDUAL" in response:
                descalation_buyer += 1

        # Specific strategy counts
        for strategy in anchor_raw:
            if strategy == "POS":
                total_pos_seller += 1
            elif strategy == "PROP":
                total_prop_seller += 1
            elif strategy == "CON":
                total_con_seller += 1
            elif strategy == "INTEREST":
                total_interest_seller += 1
            elif strategy == "FACTS":
                total_facts_seller += 1
            elif strategy == "PROC":
                total_proc_seller += 1
            elif strategy == "POWER":
                total_pow_seller += 1
            elif strategy == "RIGHTS":
                total_rights_seller += 1
            elif strategy == "RES":
                total_res_seller += 1

        # Reciprocity
        reciprocated = anchor.intersection(response)
        for strategy in reciprocated:
            if strategy == "COOPERATIVE":
                reciprocity_coop_seller += 1
            elif strategy == "COMPETITIVE":
                reciprocity_comp_seller += 1
            elif strategy == "NEUTRAL":
                reciprocity_neu_seller += 1

        # Seller -> Buyer pair (next turn)
        if i + 1 < len(speaker1_strategies):
            anchor2 = get_strategy_type(speaker1_strategies[i])
            response2 = get_strategy_type(speaker2_strategies[i + 1])
            anchor_raw_2 = speaker1_strategies[i]
            response_raw_2 = speaker2_strategies[i + 1]

            if anchor2 == {"NULL"} or response2 == {"NULL"}:
                continue

            # Count buyer's strategies
            if "COOPERATIVE" in anchor2:
                total_coop_buyer += 1
            if "COMPETITIVE" in anchor2:
                total_comp_buyer += 1
            if "NEUTRAL" in anchor2:
                total_neu_buyer += 1

            # Escalation
            if "COOPERATIVE" in anchor2 or "NEUTRAL" in anchor2 or "RESIDUAL" in anchor2:
                total_escalation_buyer += 1
                if "COMPETITIVE" in response2:
                    escalation_seller += 1

            # Descalation
            if "COMPETITIVE" in anchor2:
                total_descalation_buyer += 1
                if "COOPERATIVE" in response2 or "NEUTRAL" in response2 or "RESIDUAL" in response2:
                    descalation_seller += 1

            # Specific strategy counts
            for strategy in anchor_raw_2:
                if strategy == "POS":
                    total_pos_buyer += 1
                elif strategy == "PROP":
                    total_prop_buyer += 1
                elif strategy == "CON":
                    total_con_buyer += 1
                elif strategy == "INTEREST":
                    total_interest_buyer += 1
                elif strategy == "FACTS":
                    total_facts_buyer += 1
                elif strategy == "PROC":
                    total_proc_buyer += 1
                elif strategy == "POWER":
                    total_pow_buyer += 1
                elif strategy == "RIGHTS":
                    total_rights_buyer += 1
                elif strategy == "RES":
                    total_res_buyer += 1

            # Reciprocity
            reciprocated = anchor2.intersection(response2)
            for strategy in reciprocated:
                if strategy == "COOPERATIVE":
                    reciprocity_coop_buyer += 1
                elif strategy == "COMPETITIVE":
                    reciprocity_comp_buyer += 1
                elif strategy == "NEUTRAL":
                    reciprocity_neu_buyer += 1

    # Calculate percentages
    buyer_total_utterances = len(speaker2_strategies)
    seller_total_utterances = len(speaker1_strategies)

    return {
        # Buyer metrics
        "b_escalation": 100 * escalation_buyer / total_escalation_buyer if total_escalation_buyer > 0 else None,
        "b_descalation": 100 * descalation_buyer / total_descalation_buyer if total_descalation_buyer > 0 else None,
        "b_reciprocity_coop": 100 * reciprocity_coop_buyer / total_coop_buyer if total_coop_buyer > 0 else None,
        "b_reciprocity_comp": 100 * reciprocity_comp_buyer / total_comp_buyer if total_comp_buyer > 0 else None,
        "b_reciprocity_neu": 100 * reciprocity_neu_buyer / total_neu_buyer if total_neu_buyer > 0 else None,
        # Buyer IRP ratios
        "b_coop": 100 * total_coop_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_comp": 100 * total_comp_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_neu": 100 * total_neu_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_res": 100 * total_res_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_pos": 100 * total_pos_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_prop": 100 * total_prop_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_con": 100 * total_con_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_interest": 100 * total_interest_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_facts": 100 * total_facts_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_proc": 100 * total_proc_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_pow": 100 * total_pow_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,
        "b_rights": 100 * total_rights_buyer / buyer_total_utterances if buyer_total_utterances > 0 else 0,

        # Seller metrics
        "s_escalation": 100 * escalation_seller / total_escalation_seller if total_escalation_seller > 0 else None,
        "s_descalation": 100 * descalation_seller / total_descalation_seller if total_descalation_seller > 0 else None,
        "s_reciprocity_coop": 100 * reciprocity_coop_seller / total_coop_seller if total_coop_seller > 0 else None,
        "s_reciprocity_comp": 100 * reciprocity_comp_seller / total_comp_seller if total_comp_seller > 0 else None,
        "s_reciprocity_neu": 100 * reciprocity_neu_seller / total_neu_seller if total_neu_seller > 0 else None,
        # Seller IRP ratios
        "s_coop": 100 * total_coop_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_comp": 100 * total_comp_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_neu": 100 * total_neu_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_res": 100 * total_res_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_pos": 100 * total_pos_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_prop": 100 * total_prop_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_con": 100 * total_con_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_interest": 100 * total_interest_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_facts": 100 * total_facts_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_proc": 100 * total_proc_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_pow": 100 * total_pow_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
        "s_rights": 100 * total_rights_seller / seller_total_utterances if seller_total_utterances > 0 else 0,
    }


def normalize_strategy(strategy: str) -> str:
    """Normalize IRP strategy name."""
    # Handle common variations
    strategy_upper = strategy.upper()

    if "POSITIVE" in strategy_upper or "POSITVE" in strategy_upper:
        return "POS"
    elif "PROPOSAL" in strategy_upper:
        return "PROP"
    elif "CONCESSION" in strategy_upper or "COONCESSION" in strategy_upper or "ROSSESION" in strategy_upper:
        return "CON"
    elif "INTEREST" in strategy_upper:
        return "INTEREST"
    elif "FACTS" in strategy_upper:
        return "FACTS"
    elif "PROCEDURAL" in strategy_upper or "PROCESURAL" in strategy_upper:
        return "PROC"
    elif "POWER" in strategy_upper:
        return "POWER"
    elif "RIGHTS" in strategy_upper:
        return "RIGHTS"
    elif "RESIDUAL" in strategy_upper:
        return "RES"
    else:
        return strategy_upper  # Keep original if not matched


def get_strategy_type(strategies: Set[str]) -> Set[str]:
    """Convert strategies to their types (cooperative, competitive, neutral, residual)."""
    COOPERATIVE = {"INTEREST", "POS"}
    COMPETITIVE = {"POWER", "RIGHTS"}
    NEUTRAL = {"FACTS", "PROC"}
    RESIDUAL = {"RES"}

    strategy_type = set()
    for strategy in strategies:
        if strategy in COOPERATIVE:
            strategy_type.add("COOPERATIVE")
        elif strategy in COMPETITIVE:
            strategy_type.add("COMPETITIVE")
        elif strategy in NEUTRAL:
            strategy_type.add("NEUTRAL")
        elif strategy in RESIDUAL:
            strategy_type.add("RESIDUAL")
        elif strategy == "NULL":
            strategy_type.add("NULL")

    return strategy_type


def process_kodis_data(csv_path: str, irp_annotation_path: str) -> pd.DataFrame:
    """Process KODIS data with IRP annotations for strategic outcome analysis.

    Args:
        csv_path: Path to KODIS CSV file (original or processed)
        irp_annotation_path: Path to IRP annotations JSON file

    Returns:
        DataFrame with personality scores and IRP metrics
    """
    # Load KODIS CSV data
    df_kodis = pd.read_csv(csv_path)

    # Load IRP annotations
    with open(irp_annotation_path, 'r') as f:
        irp_annotations = json.load(f)

    results = []

    for _, row in df_kodis.iterrows():
        # Get dyad IDs
        b_id = str(row.get('b_randomid', ''))
        s_id = str(row.get('s_randomid', ''))

        # Try to find matching IRP annotation
        irp_data = None
        for key in irp_annotations.keys():
            # Extract ID from filename like 'irp_kodis_2527.json' -> '2527'
            kodis_id = key.replace('irp_kodis_', '').replace('.json', '')
            if kodis_id in b_id or kodis_id in s_id:
                irp_data = irp_annotations[key]
                break

        if irp_data is None:
            # No IRP annotation for this dyad, skip
            continue

        # Get personality scores from already processed columns
        # Check if using processed CSV format
        if 'self_extraversion_score' in df_kodis.columns:
            # Already processed - this is individual format, skip for now
            # (should use the behavioral outcomes script instead)
            continue
        else:
            # Original dyadic format - compute personality scores
            b_personality = {
                'EXT': compute_personality_score_from_10(row, 'b'),
                'AGR': compute_personality_score_from_10(row, 'b', agr=True),
                'CON': compute_personality_score_from_10(row, 'b', con=True),
                'NEU': compute_personality_score_from_10(row, 'b', neu=True),
                'OPE': compute_personality_score_from_10(row, 'b', ope=True),
            }
            s_personality = {
                'EXT': compute_personality_score_from_10(row, 's'),
                'AGR': compute_personality_score_from_10(row, 's', agr=True),
                'CON': compute_personality_score_from_10(row, 's', con=True),
                'NEU': compute_personality_score_from_10(row, 's', neu=True),
                'OPE': compute_personality_score_from_10(row, 's', ope=True),
            }

        # Calculate IRP metrics
        # KODIS uses Speaker1/Speaker2 instead of Agent1/Agent2
        # In KODIS: Speaker1 = Buyer, Speaker2 = Seller
        irp_metrics = calculate_irp_metrics_from_irp2(
            irp_data,
            buyer="Speaker1",
            seller="Speaker2"
        )

        # Combine all data
        row_data = {
            'id': f"{b_id}_{s_id}",
            # Buyer personality (speaker1 in KODIS)
            'b_extraversion_score': b_personality['EXT'],
            'b_agreeableness_score': b_personality['AGR'],
            'b_conscientiousness_score': b_personality['CON'],
            'b_neuroticism_score': b_personality['NEU'],
            'b_openness_score': b_personality['OPE'],
            # Seller personality (speaker2 in KODIS)
            's_extraversion_score': s_personality['EXT'],
            's_agreeableness_score': s_personality['AGR'],
            's_conscientiousness_score': s_personality['CON'],
            's_neuroticism_score': s_personality['NEU'],
            's_openness_score': s_personality['OPE'],
        }
        row_data.update(irp_metrics)

        results.append(row_data)

    df = pd.DataFrame(results)
    return df


def compute_personality_score_from_10(row: pd.Series, prefix: str,
                                      agr: bool = False, con: bool = False,
                                      neu: bool = False, ope: bool = False) -> float:
    """Compute personality score from KODIS Personality_1~10 columns.

    KODIS uses 10 questions for Big Five:
    - Items 1,2: Extraversion
    - Items 3,4: Agreeableness
    - Items 5,6: Conscientiousness
    - Items 7,8: Neuroticism
    - Items 9,10: Openness

    Args:
        row: DataFrame row
        prefix: 'b' for buyer, 's' for seller
        agr, con, neu, ope: Flags for specific traits

    Returns:
        Personality score (1-6 scale, or NaN if missing data)
    """
    # Map items to traits
    if not agr and not con and not neu and not ope:
        # Extraversion (items 1,2)
        items = [f'{prefix}_Personality_1', f'{prefix}_Personality_2']
    elif agr:
        # Agreeableness (items 3,4)
        items = [f'{prefix}_Personality_3', f'{prefix}_Personality_4']
    elif con:
        # Conscientiousness (items 5,6)
        items = [f'{prefix}_Personality_5', f'{prefix}_Personality_6']
    elif neu:
        # Neuroticism (items 7,8)
        items = [f'{prefix}_Personality_7', f'{prefix}_Personality_8']
    else:
        # Openness (items 9,10)
        items = [f'{prefix}_Personality_9', f'{prefix}_Personality_10']

    # Get values (1-6 scale in KODIS), filter out NaN
    values = [row.get(item, np.nan) for item in items]
    valid_values = [v for v in values if pd.notna(v)]

    if len(valid_values) == 0:
        return np.nan
    return float(np.mean(valid_values))


def process_l2l_data(data: dict) -> pd.DataFrame:
    """Process L2L model data with IRP annotations for strategic outcome analysis.

    Args:
        data: L2L simulation data with irp_1 and irp_2

    Returns:
        DataFrame with personality scores and IRP metrics
    """
    results = []

    for idx in range(len(data.get('terminated', []))):
        # Personality scores
        agent1_per, agent2_per = data['personality'][idx]
        agent1_per_score = get_personality_score(agent1_per)
        agent2_per_score = get_personality_score(agent2_per)

        # Get IRP metrics from irp_2
        irp_2 = data.get('irp_2', [])
        if idx < len(irp_2):
            irp_metrics = calculate_irp_metrics_from_irp2(
                irp_2[idx],
                buyer="Agent2",
                seller="Agent1"
            )
        else:
            # If no irp_2 data, use defaults
            irp_metrics = {
                'b_escalation': None, 'b_descalation': None,
                'b_reciprocity_coop': None, 'b_reciprocity_comp': None, 'b_reciprocity_neu': None,
                'b_coop': 0, 'b_comp': 0, 'b_neu': 0, 'b_res': 0,
                'b_pos': 0, 'b_prop': 0, 'b_con': 0, 'b_interest': 0, 'b_facts': 0, 'b_proc': 0, 'b_pow': 0, 'b_rights': 0,
                's_escalation': None, 's_descalation': None,
                's_reciprocity_coop': None, 's_reciprocity_comp': None, 's_reciprocity_neu': None,
                's_coop': 0, 's_comp': 0, 's_neu': 0, 's_res': 0,
                's_pos': 0, 's_prop': 0, 's_con': 0, 's_interest': 0, 's_facts': 0, 's_proc': 0, 's_pow': 0, 's_rights': 0,
            }

        # Combine all data
        row = {
            'id': f"{idx}",
            # Buyer personality (agent2)
            'b_extraversion_score': agent2_per_score['EXT'],
            'b_agreeableness_score': agent2_per_score['AGR'],
            'b_conscientiousness_score': agent2_per_score['CON'],
            'b_neuroticism_score': agent2_per_score['NEU'],
            'b_openness_score': agent2_per_score['OPE'],
            # Seller personality (agent1)
            's_extraversion_score': agent1_per_score['EXT'],
            's_agreeableness_score': agent1_per_score['AGR'],
            's_conscientiousness_score': agent1_per_score['CON'],
            's_neuroticism_score': agent1_per_score['NEU'],
            's_openness_score': agent1_per_score['OPE'],
        }
        row.update(irp_metrics)

        results.append(row)

    df = pd.DataFrame(results)
    return df


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for regression analysis by restructuring from dyadic to individual format.

    Args:
        df: DataFrame with buyer/seller personality and IRP metrics

    Returns:
        DataFrame in individual participant format
    """
    # Buyer perspective (self = buyer, partner = seller)
    irp_cols_buyer = [
        'b_escalation', 'b_descalation',
        'b_coop', 'b_comp', 'b_neu', 'b_res',
        'b_pos', 'b_prop', 'b_con', 'b_interest', 'b_facts', 'b_proc', 'b_pow', 'b_rights',
        'b_reciprocity_coop', 'b_reciprocity_comp', 'b_reciprocity_neu'
    ]

    b_cols = ['b_extraversion_score', 'b_agreeableness_score', 'b_conscientiousness_score',
              'b_neuroticism_score', 'b_openness_score']
    s_cols = ['s_extraversion_score', 's_agreeableness_score', 's_conscientiousness_score',
              's_neuroticism_score', 's_openness_score']
    b_data = df[b_cols + s_cols + irp_cols_buyer].copy()

    b_data = b_data.rename(columns={
        'b_extraversion_score': 'self_ext',
        'b_agreeableness_score': 'self_agr',
        'b_conscientiousness_score': 'self_con',
        'b_neuroticism_score': 'self_neu',
        'b_openness_score': 'self_ope',
        's_extraversion_score': 'partner_ext',
        's_agreeableness_score': 'partner_agr',
        's_conscientiousness_score': 'partner_con',
        's_neuroticism_score': 'partner_neu',
        's_openness_score': 'partner_ope',
    })
    b_data = b_data.rename(columns=lambda x: x.replace("b_", "") if x.startswith("b_") else x)
    b_data['position'] = 'buyer'

    # Seller perspective (self = seller, partner = buyer)
    irp_cols_seller = [
        's_escalation', 's_descalation',
        's_coop', 's_comp', 's_neu', 's_res',
        's_pos', 's_prop', 's_con', 's_interest', 's_facts', 's_proc', 's_pow', 's_rights',
        's_reciprocity_coop', 's_reciprocity_comp', 's_reciprocity_neu'
    ]

    s_cols_self = ['s_extraversion_score', 's_agreeableness_score', 's_conscientiousness_score',
                   's_neuroticism_score', 's_openness_score']
    b_cols_partner = ['b_extraversion_score', 'b_agreeableness_score', 'b_conscientiousness_score',
                      'b_neuroticism_score', 'b_openness_score']
    s_data = df[s_cols_self + b_cols_partner + irp_cols_seller].copy()

    s_data = s_data.rename(columns={
        's_extraversion_score': 'self_ext',
        's_agreeableness_score': 'self_agr',
        's_conscientiousness_score': 'self_con',
        's_neuroticism_score': 'self_neu',
        's_openness_score': 'self_ope',
        'b_extraversion_score': 'partner_ext',
        'b_agreeableness_score': 'partner_agr',
        'b_conscientiousness_score': 'partner_con',
        'b_neuroticism_score': 'partner_neu',
        'b_openness_score': 'partner_ope',
    })
    s_data = s_data.rename(columns=lambda x: x.replace("s_", "") if x.startswith("s_") else x)
    s_data['position'] = 'seller'

    # Combine perspectives
    full_df = pd.concat([b_data, s_data], ignore_index=True)

    return full_df


def run_regression_analysis(df: pd.DataFrame, model_name: str, output_dir: str) -> None:
    """Run regression analysis on strategic outcome variables.

    Args:
        df: Input DataFrame with personality and IRP metrics
        model_name: Name of the model for output files
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define variables
    personality_cols = [
        'self_ext', 'self_agr', 'self_con', 'self_neu', 'self_ope',
        'partner_ext', 'partner_agr', 'partner_con', 'partner_neu', 'partner_ope'
    ]

    my_personality = ['self_ext', 'self_agr', 'self_con', 'self_neu', 'self_ope']
    opponent_personality = ['partner_ext', 'partner_agr', 'partner_con', 'partner_neu', 'partner_ope']

    # Center personality variables
    df[personality_cols] = df[personality_cols].apply(lambda x: x - x.mean())

    # Position: buyer -> -1, seller -> +1
    df['cont_position'] = df['position'].map({'buyer': -1, 'seller': 1})

    # Create interaction terms
    for my_type in my_personality:
        df[f'{my_type}*cont_position'] = df[my_type] * df['cont_position']

    # Define dependent variables
    dv_irp_ratios = ["coop", "comp", "neu", "res", "pos", "prop", "con", "interest", "facts", "proc", "pow", "rights"]
    dv_reciprocity = ["reciprocity_coop", "reciprocity_comp", "reciprocity_neu"]
    dv_escalation = ["escalation", "descalation"]
    experiment_set = dv_irp_ratios + dv_reciprocity + dv_escalation

    # Models to test
    models = ['Full model wo Interaction', 'Full model w Interaction']
    model_mapping = {
        'Full model wo Interaction': my_personality + opponent_personality + ['cont_position'],
        'Full model w Interaction': my_personality + opponent_personality + ['cont_position'] + [f'{my_type}*cont_position' for my_type in my_personality],
    }

    # Store results
    results_list = []
    full_results_dict = {}

    # Run regression for each dependent variable
    for dv in experiment_set:
        print("=" * 80)
        print(f"Dependent Variable: {dv}")

        # Filter out None values
        available_dv = df[~df[dv].isna()].index
        if len(available_dv) == 0:
            print(f"  No valid data for {dv}, skipping...")
            continue

        # Use only rows with non-null values for this DV
        df_filtered = df.loc[available_dv].copy()

        for m in models:
            print(f"Model: {m}")

            independent_vars = model_mapping[m]

            # Drop rows with missing values
            df_filtered = df_filtered.dropna(subset=independent_vars)
            print(f"  Observations after filtering: {len(df_filtered)}")

            if len(df_filtered) == 0:
                print("  No data left after filtering, skipping...")
                continue

            # Determine regression type
            # For ratio variables (0-100), use beta regression
            if "ratio" in dv.lower():
                regmodel = lambda y, X: bg.BetaModel(y.apply(lambda x: ((x * (len(y) - 1) + 0.5) / len(y))), X)
                reg_type = "Beta regression"
                print(f"  [Regression model] Beta regression")
            else:
                regmodel = sm.OLS
                reg_type = "Linear regression"
                print(f"  [Regression model] Linear regression")

            # Add constant and fit
            X = sm.add_constant(df_filtered[independent_vars])
            y = df_filtered[dv]

            try:
                if reg_type == "Beta regression":
                    model = regmodel(y, X)
                    results = model.fit()
                    summary_df = results.summary2().tables[1]
                    p_col = 'P>|z|' if 'P>|z|' in summary_df.columns else 'P>|t|'
                else:
                    model = regmodel(y, X)
                    results = model.fit(cov_type='HC3')
                    summary_df = results.summary2().tables[1]
                    p_col = 'P>|z|' if 'P>|z|' in summary_df.columns else 'P>|t|'

                # Print summary
                print(results.summary())

                # Extract significant variables
                filtered_df = summary_df[[p_col, 'Coef.']]
                filtered_df = filtered_df[filtered_df[p_col] < 0.05]

                print("Significant variables (p < 0.05):")
                significant_vars = []
                for var, row in filtered_df.iterrows():
                    if var == 'const':
                        continue
                    stars = get_stars(row[p_col])
                    save_str = f"{var} (B={row['Coef.']:.2f}{stars} ({row[p_col]:.3f}))"
                    significant_vars.append(save_str)

                if not significant_vars:
                    significant_vars = ["No significant variables found."]

                for var in significant_vars:
                    print(f"  {var}")

                # Save to results list
                results_list.append([dv, m, reg_type, "\n".join(significant_vars)])

                # Store full results
                for var, row in summary_df.iterrows():
                    stars = get_stars(row[p_col])
                    coef_str = f"{row['Coef.']:.2f}{stars} ({row[p_col]:.3f})"
                    if var not in full_results_dict:
                        full_results_dict[var] = []
                    full_results_dict[var].append(coef_str)

            except Exception as e:
                print(f"  Error during regression: {e}")
                continue

    # Save summary results
    results_df = pd.DataFrame(results_list, columns=['DV', 'Model', 'Reg_Type', 'Significant_Vars'])
    output_file = os.path.join(output_dir, f'strategic_outcomes_{model_name}_summary.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nSummary saved to: {output_file}")

    # Save full results
    full_results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in full_results_dict.items()]))
    full_output_file = os.path.join(output_dir, f'strategic_outcomes_{model_name}_full.csv')
    full_results_df.to_csv(full_output_file, index=False)
    print(f"Full results saved to: {full_output_file}")


def get_stars(p_value: float) -> str:
    """Return significance stars based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def parse_arguments():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Analyze strategic outcome variables for L2L model or KODIS data'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input file (JSON for L2L model, CSV for KODIS)'
    )

    parser.add_argument(
        '--data-type',
        type=str,
        choices=['model', 'kodis'],
        default=None,
        help='Data type (auto-detected from file extension if not specified)'
    )

    parser.add_argument(
        '--irp-annotations',
        type=str,
        default=None,
        help='Path to IRP annotations JSON file (required for KODIS data)'
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

    # Auto-detect data type from file extension if not specified
    if args.data_type is None:
        if args.input.endswith('.csv'):
            data_type = 'kodis'
        elif args.input.endswith('.json'):
            data_type = 'model'
        else:
            print("Error: Cannot auto-detect data type. Please specify --data-type")
            return 1
    else:
        data_type = args.data_type

    # Auto-detect model name
    if not args.model_name:
        basename = os.path.basename(args.input)
        model_name = basename.replace('_irp.json', '').replace('.json', '')
        model_name = model_name.replace('_processed.csv', '').replace('.csv', '')
    else:
        model_name = args.model_name

    print("=" * 80)
    print("Strategic Outcome Variables Analysis")
    print("=" * 80)
    print(f"Data type: {data_type}")
    print(f"Input file: {args.input}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        # Load and process data
        print("\nLoading and processing data...")

        if data_type == 'kodis':
            # KODIS data processing
            if args.irp_annotations is None:
                print("Error: --irp-annotations is required for KODIS data")
                return 1

            if not os.path.exists(args.irp_annotations):
                print(f"Error: IRP annotations file not found: {args.irp_annotations}")
                return 1

            df = process_kodis_data(args.input, args.irp_annotations)
            print(f"Processed {len(df)} dyad records from KODIS data")

        else:
            # L2L model data processing
            with open(args.input, 'r') as f:
                data = json.load(f)

            # Check if irp_2 exists
            if 'irp_2' not in data or len(data.get('irp_2', [])) == 0:
                print("Error: Input file must contain 'irp_2' annotations")
                print("Please run IRP annotation first: python scripts/annotate_irp.py --input data/simulations/{model}.json")
                return 1

            df = process_l2l_data(data)
            print(f"Processed {len(df)} dyad records from L2L model data")

        if len(df) == 0:
            print("Error: No data records to analyze")
            return 1

        # Prepare for analysis
        analysis_df = prepare_analysis_data(df)
        print(f"Analysis dataset: {len(analysis_df)} participant observations")

        # Run regression
        print("\nRunning regression analysis...")
        run_regression_analysis(analysis_df, model_name, args.output_dir)

        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

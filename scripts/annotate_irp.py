"""IRP annotation script for model conversations and KODIS dataset.

This script annotates conversations with IRP (Interest, Rights, Power) strategies
using GPT-4 with majority voting for quality assurance.

Usage:
    # Annotate model conversations
    python scripts/annotate_irp.py \\
        --input data/gpt-4.1-merged_250_emo.json \\
        --data-type model \\
        --output-dir data/IRP_Annotation/gpt-4.1_annotations

    # Annotate KODIS dataset
    python scripts/annotate_irp.py \\
        --input data/KODIS_combined_dialogues_emo.json \\
        --data-type kodis \\
        --output-dir data/IRP_Annotation/KODIS_annotations
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.irp_annotation import (
    annotate_model_conversations,
    annotate_kodis_conversations,
    merge_model_annotations,
    merge_kodis_annotations
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Annotate conversations with IRP strategies'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input JSON file (emotion-annotated)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (e.g., gpt-4.1-mini) - auto-detects input from data/emotions/'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save individual IRP annotation files (default: data/IRP_Annotation/<model>_annotations)'
    )

    parser.add_argument(
        '--data-type',
        type=str,
        default='model',
        choices=['model', 'kodis'],
        help='Type of data to annotate (model or kodis)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use for annotation (default: gpt-4o)'
    )

    parser.add_argument(
        '--majority-voting',
        type=int,
        default=None,
        help='Number of annotations for majority voting (default: 3 for model, 5 for KODIS)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Skip automatic merge after annotation'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save merged output file (auto-detected from input if not specified)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Auto-detect input file if --model-name is provided
    if args.model_name and not args.input:
        args.input = f"data/emotions/{args.model_name}_emo.json"
        if not os.path.exists(args.input):
            print(f"Error: Emotion-annotated file not found: {args.input}")
            print(f"Please run emotion annotation first: python scripts/annotate_emotions.py --model {args.model_name}")
            return 1

    # Require either --input or --model-name
    if not args.input:
        print("Error: Either --input or --model-name must be specified")
        return 1

    # Auto-set output directory if not provided
    if not args.output_dir:
        if args.model_name:
            args.output_dir = f"data/IRP_Annotation/{args.model_name}_annotations"
        else:
            # Extract model name from input path
            import re
            match = re.search(r'/([\w.-]+)_emo\.json', args.input)
            if match:
                model_name = match.group(1)
                args.output_dir = f"data/IRP_Annotation/{model_name}_annotations"
            else:
                args.output_dir = f"data/IRP_Annotation/annotations"

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Set default majority voting based on data type
    if args.majority_voting is None:
        args.majority_voting = 5 if args.data_type == 'kodis' else 3

    print("=" * 60)
    print("IRP Annotation")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Model: {args.model}")
    print(f"Majority voting: {args.majority_voting}")
    print("=" * 60)

    try:
        if args.data_type == 'model':
            output_dir = annotate_model_conversations(
                input_path=args.input,
                output_dir=args.output_dir,
                model=args.model,
                majority_voting_max=args.majority_voting,
                verbose=args.verbose
            )
        else:  # kodis
            output_dir = annotate_kodis_conversations(
                input_path=args.input,
                output_dir=args.output_dir,
                model=args.model,
                majority_voting_max=args.majority_voting,
                verbose=args.verbose
            )

        print("\n" + "=" * 60)
        print("IRP Annotation Complete!")
        print("=" * 60)
        print(f"Individual annotations saved to: {output_dir}")

        # Auto-merge after annotation (unless --no-merge is specified)
        if not args.no_merge:
            # Auto-detect output file path if not specified
            if not args.output:
                import re
                if args.data_type == 'model':
                    # Extract model name from input path
                    # e.g., data/simulations/gpt-4o-mini.json -> gpt-4o-mini
                    match = re.search(r'/([^/]+)\.json$', args.input)
                    if match:
                        model_name = match.group(1)
                        # Check if input has _emo suffix
                        if '_emo' in model_name:
                            output_file = args.input.replace('_emo.json', '_emo_irp.json')
                        else:
                            output_file = args.input.replace('.json', '_irp.json')
                    else:
                        output_file = args.input.replace('.json', '_irp.json')
                else:  # kodis
                    output_file = args.input.replace('.json', '_irp.json')

                # For KODIS, output in same directory as input
                if args.data_type == 'kodis':
                    input_dir = os.path.dirname(args.input)
                    basename = os.path.basename(args.input).replace('.json', '_irp.json')
                    output_file = os.path.join(input_dir, basename)
            else:
                output_file = args.output

            print("\n" + "=" * 60)
            print("Auto-merging annotations...")
            print("=" * 60)
            print(f"Input file: {args.input}")
            print(f"Annotation directory: {output_dir}")
            print(f"Output file: {output_file}")

            try:
                if args.data_type == 'model':
                    merge_model_annotations(
                        input_data_path=args.input,
                        annotation_dir=output_dir,
                        output_path=output_file,
                        combine_same_speaker=False
                    )
                else:  # kodis
                    merge_kodis_annotations(
                        input_data_path=args.input,
                        annotation_dir=output_dir,
                        output_path=output_file,
                        combine_same_speaker=True
                    )

                print("\n" + "=" * 60)
                print("Merge Complete!")
                print("=" * 60)
                print(f"Merged data saved to: {output_file}")

            except Exception as e:
                print(f"\nWarning: Merge failed: {e}")
                print("You can manually run merge later:")
                if args.data_type == 'model':
                    print(f"  python scripts/merge_irp.py --input {args.input} --annotation-dir {output_dir} --output {output_file} --data-type model")
                else:
                    print(f"  python scripts/merge_irp.py --input {args.input} --annotation-dir {output_dir} --output {output_file} --data-type kodis")
        else:
            print("\nSkipping merge (--no-merge specified)")
            print("You can manually run merge later:")
            if args.data_type == 'model':
                print(f"  python scripts/merge_irp.py --input {args.input} --annotation-dir {output_dir} --data-type model")
            else:
                print(f"  python scripts/merge_irp.py --input {args.input} --annotation-dir {output_dir} --data-type kodis")

        return 0

    except Exception as e:
        print(f"\nError during IRP annotation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

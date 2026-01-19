#!/bin/bash

# Batch IRP annotation script for all model conversations and KODIS dataset
#
# Usage:
#   # Run with default settings
#   ./scripts/annotate_irp_batch.sh
#
#   # Specify custom data directory
#   DATA_DIR=/path/to/data ./scripts/annotate_irp_batch.sh
#
#   # Use different OpenAI model for annotation
#   ANNOTATION_MODEL=gpt-4o-mini ./scripts/annotate_irp_batch.sh

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-data}"
ANNOTATION_MODEL="${ANNOTATION_MODEL:-gpt-4.1-mini}"
ANNOTATION_BASE_DIR="${DATA_DIR}/IRP_Annotation"

# File paths
SIMULATIONS_DIR="${DATA_DIR}/simulations"
KODIS_DIR="${DATA_DIR}/KODIS"
KODIS_FILE="${KODIS_DIR}/KODIS_20_samples.json"

echo "========================================="
echo "Batch IRP Annotation Pipeline"
echo "========================================="
echo "Data directory: ${DATA_DIR}"
echo "Simulations directory: ${SIMULATIONS_DIR}"
echo "KODIS directory: ${KODIS_DIR}"
echo "Annotation model: ${ANNOTATION_MODEL}"
echo "Annotation base directory: ${ANNOTATION_BASE_DIR}"
echo "========================================="

# Create directories
mkdir -p "${ANNOTATION_BASE_DIR}"

# ===========================================
# PART 1: Model Conversations
# ===========================================

# Model names to process
MODELS=("gpt-4.1" "gpt-4.1-mini" "claude-3-7-sonnet-20250219" "gemini-2.0-flash")

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing Model: ${model}"
    echo "========================================="

    # Input file: data/simulations/{model}.json
    INPUT_FILE="${SIMULATIONS_DIR}/${model}.json"

    if [ ! -f "${INPUT_FILE}" ]; then
        echo "  Input file not found: ${INPUT_FILE}, skipping..."
        continue
    fi

    echo "  Input file: ${INPUT_FILE}"

    # IRP Annotation (auto-merge included)
    OUTPUT_DIR="${ANNOTATION_BASE_DIR}/${model}_annotations"
    echo "  Running IRP annotation with auto-merge..."
    echo "    Annotation directory: ${OUTPUT_DIR}"

    python scripts/annotate_irp.py \
        --input "${INPUT_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --data-type model \
        --model "${ANNOTATION_MODEL}"

    if [ $? -ne 0 ]; then
        echo "  ✗ IRP annotation failed for ${model}"
        continue
    fi

    echo "  ✓ Pipeline completed for ${model}"
done

# ===========================================
# PART 2: KODIS Dataset
# ===========================================

echo ""
echo "========================================="
echo "Processing: KODIS Dataset"
echo "========================================="

if [ -f "${KODIS_FILE}" ]; then
    echo "  Input file: ${KODIS_FILE}"

    # IRP Annotation with auto-merge
    OUTPUT_DIR="${ANNOTATION_BASE_DIR}/KODIS_annotations"
    echo "  Running IRP annotation with auto-merge..."
    echo "    Annotation directory: ${OUTPUT_DIR}"

    python scripts/annotate_irp.py \
        --input "${KODIS_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --data-type kodis \
        --model "${ANNOTATION_MODEL}" \
        --majority-voting 5

    if [ $? -ne 0 ]; then
        echo "  ✗ IRP annotation failed for KODIS"
    else
        echo "  ✓ Pipeline completed for KODIS"
    fi
else
    echo "  KODIS file not found: ${KODIS_FILE}, skipping..."
fi

# ===========================================
# Summary
# ===========================================

echo ""
echo "========================================="
echo "Batch IRP Annotation Complete"
echo "========================================="
echo ""
echo "Output files:"
echo "  Model conversations:"
for model in "${MODELS[@]}"; do
    OUTPUT_FILE="${SIMULATIONS_DIR}/${model}_irp.json"
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "    ✓ ${OUTPUT_FILE}"
    fi
done
echo ""
echo "  KODIS dataset:"
KODIS_OUTPUT="${KODIS_DIR}/KODIS_20_samples_irp.json"
if [ -f "${KODIS_OUTPUT}" ]; then
    echo "    ✓ ${KODIS_OUTPUT}"
fi
echo ""
echo "All annotations saved to: ${ANNOTATION_BASE_DIR}/"

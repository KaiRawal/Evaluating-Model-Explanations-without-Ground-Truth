#!/bin/bash

# Master bash script to run all experiments end-to-end
# This script runs all 5 sections of the paper experiments

set -e  # Exit on error

echo "========================================"
echo "Starting complete paper reproduction"
echo "========================================"

# Section 1: Explanation Disagreement (Figure 1)
echo ""
echo "========================================"
echo "Section 1: Explanation Disagreement - Figure 1"
echo "========================================"
cd Sec1-ExplanationDisagreement-Figure1
python3 main.py
cd ..

# Section 2.4: Failures of GT Metrics (Figure 3)
echo ""
echo "========================================"
echo "Section 2.4: Failures of Ground-Truth Metrics - Figure 3"
echo "========================================"
cd Sec2.4-FailuresOfGTMetrics-Figure3
python3 main.py
cd ..

# Section 3.2: Illustrative Axe Example (Figures 4 and 5)
echo ""
echo "========================================"
echo "Section 3.2: Illustrative AXE Example - Figures 4 and 5"
echo "========================================"
cd Sec3.2-IllustrativeAxeExample-Figure4andFigure5
python3 main.py
cd ..

# Section 4.1: Detecting Fairwashing (Table 2)
echo ""
echo "========================================"
echo "Section 4.1: Detecting Explanation Fairwashing - Table 2"
echo "========================================"
cd Sec4.1-DetectingFairwashing-Table2
echo "Running German experiment..."
python3 german_experiment.py
echo "Running COMPAS experiment..."
python3 compas_experiment.py
echo "Running Communities and Crime experiment..."
python3 cc_experiment.py
cd ..

# Section 4.2: Baseline Comparisons (Figures 6 and 7)
echo ""
echo "========================================"
echo "Section 4.2: Baseline Comparisons - Figures 6 and 7"
echo "========================================"
cd Sec4.2-BaselineComparisons-Figure6andFigure7
echo "Generating explanations..."
bash gen_explanations.sh
echo "Generating OpenXAI metrics..."
bash gen_openxai_metrics.sh
echo "Generating AXE metrics..."
bash gen_axe_metrics.sh
echo "Generating plots..."
python3 GenPlots.py
cd ..

echo ""
echo "========================================"
echo "All experiments completed successfully!"
echo "========================================"

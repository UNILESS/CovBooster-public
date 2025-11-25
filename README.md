# CovBooster 🚀

**Paper Title:** _CovBooster: Coverage Booster for Binary Code Clone Detection by Reduced Signatures_

> This work has been accepted for presentation at The 41st ACM/SIGAPP Symposium On Applied Computing (SAC 2026).

This repository contains the implementation and evaluation code for the CovBooster approach, which uses dominating set algorithms to improve binary function detection coverage.

## 📋 Overview

CovBooster is a novel approach for binary function detection that leverages dominating set algorithms to select optimal binary sets for function matching. This method improves precision and recall compared to traditional TLSH-based approaches while maintaining robustness across different threshold values.

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CovBooster-public

# Install dependencies
pip install -r requirements.txt
```

### Data Format

The code expects TLSH hash files organized in the following structure:
```
<db_root>/
├── <binary_group_1>/
│   ├── <binary_1>/
│   │   ├── <function_1>.tlsh
│   │   ├── <function_2>.tlsh
│   │   └── ...
│   └── <binary_2>/
│       └── ...
└── <binary_group_2>/
    └── ...
```

Each `.tlsh` file should contain:
- Line 1: TLSH hash value
- Line 2: Function size (strand size)

**Sample Data**: This repository includes a `sample_data/` directory containing TLSH hash files for testing. The sample data includes:
- 5 binary groups: `bool`, `direvent`, `gmp`, `libcrypto`, `libssl`
- Multiple compiler versions (clang 4.0-7.0, gcc 4.9.4-8.2.0)
- Multiple architectures (arm_32, arm_64, x86_32, x86_64)
- Multiple optimization levels (O0, O1, O2, O3)
- TLSH hash files across multiple binary variants

To use the sample data:
```bash
python3 evaluation_dominating.py sample_data 30 test_results
```

## 📁 Repository Structure

```
Github/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── ds_algo.py                         # Dominating set algorithm implementation
├── dominating_set.py                  # Dominating set construction and evaluation
├── evaluation_dominating.py           # Main evaluation script with dominating set approach
├── threshold_sensitivity_analysis.py   # Threshold sensitivity analysis
├── THRESHOLD_ANALYSIS_README.md       # Detailed threshold analysis documentation
└── sample_data/                       # Sample TLSH hash files for testing
    ├── bool/                           # bool binary group
    ├── direvent/                       # direvent binary group
    ├── gmp/                            # gmp binary group
    ├── libcrypto/                      # libcrypto binary group
    └── libssl/                         # libssl binary group
```

## 🔧 Usage

### 1. Dominating Set Evaluation

Run the main evaluation script with dominating set approach:

```bash
python3 evaluation_dominating.py <db_root> <base_result_directory>
# Example: python3 evaluation_dominating.py sample_data test_output
```

This will:
- Test multiple threshold values (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40)
- Generate results for each threshold automatically
- Save results in timestamped directories under `<base_result_directory>/exp_<timestamp>/threshold_<value>/`

### 2. Threshold Sensitivity Visualization

Generate ROC/PR curves and detailed analysis from evaluation results:

```bash
python3 threshold_sensitivity_analysis.py <exp_dir>
# Example: python3 threshold_sensitivity_analysis.py test_output/exp_20250925_131010
```

This generates:
- ROC and PR curves
- Detailed performance analysis
- Threshold sensitivity results CSV

Generate ROC/PR curves and detailed analysis:

```bash
python3 threshold_sensitivity_analysis.py <exp_dir>
# Example: python3 threshold_sensitivity_analysis.py dominating_results/exp_20250925_131010
```

## 📊 Output Files

### Evaluation Results

For each binary group, the following files are generated:

- `dominating_set_metrics.csv`: Main metrics (Precision, Recall, F1-score, etc.)
- `false_positives.csv`: False positive cases
- `topk_matches.csv`: Top-K matching results
- `grid_search_results.csv`: Grid search parameter optimization results

### Analysis Results

- `threshold_sensitivity_results.csv`: Threshold sensitivity analysis
- `threshold_roc_pr_curves.png`: ROC and PR curves
- `threshold_detailed_analysis.png`: Detailed performance analysis

## 🔬 Key Algorithms

### Dominating Set Algorithm

The core algorithm (`ds_algo.py`) implements a weighted dominating set approach:
- Modified NetworkX algorithm for isolated nodes
- Works on both directed and undirected graphs
- Optimized for function detection graphs

### Evaluation Pipeline

1. **Function Discovery**: Find common functions across binaries
2. **Graph Construction**: Build function similarity graphs
3. **Dominating Set Selection**: Select optimal binary sets
4. **Performance Evaluation**: Calculate Precision, Recall, F1-score, etc.

## 📈 Performance Metrics

The evaluation reports:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Jaccard Similarity**: TP / (TP + FP + FN)
- **NCG (Normalized Coverage Gain)**: TP / Dominating Set Size

## 🎯 Key Features

1. **Robustness**: Less sensitive to TLSH threshold variations
2. **Accuracy**: Improved precision and recall compared to baseline
3. **Efficiency**: Optimized algorithms with multiprocessing support
4. **Comprehensive Analysis**: Detailed metrics and visualizations

## 📝 Citation

If you find CovBooster useful in your research, please cite:

> CovBooster: Coverage Booster for Binary Code Clone Detection by Reduced Signatures.  
> To appear in The 41st ACM/SIGAPP Symposium on Applied Computing (SAC 2026).

## 🔍 Parameters

Key parameters that can be adjusted:

- `TLSH_THRESHOLD`: TLSH similarity threshold (default: 0-40)
- `SIZE_DIFF_THRESHOLD`: Maximum size difference ratio (default: 0.3)

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors**: Reduce the number of parallel processes or use smaller datasets
2. **File not found**: Ensure TLSH data files are in the correct directory structure
3. **Import errors**: Install all required packages from `requirements.txt`

## 📧 Contact

For questions or issues, please open an issue on the repository or contact me by email (jeongwoo@korea.ac.kr).

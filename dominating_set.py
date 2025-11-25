import os
import sys
import csv
import tlsh
import multiprocessing
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from ds_algo import min_weighted_dominating_set

# TLSH similarity comparison threshold
TLSH_THRESHOLD = int(sys.argv[2])

# Result storage directory and file path
result_directory = sys.argv[3]
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
csv_file_path = os.path.join(result_directory, "dominating_set_metrics.csv")

def find_common_functions(directory):
    """Find functions that exist in common across all binaries."""
    print("🔍 Finding common functions...")
    function_sets = []
    for root, _, files in os.walk(directory):
        function_names = {filename.replace('.tlsh', '') for filename in files if filename.endswith('.tlsh')}
        if function_names:
            function_sets.append(function_names)
    common_functions = set.intersection(*function_sets) if function_sets else set()
    print(f"✅ Number of common functions: {len(common_functions)}")
    return list(common_functions)

def load_tlsh_data(directory):
    """Load TLSH data from all binaries once and store in dictionary"""
    print("📥 Loading TLSH data...")
    tlsh_dict = defaultdict(dict)
    for root, _, files in os.walk(directory):
        binary_name = os.path.basename(root)
        for filename in files:
            if filename.endswith(".tlsh"):
                function_name = filename.replace('.tlsh', '')
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    tlsh_hashes = [line.strip() for line in file.readlines()]
                    if tlsh_hashes and tlsh_hashes[0] != "TNULL":
                        tlsh_dict[binary_name][function_name] = tlsh_hashes[0]
    print(f"✅ TLSH data loading complete! Analyzed {len(tlsh_dict)} binaries")
    return tlsh_dict

def compare_function(args):
    """
    Compare detection performance for a specific function, returning TLSH similarity
    comparison results between the function in target_bin and all functions in other binaries.
    Returns: (target_bin, function_name, matches)
      - matches: list [(db_bin, matched_function_name), ...]
    """
    function_name, target_bin, target_tlsh_dict, db_tlsh_dict = args
    matches = []
    if function_name not in target_tlsh_dict.get(target_bin, {}):
        return target_bin, function_name, matches
    target_hash = target_tlsh_dict[target_bin][function_name]
    # Compare with all functions in all other binaries
    for db_bin, db_functions in db_tlsh_dict.items():
        for g, hash_val in db_functions.items():
            diff = tlsh.diff(hash_val, target_hash)
            if diff <= TLSH_THRESHOLD:
                matches.append((db_bin, g))
    return target_bin, function_name, matches

def build_function_graph(function_coverage):
    """
    For each function, construct a graph by adding edges only when
    a TP match with the same name (ground truth) is found between binaries (nodes).
    """
    print("📊 Building function graphs...")
    function_graphs = {}
    for function_name, coverage in function_coverage.items():
        G = nx.Graph()
        G.add_nodes_from(coverage.keys())
        for binary, matches in coverage.items():
            for (db_bin, g) in matches:
                # true positive: matched function name equals ground truth and different binaries
                if g == function_name and binary != db_bin:
                    G.add_edge(binary, db_bin)
        function_graphs[function_name] = G
    print("✅ Function graphs created!")
    return function_graphs

def find_min_dominating_set(function_graphs):
    """Find minimum dominating set for each function graph"""
    print("🔍 Computing minimum dominating sets...")
    dominating_sets = {}
    for function_name, G in tqdm(function_graphs.items(), desc="Finding dominating sets"):
        if len(G.nodes) == 0:
            continue
            dominating_set = min_weighted_dominating_set(G)
            dominating_sets[function_name] = dominating_set
    print("✅ Minimum dominating set analysis complete!")
    return dominating_sets

def evaluate_dominating_set(function_name, dominating_set, function_coverage, all_binaries):
    """
    Evaluate detection performance for the function based on dominating set
      - Aggregate all cross-detection results (matched function names) from binaries in each dominating set
      - True Positive (TP): Matched function name matches ground truth (i.e., function_name)
      - False Positive (FP): Matched function name differs
      - FN: Number of binaries in the set where the function exists but not included in TP
    """
    # Ground truth: set of binaries where the function exists
    function_present_in = set(function_coverage[function_name].keys())
    
    TP_set = set()      # true positive 바이너리 (matched function name == function_name)
    FP_list = []        # false positive: (binary, matched_function_name) where matched_function_name != function_name
    for binary in dominating_set:
        matches = function_coverage[function_name].get(binary, [])
        for (db_bin, g) in matches:
            if g == function_name:
                TP_set.add(db_bin)
            else:
                FP_list.append((db_bin, g))
    
    TP = len(TP_set)
    FP = len(FP_list)
    FN = len(function_present_in - TP_set)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    ncg = TP / len(dominating_set) if len(dominating_set) > 0 else 0

    # Log output: ground truth, TP, FP, FN
    print(f"Function: {function_name}")
    print(f"  Ground Truth (function_present_in): {sorted(function_present_in)}")
    print(f"  True Positives (TP) from dominating set: {sorted(TP_set)}")
    print(f"  False Positives (FP): {FP_list}")
    print(f"  False Negatives (FN): {sorted(function_present_in - TP_set)}")
    
    return precision, recall, f1_score, jaccard, ncg, TP, FP, FN, sorted(TP_set), FP_list, sorted(function_present_in - TP_set)

def main():
    """Dominating set analysis and evaluation (analyze common functions only)"""
    db_root_path = sys.argv[1]
    
    # Find common functions
    common_functions = find_common_functions(db_root_path)
    
    # Load TLSH data from all binaries
    all_tlsh_data = load_tlsh_data(db_root_path)
    all_binaries = set(all_tlsh_data.keys())
    
    # Analyze function-by-function binary matching (cross-comparison) results
    # For each common function, collect matching results between the function in target binary
    # and all functions in all other binaries.
    function_coverage = defaultdict(lambda: defaultdict(list))
    tasks = [
        (function_name, target_bin, {target_bin: all_tlsh_data[target_bin]},
         {db_bin: funcs for db_bin, funcs in all_tlsh_data.items() if db_bin != target_bin})
        for function_name in common_functions for target_bin in all_tlsh_data.keys()
    ]
    
    print("⚡ Comparing functions...")
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(compare_function, tasks, chunksize=20), total=len(tasks), desc="Comparing"))
    
    for target_bin, function_name, matches in results:
        function_coverage[function_name][target_bin] = matches
    
    # Build function graphs (using TP only)
    function_graphs = {}
    print("📊 Building function graphs...")
    for function_name, coverage in function_coverage.items():
        G = nx.Graph()
        # Nodes are all binaries (ground truth)
        G.add_nodes_from(coverage.keys())
        for binary, matches in coverage.items():
            for (db_bin, g) in matches:
                if g == function_name and binary != db_bin:
                    G.add_edge(binary, db_bin)
        function_graphs[function_name] = G
    print("✅ Function graphs created!")
    
    # Find minimum dominating sets
    dominating_sets = find_min_dominating_set(function_graphs)
    
    # Evaluate and save dominating set results (including FN and FP lists)
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Function", "Dominating Set Size", "Precision", "Recall", "F1-score",
                             "Jaccard Similarity", "NCG", "TP", "FP", "FN", "TP_Binaries", "FP_Details", "Undetected_Binaries"])
        for function_name, dominating_set in dominating_sets.items():
            precision, recall, f1_score, jaccard, ncg, TP, FP, FN, TP_binaries, FP_details, undetected = \
                evaluate_dominating_set(function_name, dominating_set, function_coverage, all_binaries)
            csv_writer.writerow([function_name, len(dominating_set), f"{precision:.4f}", f"{recall:.4f}",
                                 f"{f1_score:.4f}", f"{jaccard:.4f}", f"{ncg:.4f}", TP, FP, FN,
                                 ",".join(TP_binaries), ";".join([f"{b}:{g}" for b, g in FP_details]), ",".join(undetected)])
    
    print(f"\n✅ Dominating set results saved! {csv_file_path}")

if __name__ == "__main__":
    main()
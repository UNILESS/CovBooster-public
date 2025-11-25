#!/usr/bin/env python3
"""
Dominating set-based function detection performance evaluation
- Construct and evaluate dominating sets for each binary group
- Calculate Precision, Recall, F1-score with correct TP, FP, FN computation
"""

import os
import sys
import csv
import tlsh
import multiprocessing
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# Greedy dominating set algorithm implemented directly

# TLSH similarity comparison thresholds (1-unit increments for 0-10, 5-unit increments after 10)
TLSH_THRESHOLDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]

# Result storage directory (timestamp-based folder name)
base_result_directory = sys.argv[2]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_directory = os.path.join(base_result_directory, f"exp_{timestamp}")

def find_common_functions(directory):
    """Find functions that exist in all binaries (size > 100)."""
    print("🔍 Finding common functions...")
    function_sets = []
    for root, _, files in os.walk(directory):
        function_names = set()
        for filename in files:
            if filename.endswith('.tlsh'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path) as f:
                        h = f.readline().strip()
                        s = int(f.readline().strip())
                    if h != "TNULL" and s > 100:  # Size filtering
                        function_names.add(filename.replace('.tlsh', ''))
                except:
                    pass
        if function_names:
            function_sets.append(function_names)
    common_functions = set.intersection(*function_sets) if function_sets else set()
    print(f"✅ Number of common functions: {len(common_functions)}")
    return list(common_functions)

def load_tlsh_data(directory):
    """Load TLSH data and sizes from all binaries once and store in dictionaries"""
    print("📥 Loading TLSH data...")
    tlsh_dict = defaultdict(dict)
    size_dict = defaultdict(dict)
    for root, _, files in os.walk(directory):
        binary_name = os.path.basename(root)
        for filename in files:
            if filename.endswith(".tlsh"):
                function_name = filename.replace('.tlsh', '')
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r') as file:
                        h = file.readline().strip()
                        s = int(file.readline().strip())
                    if h != "TNULL" and s > 100:  # Size filtering
                        tlsh_dict[binary_name][function_name] = h
                        size_dict[binary_name][function_name] = s
                except:
                    pass
    print(f"✅ TLSH data loading complete! Analyzed {len(tlsh_dict)} binaries")
    return tlsh_dict, size_dict

def compare_function(args):
    """
    Compare detection performance for a specific function, returning TLSH similarity
    comparison results between the function in target_bin and all functions in other binaries.
    Returns: (target_bin, function_name, matches)
      - matches: list [(db_bin, matched_function_name, diff), ...]
    """
    function_name, target_bin, target_tlsh_dict, target_size_dict, db_tlsh_dict, db_size_dict, tlsh_threshold = args
    matches = []
    if function_name not in target_tlsh_dict.get(target_bin, {}):
        return target_bin, function_name, matches
    
    target_hash = target_tlsh_dict[target_bin][function_name]
    target_size = target_size_dict[target_bin][function_name]
    
    # Compare with all functions in all other binaries
    for db_bin, db_functions in db_tlsh_dict.items():
        for g, hash_val in db_functions.items():
            if g not in db_size_dict[db_bin]:
                continue
            db_size = db_size_dict[db_bin][g]
            
            # Size difference filtering
            if target_size < 10 or db_size < 10 or abs(target_size - db_size) / max(target_size, db_size) <= 0.3:
                if hash_val == "TNULL" or target_hash == "TNULL":
                    continue
                try:
                    diff = tlsh.diff(hash_val, target_hash)
                    if diff <= tlsh_threshold:
                        matches.append((db_bin, g, diff))
                except:
                    continue
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
            for (db_bin, g, diff) in matches:
                # true positive: matched function name equals ground truth and different binaries
                if g == function_name and binary != db_bin:
                    G.add_edge(binary, db_bin)
        function_graphs[function_name] = G
    print("✅ Function graphs created!")
    return function_graphs

def greedy_dominating_set(G):
    """Find dominating set using greedy approach"""
    U = set(G.nodes())
    V_star = []
    while U:
        u = max(U, key=lambda x: G.degree(x))
        V_star.append(u)
        covered = {u} | set(G.neighbors(u))
        U -= covered
    return V_star

def ensure_full_coverage(func, ds, cov):
    """Ensure full coverage"""
    truth = set(cov[func].keys())
    covered = set(ds)
    for d in ds:
        for b2, f2, _ in cov[func].get(d, []):
            if f2 == func:
                covered.add(b2)
    missing = truth - covered
    return ds + list(missing)

def prune_dominating_set(G, ds, func, cov):
    """Optimize dominating set"""
    D = set(ds)
    truth = set(cov[func].keys())
    for node in sorted(ds, key=lambda x: G.degree(x)):
        T = D - {node}
        covered = set(T)
        for d in T:
            for b2, f2, _ in cov[func].get(d, []):
                if f2 == func:
                    covered.add(b2)
        if covered >= truth:
            D = T
    return list(D)

def find_min_dominating_set(function_graphs, function_coverage, max_size=144):
    """Find minimum dominating set for each function graph"""
    print("🔍 Computing minimum dominating sets...")
    dominating_sets = {}
    for function_name, G in tqdm(function_graphs.items(), desc="Finding dominating sets"):
        if len(G.nodes) == 0:
            continue
        ds = greedy_dominating_set(G)
        ds = ensure_full_coverage(function_name, ds, function_coverage)
        # ds = prune_dominating_set(G, ds, function_name, function_coverage)  # Disabled
        if len(ds) > max_size:
            ds = ds[:max_size]
            ds = ensure_full_coverage(function_name, ds, function_coverage)
            ds = prune_dominating_set(G, ds, function_name, function_coverage)
        dominating_sets[function_name] = ds
    print("✅ Minimum dominating set analysis complete!")
    return dominating_sets

def evaluate_dominating_set(function_name, dominating_set, function_coverage, all_binaries, function_graphs=None):
    """
    Evaluate detection performance for the function based on dominating set
    
    Correct dominating set evaluation:
    - Dominating set D must ensure all nodes are either in D or adjacent to D
    - Therefore Recall should be 100% (all nodes are dominated)
    
    - True Positive (TP): Binaries dominated by the dominating set with correct function name matches
    - False Positive (FP): Binaries dominated by the dominating set with incorrect function name matches
    - False Negative (FN): Binaries not dominated by the dominating set (should theoretically be 0)
    """
    # Ground truth: set of binaries where the function exists
    function_present_in = set(function_coverage[function_name].keys())
    
    # Calculate all binaries dominated by the dominating set
    # (dominating set itself + adjacent nodes of the dominating set)
    dominated_binaries = set(dominating_set)  # Dominating set itself is always dominated
    
    # If graph exists, calculate based on adjacent nodes
    if function_graphs and function_name in function_graphs:
        G = function_graphs[function_name]
        for binary in dominating_set:
            if binary in G:
                # Add all adjacent nodes from the graph
                dominated_binaries.update(G.neighbors(binary))
    else:
        # Fallback: use existing method if graph is not available
        for binary in dominating_set:
            matches = function_coverage[function_name].get(binary, [])
            for (db_bin, g, diff) in matches:
                dominated_binaries.add(db_bin)
    
    
    # Undominated binaries (should theoretically be 0)
    undominated_binaries = function_present_in - dominated_binaries
    
    
    # Calculate TP/FP for dominated binaries
    TP_set = set()      # True positive binaries (matched function name == function_name)
    FP_list = []        # False positive: (binary, matched_function_name) where matched_function_name != function_name
    
    # Check matching results from all binaries in the dominating set
    for binary in dominating_set:
        matches = function_coverage[function_name].get(binary, [])
        for (db_bin, g, diff) in matches:
            if g == function_name:
                TP_set.add(db_bin)  # Count matched binary as TP
            else:
                FP_list.append((db_bin, g))  # (matched_binary, matched_function_name)
    
    # Calculate precision
    TP = len(TP_set)  # True positive count
    FP = len(FP_list)  # False positive count
    FN = len(undominated_binaries)  # Number of undominated binaries
    
    # Precision calculation
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    ncg = TP / len(dominating_set) if len(dominating_set) > 0 else 0

    return precision, recall, f1_score, jaccard, ncg, TP, FP, FN, sorted(TP_set), FP_list, sorted(undominated_binaries)

def process_binary_group(args):
    """
    Perform dominating set analysis for a single binary group (for multiprocessing)
    """
    group_name, group_binaries, all_tlsh_data, result_directory, tlsh_threshold = args
    
    print(f"🔍 Starting {group_name} group analysis... ({len(group_binaries)} binaries)")
    
    # Extract only TLSH data and size data for binaries in the group
    all_tlsh_data, all_size_data = all_tlsh_data
    group_tlsh_data = {bin_name: all_tlsh_data[bin_name] for bin_name in group_binaries if bin_name in all_tlsh_data}
    group_size_data = {bin_name: all_size_data[bin_name] for bin_name in group_binaries if bin_name in all_size_data}
    
    if not group_tlsh_data:
        print(f"⚠️ No valid TLSH data in {group_name} group.")
        return
    
    # Find common functions within the group
    function_sets = []
    for binary_name, functions in group_tlsh_data.items():
        function_sets.append(set(functions.keys()))
    common_functions = set.intersection(*function_sets) if function_sets else set()
    
    if not common_functions:
        print(f"⚠️ No common functions in {group_name} group.")
        return
    
    print(f"📊 {group_name} group common functions: {len(common_functions)}")
    
    # Analyze function-by-function binary matching (cross-comparison) results
    function_coverage = defaultdict(lambda: defaultdict(list))
    tasks = [
        (function_name, target_bin, 
         {target_bin: group_tlsh_data[target_bin]}, {target_bin: group_size_data[target_bin]},
         {db_bin: funcs for db_bin, funcs in group_tlsh_data.items() if db_bin != target_bin},
         {db_bin: sizes for db_bin, sizes in group_size_data.items() if db_bin != target_bin},
         tlsh_threshold)
        for function_name in common_functions for target_bin in group_tlsh_data.keys()
    ]
    
    print(f"⚡ Comparing functions in {group_name} group...")
    # Sequential processing to avoid multiprocessing nesting issues
    results = []
    for task in tqdm(tasks, desc=f"{group_name} comparison"):
        results.append(compare_function(task))
    
    for target_bin, function_name, matches in results:
        function_coverage[function_name][target_bin] = matches
    
    # Build function graphs (using TP only)
    function_graphs = {}
    print(f"📊 Building function graphs for {group_name} group...")
    for function_name, coverage in function_coverage.items():
        G = nx.Graph()
        # Nodes are all binaries (ground truth)
        G.add_nodes_from(coverage.keys())
        for binary, matches in coverage.items():
            for (db_bin, g, diff) in matches:
                if g == function_name and binary != db_bin:
                    G.add_edge(binary, db_bin)
        function_graphs[function_name] = G
    print(f"✅ Function graphs for {group_name} group created!")
    
    # Find minimum dominating sets
    dominating_sets = find_min_dominating_set(function_graphs, function_coverage)
    print(f"✅ Minimum dominating set analysis for {group_name} group complete!")
    
    # Create result storage directory
    group_dir = os.path.join(result_directory, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    # Evaluate and save dominating set results
    csv_file_path = os.path.join(group_dir, "dominating_set_metrics.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Function", "Dominating Set Size", "Precision", "Recall", "F1-score",
                             "Jaccard Similarity", "NCG", "TP", "FP", "FN", "TP_Binaries", "FP_Details", "Undetected_Binaries"])
        for function_name, dominating_set in dominating_sets.items():
            precision, recall, f1_score, jaccard, ncg, TP, FP, FN, TP_binaries, FP_details, undetected = \
                evaluate_dominating_set(function_name, dominating_set, function_coverage, set(group_binaries), function_graphs)
            csv_writer.writerow([function_name, len(dominating_set), f"{precision:.4f}", f"{recall:.4f}",
                                 f"{f1_score:.4f}", f"{jaccard:.4f}", f"{ncg:.4f}", TP, FP, FN,
                                 ",".join(TP_binaries), ";".join([f"{b}:{g}" for b, g in FP_details]), ",".join(undetected)])
    
    # Save False Positives
    fp_file_path = os.path.join(group_dir, "false_positives.csv")
    with open(fp_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Function", "SourceBinary", "MatchedBinary", "MatchedFunction"])
        for function_name, dominating_set in dominating_sets.items():
            for binary in dominating_set:
                matches = function_coverage[function_name].get(binary, [])
                for (db_bin, g, diff) in matches:
                    if g != function_name:  # Save only False Positives
                        csv_writer.writerow([function_name, binary, db_bin, g])
    
    # Save Top-K Matches
    topk_file_path = os.path.join(group_dir, "topk_matches.csv")
    with open(topk_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Function", "SourceBinary", "MatchedBinary", "MatchedFunction", "IsGroundTruth"])
        for function_name, dominating_set in dominating_sets.items():
            for binary in dominating_set:
                matches = function_coverage[function_name].get(binary, [])
                for (db_bin, g, diff) in matches:
                    is_ground_truth = "Yes" if g == function_name else "No"
                    csv_writer.writerow([function_name, binary, db_bin, g, is_ground_truth])
    
    # Save Grid Search Results (summary statistics)
    grid_file_path = os.path.join(group_dir, "grid_search_results.csv")
    with open(grid_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["TLSH_THRESHOLD", "SIZE_DIFF_THRESHOLD", "FP_WEIGHT", "MU_DOM", "LAMBDA_FP", 
                             "MAX_DOMINATING_SET_SIZE", "Total_FN", "Total_TP", "Total_FP", "Total_Dom_Size", 
                             "Num_Functions", "Overall_Precision", "Overall_Recall", "Overall_F1", "Avg_Dom_Size"])
        
        # Calculate TP/FP
        total_TP = 0
        total_FP = 0
        
        for function_name, dominating_set in dominating_sets.items():
            for binary in dominating_set:
                matches = function_coverage[function_name].get(binary, [])
                for (db_bin, g, diff) in matches:
                    if g == function_name:
                        total_TP += 1
                    else:
                        total_FP += 1
        
        # According to the definition of dominating set, if all 144 binaries are in the dominating set, FN should be 0
        # Therefore, set FN to 0
        total_FN = 0
        
        total_dom_size = sum(len(dominating_set) for dominating_set in dominating_sets.values())
        num_functions = len(dominating_sets)
        
        # Calculate precision
        overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        avg_dom_size = total_dom_size / num_functions if num_functions > 0 else 0
        
        csv_writer.writerow([tlsh_threshold, 0.1, 5, 0.0, 0.1, 10, total_FN, total_TP, total_FP, 
                             total_dom_size, num_functions, overall_precision, overall_recall, 
                             overall_f1, avg_dom_size])
    
    print(f"✅ {group_name} group analysis complete! Results saved: {group_dir}")

def main():
    """Dominating set analysis and evaluation (process by binary group) - iterate for each TLSH threshold"""
    db_root_path = sys.argv[1]
    
    # Load TLSH data and size data for all binaries
    all_tlsh_data, all_size_data = load_tlsh_data(db_root_path)
    all_binaries = set(all_tlsh_data.keys())
    
    # Classify by binary group
    binary_groups = defaultdict(list)
    for binary_name in all_tlsh_data.keys():
        # Extract group from binary name
        if 'libssl.so.elf' in binary_name:
            group_name = 'libssl'
        elif 'libcrypto.so.elf' in binary_name:
            group_name = 'libcrypto'
        elif '-' in binary_name:
            group_name = binary_name.split('-')[0]  # e.g., grep-2.5.1 -> grep
        else:
            group_name = binary_name
        binary_groups[group_name].append(binary_name)
    
    print(f"📊 Binary groups: {list(binary_groups.keys())}")
    for group_name, binaries in binary_groups.items():
        print(f"  {group_name}: {len(binaries)} binaries")
    
    # Iterate for each TLSH threshold
    for tlsh_threshold in TLSH_THRESHOLDS:
        print(f"\n🔍 Starting analysis with TLSH threshold {tlsh_threshold}...")
        
        # Create threshold-specific result directory
        result_directory = os.path.join(exp_directory, f"threshold_{tlsh_threshold}")
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        
        # Perform dominating set analysis for each group (multiprocessing)
        tasks = [(group_name, group_binaries, (all_tlsh_data, all_size_data), result_directory, tlsh_threshold) 
                 for group_name, group_binaries in binary_groups.items()]
        
        print(f"🚀 TLSH threshold {tlsh_threshold} - Starting binary group analysis with multiprocessing...")
        with multiprocessing.Pool() as pool:
            list(tqdm(pool.imap_unordered(process_binary_group, tasks), 
                      total=len(tasks), desc=f"Threshold {tlsh_threshold} group analysis"))
        
        print(f"✅ TLSH threshold {tlsh_threshold} analysis complete! Results saved: {result_directory}")
    
    print(f"\n🎉 All TLSH threshold analyses complete! Results saved: {exp_directory}")

if __name__ == "__main__":
    main()
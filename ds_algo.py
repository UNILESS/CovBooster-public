# Slightly modified version of min_weighted_dominating_set algorithm of Networkx
# so that it can now perform well on graphs with isolated nodes
# also it can now work on directed graphs

# For original algorithm, See: https://networkx.org/documentation/stable/_modules/networkx/algorithms/approximation/dominating_set.html#min_weighted_dominating_set

def min_weighted_dominating_set(G, weight=None):
    if len(G) == 0:
        return set()
    dom_set = set()

    def _cost(node_and_neighborhood):
        v, neighborhood = node_and_neighborhood
        return G.nodes[v].get(weight, 1) / len(neighborhood - dom_set)

    vertices = set(G)
    neighborhoods = {v: {v} | set(G[v]) for v in G}

    while vertices:
        dom_node, min_set = min(neighborhoods.items(), key=_cost)
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set

        # Edited part: update neighborhoods
        neighborhoods = {v: {v} | set(G[v]) for v in vertices}

    return dom_set

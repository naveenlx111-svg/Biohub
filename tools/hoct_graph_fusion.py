"""Label-free, deterministic fusion of HOCT forks with a frozen baseline graph."""
from collections import Counter, defaultdict


def fuse_hoct_forks(nodes, edges, pairs, allow_reparent=False):
    children = defaultdict(set)
    proposed = defaultdict(set)
    incoming = {}
    for edge in edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        children[source].add(target)
        assert target not in incoming, "Baseline has multiple parents"
        incoming[target] = source
    for source, target in pairs:
        assert source in nodes and target in nodes
        assert int(nodes[target]["t"]) == int(nodes[source]["t"]) + 1
        proposed[source].add(target)
    candidates = []
    for source, targets in proposed.items():
        if len(targets) == 2 and len(children[source]) == 1 and children[source] <= targets:
            target = next(iter(targets - children[source]))
            candidates.append((source, target))
    # HOCT is already an ILP solution with one parent per child. Stable ordering
    # makes conflict handling independent of dataframe row order.
    result = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    edits = []
    for source, target in sorted(candidates):
        previous = incoming.get(target)
        if previous is not None:
            if not allow_reparent or len(children[previous]) != 1:
                continue
            result.remove((previous, target))
            children[previous].remove(target)
        result.add((source, target))
        children[source].add(target)
        incoming[target] = source
        edits.append(dict(source_id=source, target_id=target, previous_source_id=previous))
    indegree = Counter(t for s, t in result)
    outdegree = Counter(s for s, t in result)
    assert max(indegree.values(), default=0) <= 1
    assert max(outdegree.values(), default=0) <= 2
    return [dict(source_id=s, target_id=t) for s, t in sorted(result)], edits

"""Tests for group-disjoint split invariants to guarantee zero data leakage."""

from collections import defaultdict


def partition_group_disjoint(items, n_folds=5):
    """Reference implementation of deterministic group-disjoint K-fold partitioning."""
    groups = defaultdict(list)
    for idx, (item_id, group_id) in enumerate(items):
        groups[group_id].append((idx, item_id))

    sorted_groups = sorted(groups.keys(), key=lambda g: (len(groups[g]), str(g)), reverse=True)

    fold_groups = [set() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for g in sorted_groups:
        min_fold = min(range(n_folds), key=lambda i: fold_sizes[i])
        fold_groups[min_fold].add(g)
        fold_sizes[min_fold] += len(groups[g])

    splits = []
    for test_fold in range(n_folds):
        test_g = fold_groups[test_fold]
        train_g = set().union(*(fold_groups[f] for f in range(n_folds) if f != test_fold))
        splits.append((train_g, test_g))

    return splits


def test_no_group_leakage():
    # Simulate 50 observations across 10 logical groups
    synthetic_dataset = []
    for g in range(10):
        # Each group has multiple angle variants (simulating H -> RZ(theta) -> CX)
        num_variants = 3 + (g % 5)
        for v in range(num_variants):
            synthetic_dataset.append((f"obs_{g}_{v}", f"logical_group_{g}"))

    splits = partition_group_disjoint(synthetic_dataset, n_folds=5)
    assert len(splits) == 5

    for fold_idx, (train_groups, test_groups) in enumerate(splits):
        # INVARIANT 1: Disjointness
        overlap = train_groups.intersection(test_groups)
        assert len(overlap) == 0, f"Fold {fold_idx} leaked groups: {overlap}"

        # INVARIANT 2: Coverage
        total_groups = train_groups.union(test_groups)
        assert len(total_groups) == 10, f"Fold {fold_idx} missing groups"
        assert len(test_groups) > 0, f"Fold {fold_idx} has empty test groups"

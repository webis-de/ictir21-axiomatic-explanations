def fixed_ranges(i1, i2):
    # Note: i1 < i2 holds.
    step_ranks = {
        0, 10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
        100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
    return (
            i2 < 20 or (i1 in step_ranks and i2 in step_ranks) or
            (i1 < 10 and i2 > 90 and i2 < 100) or
            (i1 < 10 and i2 > 490 and i2 < 500) or
            (i1 < 10 and i2 > 990 and i2 < 1000) or
            (i1 > 90 and i1 < 100 and i2 > 490 and i2 < 500) or
            (i1 > 90 and i1 < 100 and i2 > 990 and i2 < 1000) or
            (i1 > 490 and i1 < 500 and i2 > 990 and i2 < 1000)
    )


def mk_random_sampler(rnk_len: int, target_n: int, seed: int = 1):
    """Returns a pair sampler that selects pairs at random with uniform probability,
    such that approximately _target_n_ samples will be retained when applied to all pairs
    of a ranking of length _rnk_len_.

    Note that the frequency distribution of pairwise rank differences is linearly biased
    towards close neighbors, and this bias is retained by the uniform sampling.
    """
    import numpy as np
    npairs = rnk_len ** 2 / 2 - rnk_len
    if target_n >= npairs:
        return lambda a, b: True
    prob = target_n / npairs
    rnd = np.random.RandomState(seed=seed)
    return lambda a, b: rnd.rand() <= prob


def mk_topk_random_sampler(topk: int, rnk_len: int, target_n_rand: int, seed=1):
    rs = mk_random_sampler(rnk_len, target_n_rand, seed)
    return lambda i1, i2: (i1 <= topk and i2 <= topk) or rs(i1, i2)

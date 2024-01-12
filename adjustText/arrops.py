# Code copied from bioframe.core.arrops
import numpy as np

def arange_multi(starts, stops=None, lengths=None):
    """
    Create concatenated ranges of integers for multiple start/length.

    Parameters
    ----------
    starts : numpy.ndarray
        Starts for each range
    stops : numpy.ndarray
        Stops for each range
    lengths : numpy.ndarray
        Lengths for each range. Either stops or lengths must be provided.

    Returns
    -------
    concat_ranges : numpy.ndarray
        Concatenated ranges.

    Notes
    -----
    See the following illustrative example:

    starts = np.array([1, 3, 4, 6])
    stops = np.array([1, 5, 7, 6])

    print arange_multi(starts, lengths)
    >>> [3 4 4 5 6]

    From: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop

    """

    if (stops is None) == (lengths is None):
        raise ValueError("Either stops or lengths must be provided!")

    if lengths is None:
        lengths = stops - starts

    if np.isscalar(starts):
        starts = np.full(len(stops), starts)

    # Repeat start position index length times and concatenate
    cat_start = np.repeat(starts, lengths)

    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(
        lengths.cumsum() - lengths, lengths
    )

    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter

    return cat_range

def overlap_intervals(starts1, ends1, starts2, ends2, closed=False, sort=False):
    """
    Take two sets of intervals and return the indices of pairs of overlapping intervals.

    Parameters
    ----------
    starts1, ends1, starts2, ends2 : numpy.ndarray
        Interval coordinates. Warning: if provided as pandas.Series, indices
        will be ignored.

    closed : bool
        If True, then treat intervals as closed and report single-point overlaps.
    Returns
    -------
    overlap_ids : numpy.ndarray
        An Nx2 array containing the indices of pairs of overlapping intervals.
        The 1st column contains ids from the 1st set, the 2nd column has ids
        from the 2nd set.

    """

    # Convert to numpy arrays
    starts1 = np.asarray(starts1)
    ends1 = np.asarray(ends1)
    starts2 = np.asarray(starts2)
    ends2 = np.asarray(ends2)

    # Concatenate intervals lists
    n1 = len(starts1)
    n2 = len(starts2)
    ids1 = np.arange(0, n1)
    ids2 = np.arange(0, n2)

    # Sort all intervals together
    order1 = np.lexsort([ends1, starts1])
    order2 = np.lexsort([ends2, starts2])
    starts1, ends1, ids1 = starts1[order1], ends1[order1], ids1[order1]
    starts2, ends2, ids2 = starts2[order2], ends2[order2], ids2[order2]

    # Find interval overlaps
    match_2in1_starts = np.searchsorted(starts2, starts1, "left")
    match_2in1_ends = np.searchsorted(starts2, ends1, "right" if closed else "left")
    # "right" is intentional here to avoid duplication
    match_1in2_starts = np.searchsorted(starts1, starts2, "right")
    match_1in2_ends = np.searchsorted(starts1, ends2, "right" if closed else "left")

    # Ignore self-overlaps
    match_2in1_mask = match_2in1_ends > match_2in1_starts
    match_1in2_mask = match_1in2_ends > match_1in2_starts
    match_2in1_starts, match_2in1_ends = (
        match_2in1_starts[match_2in1_mask],
        match_2in1_ends[match_2in1_mask],
    )
    match_1in2_starts, match_1in2_ends = (
        match_1in2_starts[match_1in2_mask],
        match_1in2_ends[match_1in2_mask],
    )

    # Generate IDs of pairs of overlapping intervals
    overlap_ids = np.block(
        [
            [
                np.repeat(ids1[match_2in1_mask], match_2in1_ends - match_2in1_starts)[
                    :, None
                ],
                ids2[arange_multi(match_2in1_starts, match_2in1_ends)][:, None],
            ],
            [
                ids1[arange_multi(match_1in2_starts, match_1in2_ends)][:, None],
                np.repeat(ids2[match_1in2_mask], match_1in2_ends - match_1in2_starts)[
                    :, None
                ],
            ],
        ]
    )

    if sort:
        # Sort overlaps according to the 1st
        overlap_ids = overlap_ids[np.lexsort([overlap_ids[:, 1], overlap_ids[:, 0]])]

    return overlap_ids
from __future__ import division, annotations

from .arrops import overlap_intervals
from ._version import __version__

from functools import lru_cache
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import get_path_collection_extents
import scipy.spatial.distance
from logging import getLogger
from timeit import default_timer as timer
import io

logger = getLogger(__name__)

try:
    from matplotlib.backend_bases import _get_renderer as matplot_get_renderer
except ImportError:
    matplot_get_renderer = None


# Modified from https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
def intersect(seg1, seg2):
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return False
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return False
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return False
    return True


def get_renderer(fig):
    # If the backend support get_renderer() or renderer, use that.
    if hasattr(fig.canvas, "get_renderer"):
        return fig.canvas.get_renderer()

    if hasattr(fig.canvas, "renderer"):
        return fig.canvas.renderer

    # Otherwise, if we have the matplotlib function available, use that.
    if matplot_get_renderer:
        return matplot_get_renderer(fig)

    # No dice, try and guess.
    # Write the figure to a temp location, and then retrieve whichever
    # render was used (doesn't work in all matplotlib versions).
    fig.canvas.print_figure(io.BytesIO())
    try:
        return fig._cachedRenderer

    except AttributeError:
        # No luck.
        # We're out of options.
        raise ValueError("Unable to determine renderer") from None


def get_bboxes_pathcollection(sc, ax):
    """Function to return a list of bounding boxes in display coordinates
    for a scatter plot
    Thank you to ImportanceOfBeingErnest
    https://stackoverflow.com/a/55007838/1304161"""
    #    ax.figure.canvas.draw() # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]] * len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]] * len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t], [o], transOffset.frozen()
            )
            bboxes.append(result.transformed(ax.transData.inverted()))

    return bboxes


def get_bboxes(objs, r=None, expand=(1, 1), ax=None):
    """


    Parameters
    ----------
    objs : list, or PathCollection
        List of objects to get bboxes from. Also works with mpl PathCollection.
    r : renderer
        Renderer. The default is None, then automatically deduced from ax.
    expand : (float, float), optional
        How much to expand bboxes in (x, y), in fractions. The default is (1, 1).
    ax : Axes, optional
        The default is None, then uses current axes.

    Returns
    -------
    list
        List of bboxes.

    """
    ax = ax or plt.gca()
    r = r or get_renderer(ax.get_figure())
    try:
        objs = [i.get_bbox() for i in objs]
    except (AttributeError, TypeError):
        pass

    try:
        return [i.get_window_extent(r).expanded(*expand) for i in objs]
    except (AttributeError, TypeError):
        try:
            if all([isinstance(obj, matplotlib.transforms.BboxBase) for obj in objs]):
                return objs
            else:
                raise ValueError("Something is wrong")
        except TypeError:
            return get_bboxes_pathcollection(objs, ax)


def get_2d_coordinates(objs, ax):
    bboxes = get_bboxes(objs, get_renderer(ax.get_figure()), (1.0, 1.0), ax)
    xs = [
        (ax.convert_xunits(bbox.xmin), ax.convert_yunits(bbox.xmax)) for bbox in bboxes
    ]
    ys = [
        (ax.convert_xunits(bbox.ymin), ax.convert_yunits(bbox.ymax)) for bbox in bboxes
    ]
    coords = np.hstack([np.array(xs), np.array(ys)])
    return coords


def get_shifts_texts(coords):
    N = coords.shape[0]
    xoverlaps = overlap_intervals(
        coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1]
    )
    xoverlaps = xoverlaps[xoverlaps[:, 0] != xoverlaps[:, 1]]
    yoverlaps = overlap_intervals(
        coords[:, 2], coords[:, 3], coords[:, 2], coords[:, 3]
    )
    yoverlaps = yoverlaps[yoverlaps[:, 0] != yoverlaps[:, 1]]
    overlaps = yoverlaps[(yoverlaps[:, None] == xoverlaps).all(-1).any(-1)]
    if len(overlaps) == 0:
        return np.zeros((coords.shape[0])), np.zeros((coords.shape[0]))
    diff = coords[overlaps[:, 1]] - coords[overlaps[:, 0]]
    xshifts = np.where(np.abs(diff[:, 0]) < np.abs(diff[:, 1]), diff[:, 0], diff[:, 1])
    yshifts = np.where(np.abs(diff[:, 2]) < np.abs(diff[:, 3]), diff[:, 2], diff[:, 3])
    xshifts = np.bincount(overlaps[:, 0], xshifts, minlength=N)
    yshifts = np.bincount(overlaps[:, 0], yshifts, minlength=N)
    return xshifts, yshifts


def get_shifts_extra(coords, extra_coords):
    N = coords.shape[0]

    xoverlaps = overlap_intervals(
        coords[:, 0], coords[:, 1], extra_coords[:, 0], extra_coords[:, 1]
    )
    yoverlaps = overlap_intervals(
        coords[:, 2], coords[:, 3], extra_coords[:, 2], extra_coords[:, 3]
    )
    overlaps = yoverlaps[(yoverlaps[:, None] == xoverlaps).all(-1).any(-1)]

    if len(overlaps) == 0:
        return np.zeros((coords.shape[0])), np.zeros((coords.shape[0]))

    diff_x = coords[overlaps[:, 0], :2] - extra_coords[overlaps[:, 1], -3::-1]
    diff_y = coords[overlaps[:, 0], 2:] - extra_coords[overlaps[:, 1], -1:-3:-1]

    xshifts = np.where(
        np.abs(diff_x[:, 0]) < np.abs(diff_x[:, 1]), diff_x[:, 0], diff_x[:, 1]
    )
    yshifts = np.where(
        np.abs(diff_y[:, 0]) < np.abs(diff_y[:, 1]), diff_y[:, 0], diff_y[:, 1]
    )

    xshifts = np.bincount(overlaps[:, 0], xshifts, minlength=N)
    yshifts = np.bincount(overlaps[:, 0], yshifts, minlength=N)
    return xshifts, yshifts


def expand_coords(coords, x_frac, y_frac):
    mid_x = np.mean(coords[:, :2], axis=1)
    mid_y = np.mean(coords[:, 2:], axis=1)
    x = np.subtract(coords[:, :2], mid_x[:, np.newaxis]) * x_frac + mid_x[:, np.newaxis]
    y = np.subtract(coords[:, 2:], mid_y[:, np.newaxis]) * y_frac + mid_y[:, np.newaxis]
    return np.hstack([x, y])


def expand_axes_to_fit(coords, ax, transform):
    max_x, max_y = np.max(transform.inverted().transform(coords[:, [1, 3]]), axis=0)
    min_x, min_y = np.min(transform.inverted().transform(coords[:, [0, 2]]), axis=0)
    if min_x < ax.get_xlim()[0]:
        ax.set_xlim(xmin=min_x)
    if min_y < ax.get_ylim()[0]:
        ax.set_ylim(ymin=min_y)
    if max_x > ax.get_xlim()[1]:
        ax.set_xlim(xmax=max_x)
    if max_y > ax.get_ylim()[1]:
        ax.set_ylim(ymax=max_y)


def apply_shifts(coords, shifts_x, shifts_y):
    coords[:, :2] = np.subtract(coords[:, :2], shifts_x[:, np.newaxis])
    coords[:, 2:] = np.subtract(coords[:, 2:], shifts_y[:, np.newaxis])
    return coords


def force_into_bbox(coords, bbox):
    xmin, xmax, ymin, ymax = bbox
    dx, dy = np.zeros((coords.shape[0])), np.zeros((coords.shape[0]))
    if np.any((coords[:, 0] < xmin) & (coords[:, 1] > xmax)):
        logger.warn("Some labels are too long, can't fit inside the X axis")
    if np.any((coords[:, 2] < ymin) & (coords[:, 3] > ymax)):
        logger.warn("Some labels are too tall, can't fit inside the Y axis")
    dx[coords[:, 0] < xmin] = (xmin - coords[:, 0])[coords[:, 0] < xmin]
    dx[coords[:, 1] > xmax] = (xmax - coords[:, 1])[coords[:, 1] > xmax]
    dy[coords[:, 2] < ymin] = (ymin - coords[:, 2])[coords[:, 2] < ymin]
    dy[coords[:, 3] > ymax] = (ymax - coords[:, 3])[coords[:, 3] > ymax]
    return apply_shifts(coords, -dx, -dy)


def random_shifts(coords, only_move="xy"):
    # logger.debug(f"Random shifts with max_move: {max_move}")
    mids = np.hstack(
        [
            np.mean(coords[:, :2], axis=1)[:, np.newaxis],
            np.mean(coords[:, 2:], axis=1)[:, np.newaxis],
        ]
    )
    # if max_move is None:
    #     max_move = 1
    unq, count = np.unique(mids, axis=0, return_counts=True)
    repeated_groups = unq[count > 1]

    for repeated_group in repeated_groups:
        repeated_idx = np.argwhere(np.all(mids == repeated_group, axis=1)).flatten()
        logger.debug(f"Repeating group: {repeated_group}, idx: {repeated_idx}")
        for idx in repeated_idx:
            shifts = (np.random.rand(2) - 0.5) * 2
            if "x" not in only_move:
                shifts[0] = 0
            elif "x+" in only_move:
                shifts[0] = np.abs(shifts[0])
            elif "x-" in only_move:
                shifts[0] = -np.abs(shifts[0])
            if "y" not in only_move:
                shifts[1] = 0
            elif "y+" in only_move:
                shifts[1] = np.abs(shifts[1])
            elif "y-" in only_move:
                shifts[1] = -np.abs(shifts[1])
            print(idx, shifts)
            coords[idx] += np.asarray([shifts[0], shifts[0], shifts[1], shifts[1]])
    return coords


def pull_back(coords, targets):
    dx = np.max(np.subtract(targets[:, 0][:, np.newaxis], coords[:, :2]), axis=1)
    dy = np.max(np.subtract(targets[:, 1][:, np.newaxis], coords[:, 2:]), axis=1)
    return dx, dy


def explode(coords, static_coords, max_move, r=None):
    N = coords.shape[0]
    x = coords[:, [0, 1]].mean(axis=1)
    y = coords[:, [2, 3]].mean(axis=1)
    points = np.vstack([x, y]).T
    if static_coords.shape[0] > 0:
        static_x = np.mean(static_coords[:, [0, 1]], axis=1)
        static_y = np.mean(static_coords[:, [2, 3]], axis=1)
        static_centers = np.vstack([static_x, static_y]).T
        points = np.vstack([points, static_centers])
    tree = scipy.spatial.KDTree(points)
    pairs = tree.query_pairs(r, output_type="ndarray")
    pairs = pairs[pairs[:, 0] < N]
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    diff = points[pairs[:, 0]] - points[pairs[:, 1]]
    xshifts = np.clip(
        np.bincount(pairs[:, 0], diff[:, 0], minlength=N), -max_move[0], max_move[0]
    )
    yshifts = np.clip(
        np.bincount(pairs[:, 0], diff[:, 1], minlength=N), -max_move[1], max_move[1]
    )
    return xshifts, yshifts


def iterate(
    coords,
    target_coords,
    static_coords=None,
    force_text: tuple[float, float] = (0.1, 0.2),
    force_static: tuple[float, float] = (0.05, 0.1),
    force_pull: tuple[float, float] = (0.05, 0.1),
    pull_threshold: float = 10,
    expand: tuple[float, float] = (1.05, 1.1),
    max_move: tuple[int, int] = (10, 10),
    bbox_to_contain=False,
    only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
):
    coords = random_shifts(coords, only_move.get("explode", "xy"))
    text_shifts_x, text_shifts_y = get_shifts_texts(
        expand_coords(coords, expand[0], expand[1])
    )
    if static_coords.shape[0] > 0:
        static_shifts_x, static_shifts_y = get_shifts_extra(
            expand_coords(coords, expand[0], expand[1]), static_coords
        )
    else:
        static_shifts_x, static_shifts_y = np.zeros((1)), np.zeros((1))
    error_x = np.abs(text_shifts_x) + np.abs(static_shifts_x)
    error_y = np.abs(text_shifts_y) + np.abs(static_shifts_y)
    error = np.sum(np.append(error_x, error_y))

    pull_x, pull_y = pull_back(coords, target_coords)

    pull_x[np.abs(pull_x) < pull_threshold] = 0
    pull_y[np.abs(pull_y) < pull_threshold] = 0

    text_shifts_x *= force_text[0]
    text_shifts_y *= force_text[1]
    static_shifts_x *= force_static[0]
    static_shifts_y *= force_static[1]
    # Pull is in the opposite direction, so need to negate it
    pull_x *= -force_pull[0]
    pull_y *= -force_pull[1]
    pull_x[error_x != 0] = 0
    pull_y[error_y != 0] = 0

    if only_move:
        if "x" not in only_move.get("text", "xy"):
            text_shifts_x = np.zeros_like(text_shifts_x)
        elif "x+" in only_move.get("text", "xy"):
            text_shifts_x[text_shifts_x > 0] = 0
        elif "x-" in only_move.get("text", "xy"):
            text_shifts_x[text_shifts_x < 0] = 0

        if "y" not in only_move.get("text", "xy"):
            text_shifts_y = np.zeros_like(text_shifts_y)
        elif "y+" in only_move.get("text", "xy"):
            text_shifts_y[text_shifts_y > 0] = 0
        elif "y-" in only_move.get("text", "xy"):
            text_shifts_y[text_shifts_y < 0] = 0

        if "x" not in only_move.get("static", "xy"):
            static_shifts_x = np.zeros_like(static_shifts_x)
        elif "x+" in only_move.get("static", "xy"):
            static_shifts_x[static_shifts_x > 0] = 0
        elif "x-" in only_move.get("static", "xy"):
            static_shifts_x[static_shifts_x < 0] = 0

        if "y" not in only_move.get("static", "xy"):
            static_shifts_y = np.zeros_like(static_shifts_y)
        elif "y+" in only_move.get("static", "xy"):
            static_shifts_y[static_shifts_y > 0] = 0
        elif "y-" in only_move.get("static", "xy"):
            static_shifts_y[static_shifts_y < 0] = 0

        if "x" not in only_move.get("pull", "xy"):
            pull_x = np.zeros_like(pull_x)
        elif "x+" in only_move.get("pull", "xy"):
            pull_x[pull_x > 0] = 0
        elif "x-" in only_move.get("pull", "xy"):
            pull_x[pull_x < 0] = 0

        if "y" not in only_move.get("pull", "xy"):
            pull_y = np.zeros_like(pull_y)
        elif "y+" in only_move.get("pull", "xy"):
            pull_y[pull_y > 0] = 0
        elif "y-" in only_move.get("pull", "xy"):
            pull_y[pull_y < 0] = 0

    shifts_x = text_shifts_x + static_shifts_x + pull_x
    shifts_y = text_shifts_y + static_shifts_y + pull_y

    # Ensure that the shifts are not too large
    shifts_x = np.clip(
        np.sign(shifts_x) * np.ceil(np.abs(shifts_x)), -max_move[0], max_move[0]
    )
    shifts_y = np.clip(
        np.sign(shifts_y) * np.ceil(np.abs(shifts_y)), -max_move[1], max_move[1]
    )

    coords = apply_shifts(coords, shifts_x, shifts_y)
    if bbox_to_contain:
        coords = force_into_bbox(coords, bbox_to_contain)
    return coords, error


def force_draw(ax):
    try:
        ax.figure.draw_without_rendering()
    except AttributeError:
        logger.warn(
            """Looks like you are using an old matplotlib version.
               In some cases adjust_text might fail, if possible update
               matplotlib to version >=3.5.0"""
        )
        ax.figure.canvas.draw()


@lru_cache(None)
def warn_once(msg: str):
    logger.warning(msg)


def remove_crossings(coords, target_coords, step):
    connections = np.hstack(
        [
            np.mean(coords[:, :2], axis=1)[:, np.newaxis],
            np.mean(coords[:, 2:], axis=1)[:, np.newaxis],
            target_coords,
        ]
    )
    for i, seg1 in enumerate(connections):
        for j, seg2 in enumerate(connections):
            if i <= j:
                continue
            inter = intersect(seg1, seg2)
            if inter:
                logger.debug(f"Removing crossing at step {step}: {i} and {j}")
                logger.debug(f"Segments: {seg1} and {seg2}")
                coords[i], coords[j] = coords[j].copy(), coords[i].copy()
    return coords


def adjust_text(
    texts,
    x=None,
    y=None,
    objects=None,
    target_x=None,
    target_y=None,
    avoid_self=True,
    prevent_crossings=True,
    force_text: tuple[float, float] | float = (0.1, 0.2),
    force_static: tuple[float, float] | float = (0.1, 0.2),
    force_pull: tuple[float, float] | float = (0.01, 0.01),
    force_explode: tuple[float, float] | float = (0.1, 0.5),
    pull_threshold: float = 10,
    expand: tuple[float, float] = (1.05, 1.2),
    max_move: tuple[int, int] | int | None = (10, 10),
    explode_radius: str | float = "auto",
    ensure_inside_axes: bool = True,
    expand_axes: bool = False,
    only_move: dict = {"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
    ax: matplotlib.axes.Axes | None = None,
    min_arrow_len: float = 5,
    time_lim: float | None = None,
    iter_lim: int | None = None,
    *args,
    **kwargs,
):
    """Iteratively adjusts the locations of texts.

    Call adjust_text the very last, after all plotting (especially
    anything that can change the axes limits) has been done. This is
    because to move texts the function needs to use the dimensions of
    the axes, and without knowing the final size of the plots the
    results will be completely nonsensical, or suboptimal.

    First "explodes" all texts to move them apart.
    Then in each iteration pushes all texts away from each other, and any specified
    points or objects. At the same time slowly tries to pull the texts closer to their
    origianal locations that they label (this reduces chances that a text ends up super
    far away). In the end adds arrows connecting the texts to the respective points.

    Parameters
    ----------
    texts : list
        A list of :obj:`matplotlib.text.Text` objects to adjust.

    Other Parameters
    ----------------
    x : array_like
        x-coordinates of points to repel from; with avoid_self=True, the original
        text coordinates will be added to this array
    y : array_like
        y-coordinates of points to repel from; with avoid_self=True, the original
        text coordinates will be added to this array
    objects : list or PathCollection
        a list of additional matplotlib objects to avoid; they must have a
        `.get_window_extent()` method; alternatively, a PathCollection or a
        list of Bbox objects.
    target_x : array_like
        if provided, x-coordinates of points to connect adjusted texts to; if not
        provided, uses the original text coordinates.
        Provide together with target_y.
        Should be the same length as texts and in the same order, or None.
    target_y : array_like
        if provided, y-coordinates of points to connect adjusted texts to; if not
        provided, uses the original text coordinates.
        Provide together with target_x.
        Should be the same length as texts and in the same order, or None.
    avoid_self : bool, default True
        whether to repel texts from its original positions.
    prevent_crossings : bool, default True
        whether to prevent arrows from crossing each other [NEW, EXPERIMENTAL]
    force_text : tuple[float, float] | float, default (0.1, 0.2)
        the repel force from texts is multiplied by this value
    force_static : tuple[float, float] | float, default (0.1, 0.2)
        the repel force from points and objects is multiplied by this value
    force_pull : tuple[float, float] | float, default (0.01, 0.01)
        same as other forces, but for pulling texts back to original positions
    force_explode : tuple[float, float] | float, default (0.1, 0.5)
        same as other forces, but for the forced move of texts away from nearby texts
        and static positions before iterative adjustment
    pull_threshold : float, default 10
        how close to the original position the text should be pulled (if it's closer
        along one of the axes, don't pull along it) - in display coordinates
    expand : array_like, default (1.05, 1.2)
        a tuple/list/... with 2 multipliers (x, y) by which to expand the
        bounding box of texts when repelling them from each other.
    max_move : tuple[int, int] | int | None, default (10, 10)
        the maximum distance a text can be moved in one iteration in display units
        (in x and y directions); if a single integer or float is provided, it will be used for
        both x and y
    explode_radius : float or "auto", default "auto"
        how far to check for nearest objects to move the texts away in the beginning
        in display units, so on the order of 100 is the typical value.
        "auto" uses the mean size of the texts
    ensure_inside_axes : bool, default True
        Whether to force texts to stay inside the axes
    expand_axes : bool, default False
        Whether to expand the axes to fit all texts before adjusting there positions
    only_move : dict, default {"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"}
        a dict to restrict movement of texts to only certain axes for certain
        types of overlaps.
        Valid keys are 'text', 'static', 'explode' and 'pull'.
        'explode' is the initial explosion of texts to avoid overlaps, and this value is
        also used for random shifts of perfectly overlapping texts to ensure they don't
        stay in the same place.
        Can contain 'x', 'y', 'x+', 'x-', 'y+', 'y-', or combinations of one 'x?' and
        one 'y?'. 'x' and 'y' mean that the text can move in that direction, 'x+' and
        'x-' mean that the text can move in the positive or negative direction along
        the x axis, and similarly for 'y+' and 'y-'.
    ax : matplotlib axes, default is current axes (plt.gca())
        ax object with the plot
    min_arrow_len : float, default 5
        If the text is closer than this to the target point, don't add an arrow
        (in display units)
    time_lim : float, default None
        How much time to allow for the adjustments, in seconds.
        If both `time_lim` and iter_lim are set, faster will be used.
        If both are None, `time_lim` is set to 1 seconds.
    iter_lim : int, default None
        How many iterations to allow for the adjustments.
        If both `time_lim` and iter_lim are set, faster will be used.
        If both are None, `time_lim` is set to 1 seconds.
    args and kwargs :
        any arguments will be fed into obj:`FancyArrowPatch` after all the
        optimization is done just for plotting the connecting arrows if
        required.

    Returns
    -------
    texts : list
        List of adjusted text objects
    patches : list
        List of arrows connecting the texts to the target points.
        Typically they are :obj:`FancyArrowPatch` objects, but in some cases can be
        :obj:`matplotlib.text.Annotation` objects with empty text strings.
    """
    if not texts:
        return
    if ax is None:
        ax = plt.gca()

    force_draw(ax)

    try:
        transform = texts[0].get_transform()
    except IndexError:
        logger.warn(
            "Something wrong with the texts. Did you pass a list of matplotlib text objects?"
        )
        return
    if time_lim is None and iter_lim is None:
        time_lim = 1
    elif time_lim is not None and iter_lim is not None:
        logger.warn("Both time_lim and iter_lim are set, faster will be used")
    start_time = timer()
    coords = get_2d_coordinates(texts, ax)

    if expand_axes:
        expand_axes_to_fit(coords, ax, transform)
        force_draw(ax)
        transform = texts[0].get_transform()
        coords = get_2d_coordinates(texts, ax)

    original_coords = [text.get_unitless_position() for text in texts]
    original_coords_disp_coord = transform.transform(original_coords)

    target_xy = (
        list(zip(target_x, target_y))
        if (target_x is not None and target_y is not None)
        else original_coords
    )
    target_xy_disp_coord = transform.transform(target_xy)
    if isinstance(only_move, str):
        only_move = {
            "text": only_move,
            "static": only_move,
            "explode": only_move,
            "pull": only_move,
        }
    elif isinstance(only_move, dict):
        if "text" not in only_move:
            only_move["text"] = "xy"
        if "static" not in only_move:
            only_move["static"] = "xy"
        if "explode" not in only_move:
            only_move["explode"] = "xy"
        if "pull" not in only_move:
            only_move["pull"] = "xy"

    # coords += np.random.rand(*coords.shape)*1e-6
    if x is not None and y is not None:
        point_coords = transform.transform(np.vstack([x, y]).T)
    else:
        point_coords = np.empty((0, 2))
    if avoid_self:
        point_coords = np.vstack([point_coords, original_coords_disp_coord])

    if objects is None:
        obj_coords = np.empty((0, 4))
    else:
        obj_coords = get_2d_coordinates(objects, ax)
        obj_coords[:, [0, 2]] = transform.transform(obj_coords[:, [0, 2]])
        obj_coords[:, [1, 3]] = transform.transform(obj_coords[:, [1, 3]])
    static_coords = np.vstack([point_coords[:, [0, 0, 1, 1]], obj_coords])

    if isinstance(max_move, float) or isinstance(max_move, int):
        max_move = (max_move, max_move)
    elif max_move is None:
        max_move = (np.inf, np.inf)

    if isinstance(force_explode, float) or isinstance(force_explode, int):
        force_explode = (force_explode, force_explode)

    if isinstance(force_text, float) or isinstance(force_text, int):
        force_text = (force_text, force_text)

    if isinstance(force_static, float) or isinstance(force_static, int):
        force_static = (force_static, force_static)

    if isinstance(force_pull, float) or isinstance(force_pull, int):
        force_pull = (force_pull, force_pull)

    if explode_radius == "auto":
        explode_radius = max(
            (coords[:, 1] - coords[:, 0]).mean(), (coords[:, 3] - coords[:, 2]).mean()
        )
        logger.debug(f"Auto-explode radius: {explode_radius}")
    if explode_radius > 0 and np.any(np.asarray(force_explode) > 0):
        explode_x, explode_y = explode(
            coords, static_coords, max_move=max_move, r=explode_radius
        )
        if "x" not in only_move.get("explode", "xy"):
            explode_x = np.zeros_like(explode_x)
        elif "x+" in only_move.get("explode", "xy"):
            explode_x[explode_x < 0] = 0
        elif "x-" in only_move.get("explode", "xy"):
            explode_x[explode_x > 0] = 0
        if "y" not in only_move.get("explode", "xy"):
            explode_y = np.zeros_like(explode_y)
        elif "y+" in only_move.get("explode", "xy"):
            explode_y[explode_y < 0] = 0
        elif "y-" in only_move.get("explode", "xy"):
            explode_y[explode_y > 0] = 0

        coords = apply_shifts(
            coords, -explode_x * force_explode[0], -explode_y * force_explode[1]
        )

    error = np.inf

    # i_0 = 100
    # i = i_0
    # expand_start = 1.05, 1.5
    # expand_end = 1.05, 1.5
    # expand_steps = 100

    # expands = list(zip(np.linspace(expand_start[0], expand_end[0], expand_steps),
    #                 np.linspace(expand_start[1], expand_end[1], expand_steps)))

    if ensure_inside_axes:
        ax_bbox = ax.patch.get_extents()
        ax_bbox = ax_bbox.xmin, ax_bbox.xmax, ax_bbox.ymin, ax_bbox.ymax
    else:
        ax_bbox = False

    step = 0
    while error > 0:
        # expand = expands[min(i, expand_steps-1)]
        logger.debug(step)
        coords, error = iterate(
            coords,
            target_xy_disp_coord,
            static_coords,
            force_text=force_text,
            force_static=force_static,
            force_pull=force_pull,
            pull_threshold=pull_threshold,
            expand=expand,
            max_move=max_move,
            bbox_to_contain=ax_bbox,
            only_move=only_move,
        )
        if prevent_crossings:
            coords = remove_crossings(coords, target_xy_disp_coord, step)

        step += 1
        if time_lim is not None and timer() - start_time > time_lim:
            break
        if iter_lim is not None and step == iter_lim:
            break

    logger.debug(f"Adjustment took {step} iterations")
    logger.debug(f"Time: {timer() - start_time}")
    logger.debug(f"Error: {error}")

    xdists = np.min(
        np.abs(np.subtract(coords[:, :2], target_xy_disp_coord[:, 0][:, np.newaxis])),
        axis=1,
    )
    ydists = np.min(
        np.abs(np.subtract(coords[:, 2:], target_xy_disp_coord[:, 1][:, np.newaxis])),
        axis=1,
    )
    display_dists = np.max(np.vstack([xdists, ydists]), axis=0)
    connections = np.hstack(
        [
            np.mean(coords[:, :2], axis=1)[:, np.newaxis],
            np.mean(coords[:, 2:], axis=1)[:, np.newaxis],
            target_xy_disp_coord,
        ]
    )
    transformed_connections = np.empty_like(connections)
    transformed_connections[:, :2] = transform.inverted().transform(connections[:, :2])
    transformed_connections[:, 2:] = transform.inverted().transform(connections[:, 2:])

    if "arrowprops" in kwargs:
        ap = kwargs.pop("arrowprops")
    else:
        ap = {}
    patches = []
    for i, text in enumerate(texts):
        text_mid = transformed_connections[i, :2]
        target = transformed_connections[i, 2:]
        text.set_verticalalignment("center")
        text.set_horizontalalignment("center")
        text.set_position(text_mid)

        if ap and display_dists[i] >= min_arrow_len:
            try:
                arrowpatch = FancyArrowPatch(
                    posA=text_mid,
                    posB=target,
                    patchA=text,
                    transform=transform,
                    *args,
                    **kwargs,
                    **ap,
                )
                ax.add_patch(arrowpatch)
                patches.append(arrowpatch)
            except AttributeError:
                warn_once(
                    "Looks like you are using a tranform that doesn't support "
                    "FancyArrowPatch, using ax.annotate instead. The arrows might "
                    "strike through texts. "
                    "Increasing shrinkA in arrowprops might help."
                )
                ann = ax.annotate(
                    "",
                    xy=target,
                    xytext=text_mid,
                    arrowprops=ap,
                    xycoords=transform,
                    textcoords=transform,
                )
                # Theoretically something like this should avoid the arrow striking through the text, but doesn't work...
                ann.arrow_patch.set_patchA(text)
                patches.append(ann)
    return texts, patches

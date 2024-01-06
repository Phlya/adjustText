from __future__ import division, annotations
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import get_path_collection_extents
import bioframe as bf
import scipy.spatial.distance
import logging
from timeit import default_timer as timer
import io

try:
    from matplotlib.backend_bases import _get_renderer as matplot_get_renderer
except ImportError:
    matplot_get_renderer = None

from ._version import __version__


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
            bboxes.append(result)

    return bboxes


def get_bboxes(objs, r=None, expand=(1, 1), ax=None, transform=None):
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
    transform : optional
        Transform to apply to the objects, if they don't return they window extent.
        The default is None, then applies the default ax transform.

    Returns
    -------
    list
        List of bboxes.

    """
    ax = ax or plt.gca()
    r = r or get_renderer(ax.get_figure())
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


def get_2d_coordinates(objs):
    try:
        ax = objs[0].axes
    except:
        ax = objs.axes
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
    xoverlaps = bf.core.arrops.overlap_intervals(
        coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1]
    )
    xoverlaps = xoverlaps[xoverlaps[:, 0] != xoverlaps[:, 1]]
    yoverlaps = bf.core.arrops.overlap_intervals(
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

    xoverlaps = bf.core.arrops.overlap_intervals(
        coords[:, 0], coords[:, 1], extra_coords[:, 0], extra_coords[:, 1]
    )
    yoverlaps = bf.core.arrops.overlap_intervals(
        coords[:, 2], coords[:, 3], extra_coords[:, 2], extra_coords[:, 3]
    )
    overlaps = yoverlaps[(yoverlaps[:, None] == xoverlaps).all(-1).any(-1)]

    if len(overlaps) == 0:
        return np.zeros((coords.shape[0])), np.zeros((coords.shape[0]))

    diff = coords[overlaps[:, 0]] - extra_coords[overlaps[:, 1]]

    xshifts = np.where(np.abs(diff[:, 0]) < np.abs(diff[:, 1]), diff[:, 0], diff[:, 1])
    yshifts = np.where(np.abs(diff[:, 2]) < np.abs(diff[:, 3]), diff[:, 2], diff[:, 3])
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
    max_x, max_y = np.max(transform.transform(coords[:, [1, 3]]), axis=0)
    min_x, min_y = np.min(transform.transform(coords[:, [1, 3]]), axis=0)
    if min_x < ax.get_xlim()[0]:
        ax.set_xlim(xmin=min_x)
    if min_y < ax.get_ylim()[0]:
        ax.set_ylim(ymin=min_y)
    if max_x < ax.get_xlim()[1]:
        ax.set_xlim(xmin=max_x)
    if max_y < ax.get_ylim()[1]:
        ax.set_ylim(ymin=max_y)


def apply_shifts(coords, shifts_x, shifts_y):
    coords[:, :2] = np.subtract(coords[:, :2], shifts_x[:, np.newaxis])
    coords[:, 2:] = np.subtract(coords[:, 2:], shifts_y[:, np.newaxis])
    return coords


def force_into_bbox(coords, bbox):
    xmin, xmax, ymin, ymax = bbox
    dx, dy = np.zeros((coords.shape[0])), np.zeros((coords.shape[0]))
    if np.any((coords[:, 0] < xmin) & (coords[:, 1] > xmax)):
        logging.warn("Some labels are too long, can't fit inside the X axis")
    if np.any((coords[:, 2] < ymin) & (coords[:, 3] > ymax)):
        logging.warn("Some labels are too toll, can't fit inside the Y axis")
    dx[coords[:, 0] < xmin] = (xmin - coords[:, 0])[coords[:, 0] < xmin]
    dx[coords[:, 1] > xmax] = (xmax - coords[:, 1])[coords[:, 1] > xmax]
    dy[coords[:, 2] < ymin] = (ymin - coords[:, 2])[coords[:, 2] < ymin]
    dy[coords[:, 3] > ymax] = (ymax - coords[:, 3])[coords[:, 3] > ymax]
    return apply_shifts(coords, -dx, -dy)


def pull_back(coords, targets):
    dx = np.min(np.subtract(targets[:, 0][:, np.newaxis], coords[:, :2]), axis=1)
    dy = np.min(np.subtract(targets[:, 1][:, np.newaxis], coords[:, 2:]), axis=1)
    return dx, dy


def explode(coords, static_coords, r=None):
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
    xshifts = np.bincount(pairs[:, 0], diff[:, 0], minlength=N)
    yshifts = np.bincount(pairs[:, 0], diff[:, 1], minlength=N)
    return xshifts, yshifts


def iterate(
    coords,
    orig_coords,
    static_coords=None,
    force_text: tuple[float, float] = (0.1, 0.2),
    force_static: tuple[float, float] = (0.05, 0.1),
    force_pull: tuple[float, float] = (0.05, 0.1),
    expand: tuple[float, float] = (1.05, 1.1),
    bbox_to_contain=False,
    only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
):

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

    pull_x, pull_y = pull_back(coords, orig_coords)

    text_shifts_x *= force_text[0]
    text_shifts_y *= force_text[1]
    static_shifts_x *= force_static[0]
    static_shifts_y *= force_static[1]
    # Pull is in the opposite direction, so need to negate it
    pull_x *= -force_pull[0]
    pull_y *= -force_pull[1]
    # pull_x[error_x != 0] = 0
    # pull_y[error_y != 0] = 0

    if only_move:
        if "x" not in only_move["text"]:
            text_shifts_x = np.zeros_like(text_shifts_x)
        elif "x+" in only_move["text"]:
            text_shifts_x[text_shifts_x > 0] = 0
        elif "x-" in only_move["text"]:
            text_shifts_x[text_shifts_x < 0] = 0

        if "y" not in only_move["text"]:
            text_shifts_y = np.zeros_like(text_shifts_y)
        elif "y+" in only_move["text"]:
            text_shifts_y[text_shifts_y > 0] = 0
        elif "y-" in only_move["text"]:
            text_shifts_y[text_shifts_y < 0] = 0

        if "x" not in only_move["static"]:
            static_shifts_x = np.zeros_like(static_shifts_x)
        elif "x+" in only_move["static"]:
            static_shifts_x[static_shifts_x > 0] = 0
        elif "x-" in only_move["static"]:
            static_shifts_x[static_shifts_x < 0] = 0

        if "y" not in only_move["static"]:
            static_shifts_y = np.zeros_like(static_shifts_y)
        elif "y+" in only_move["static"]:
            static_shifts_y[static_shifts_y > 0] = 0
        elif "y-" in only_move["static"]:
            static_shifts_y[static_shifts_y < 0] = 0

        if "x" not in only_move["pull"]:
            pull_x = np.zeros_like(pull_x)
        elif "x+" in only_move["pull"]:
            pull_x[pull_x > 0] = 0
        elif "x-" in only_move["pull"]:
            pull_x[pull_x < 0] = 0

        if "y" not in only_move["pull"]:
            pull_y = np.zeros_like(pull_y)
        elif "y+" in only_move["pull"]:
            pull_y[pull_y > 0] = 0
        elif "y-" in only_move["pull"]:
            pull_y[pull_y < 0] = 0

    shifts_x = text_shifts_x + static_shifts_x + pull_x
    shifts_y = text_shifts_y + static_shifts_y + pull_y

    coords = apply_shifts(coords, shifts_x, shifts_y)
    if bbox_to_contain:
        coords = force_into_bbox(coords, bbox_to_contain)
    return coords, error


def adjust_text(
    texts,
    x=None,
    y=None,
    objects=None,
    avoid_self=True,
    force_text: tuple[float, float] = (0.1, 0.2),
    force_static: tuple[float, float] = (0.1, 0.2),
    force_pull: tuple[float, float] = (0.01, 0.01),
    force_explode: tuple[float, float] = (0.01, 0.02),
    expand: tuple[float, float] = (1.05, 1.1),
    explode_radius="auto",
    ensure_inside_axes=True,
    expand_axes=False,
    only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
    ax=None,
    min_arrow_len=5,
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

    First moves all texts that are outside the axes limits
    inside. Then in each iteration moves all texts away from each
    other and from points. In the end hides texts and substitutes them
    with annotations to link them to the respective points.

    Parameters
    ----------
    texts : list
        A list of :obj:`matplotlib.text.Text` objects to adjust.

    Other Parameters
    ----------------
    x : array_like
        x-coordinates of points to repel from; if not provided only uses text
        coordinates.
    y : array_like
        y-coordinates of points to repel from; if not provided only uses text
        coordinates
    objects : list or PathCollection
        a list of additional matplotlib objects to avoid; they must have a
        `.get_window_extent()` method; alternatively, a PathCollection or a
        list of Bbox objects.
    avoid_self : bool, default True
        whether to repel texts from its original positions.
    force_text : tuple, default (0.1, 0.2)
        the repel force from texts is multiplied by this value
    force_static : tuple, default (0.1, 0.2)
        the repel force from points and objects is multiplied by this value
    force_pull : tuple, default (0.01, 0.01)
        same as other forces, but for pulling texts back to original positions
    force_explode : float, default (0.01, 0.02)
        same as other forces, but for the forced move of texts away from nearby texts
        and static positions before iterative adjustment
    expand : array_like, default (1.05, 1.1)
        a tuple/list/... with 2 multipliers (x, y) by which to expand the
        bounding box of texts when repelling them from each other.
    explode_radius : float or "auto", default "auto"
        how far to check for nearest objects to move the texts away in the beginning
        in display units, so on the order of 100 is the typical value
        "auto" uses the size of the largest text
    ensure_inside_axes : bool, default True
        Whether to force texts to stay inside the axes
    expand_axes : bool, default False
        Whether to expand the axes to fit all texts before adjusting there positions
    only_move : dict, default {"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"}
        a dict to restrict movement of texts to only certain axes for certain
        types of overlaps.
        Valid keys are 'text', 'static', 'explode' and 'pull'.
        Valid values are '', 'x', 'y', and 'xy'.
    ax : matplotlib axe, default is current axe (plt.gca())
        ax object with the plot
    min_arrow_len : float, default 5
        If the text is closer than this to the target point, don't add an arrow
        (in display units)
    time_lim : float, default None
        How much time to allow for the adjustments, in seconds.
        If both `time_lim` and iter_lim are set, faster will be used.
        If both are None, `time_lim` is set to 0.5 seconds.
    iter_lim : int, default None
        How many iterations to allow for the adjustments.
        If both `time_lim` and iter_lim are set, faster will be used.
        If both are None, `time_lim` is set to 0.5 seconds.
    args and kwargs :
        any arguments will be fed into obj:`FancyArrowPatch` after all the
        optimization is done just for plotting the connecting arrows if
        required.
    """
    if not texts:
        return
    if ax is None:
        ax = plt.gca()
    try:
        ax.figure.draw_without_rendering()
    except AttributeError:
        logging.warn(
            """Looks like you are using an old matplotlib version.
               In some cases adjust_text might fail, if possible update
               matplotlib to version >=3.5.0"""
        )
        ax.figure.canvas.draw()
    try:
        transform = texts[0].get_transform()
    except IndexError:
        logging.warn(
            "Something wrong with the texts. Did you pass a list of matplotlib text objects?"
        )
        return
    if time_lim is None and iter_lim is None:
        time_lim = 0.5
    elif time_lim is not None and iter_lim is not None:
        logging.warn("Both time_lim and iter_lim are set, faster will be used")
    start_time = timer()

    orig_xy = [text.get_unitless_position() for text in texts]
    orig_xy_disp_coord = transform.transform(orig_xy)
    coords = get_2d_coordinates(texts)

    if isinstance(only_move, str):
        only_move = {
            "text": only_move,
            "static": only_move,
            "explode": only_move,
            "pull": only_move,
        }

    # coords += np.random.rand(*coords.shape)*1e-6
    if x is not None and y is not None:
        point_coords = transform.transform(np.vstack([x, y]).T)
    else:
        point_coords = np.empty((0, 2))
    if avoid_self:
        point_coords = np.vstack(
            [
                point_coords,
                np.hstack(
                    [
                        np.mean(coords[:, :2], axis=1)[:, np.newaxis],
                        np.mean(coords[:, 2:], axis=1)[:, np.newaxis],
                    ]
                ),
            ]
        )

    if objects is None:
        obj_coords = np.empty((0, 4))
    else:
        obj_coords = get_2d_coordinates(objects)
    static_coords = np.vstack([point_coords[:, [0, 0, 1, 1]], obj_coords])
    if explode_radius == "auto":
        explode_radius = max(
            (coords[:, 1] - coords[:, 0]).max(), (coords[:, 3] - coords[:, 2]).max()
        )
    if explode_radius > 0 and np.all(np.asarray(force_explode) > 0):
        explode_x, explode_y = explode(coords, static_coords, explode_radius)
        if "x" not in only_move["explode"]:
            explode_x = np.zeros_like(explode_x)
        if "y" not in only_move["explode"]:
            explode_y = np.zeros_like(explode_y)
        coords = apply_shifts(
            coords, -explode_x * force_explode[0], -explode_y * force_explode[1]
        )

    error = np.Inf

    # i_0 = 100
    # i = i_0
    # expand_start = 1.05, 1.5
    # expand_end = 1.05, 1.5
    # expand_steps = 100

    # expands = list(zip(np.linspace(expand_start[0], expand_end[0], expand_steps),
    #                 np.linspace(expand_start[1], expand_end[1], expand_steps)))

    if expand_axes:
        expand_axes_to_fit(coords, ax, transform)
    if ensure_inside_axes:
        ax_bbox = ax.patch.get_extents()
        ax_bbox = ax_bbox.xmin, ax_bbox.xmax, ax_bbox.ymin, ax_bbox.ymax
    else:
        ax_bbox = False

    i = 0
    while error > 0:
        # expand = expands[min(i, expand_steps-1)]
        coords, error = iterate(
            coords,
            orig_xy_disp_coord,
            static_coords,
            force_text=force_text,
            force_static=force_static,
            force_pull=force_pull,
            expand=expand,
            bbox_to_contain=ax_bbox,
            only_move=only_move,
        )

        i += 1
        if time_lim is not None and timer() - start_time > time_lim:
            break
        if iter_lim is not None and i == iter_lim:
            break

    xdists = np.min(
        np.abs(np.subtract(coords[:, :2], orig_xy_disp_coord[:, 0][:, np.newaxis])),
        axis=1,
    )
    ydists = np.min(
        np.abs(np.subtract(coords[:, 2:], orig_xy_disp_coord[:, 1][:, np.newaxis])),
        axis=1,
    )
    display_dists = np.max(np.vstack([xdists, ydists]), axis=0)

    connections = np.hstack(
        [
            np.mean(coords[:, :2], axis=1)[:, np.newaxis],
            np.mean(coords[:, 2:], axis=1)[:, np.newaxis],
            orig_xy_disp_coord,
        ]
    )  # For the future to move into the loop and resolve crossing connections

    transformed_connections = np.empty_like(connections)
    transformed_connections[:, :2] = transform.inverted().transform(connections[:, :2])
    transformed_connections[:, 2:] = transform.inverted().transform(connections[:, 2:])

    if "arrowprops" in kwargs:
        ap = kwargs.pop("arrowprops")
    else:
        ap = {}

    for i, text in enumerate(texts):
        text_mid = transformed_connections[i, :2]
        target = transformed_connections[i, 2:]
        text.set_verticalalignment("center")
        text.set_horizontalalignment("center")
        text.set_position(text_mid)

        if ap and display_dists[i] >= min_arrow_len:
            arrowpatch = FancyArrowPatch(
                posA=text_mid, posB=target, patchA=text, *args, **kwargs, **ap
            )
            ax.add_patch(arrowpatch)

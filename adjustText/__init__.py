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

from ._version import __version__


def get_renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig.canvas.renderer


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
    xmin = bbox.xmin
    xmax = bbox.xmax
    ymin = bbox.ymin
    ymax = bbox.ymax
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


def adjust_text(
    texts,
    x=None,
    y=None,
    objects=None,
    avoid_self=True,
    force_text: tuple[float, float] = (0.1, 0.2),
    force_static: tuple[float, float] = (0.05, 0.1),
    force_pull: tuple[float, float] = (0.01, 0.001),
    explode_force: tuple[float, float] = (0.01, 0.01),
    expand: tuple[float, float] = (1.05, 1.1),
    explode_radius="auto",
    ensure_inside_axes=True,
    expand_axes=False,
    only_move={"text": "xy", "static": "xy", "explode": "xy"},
    ax=None,
    min_arrow_len=5,
    time_lim=0.1,
    *args,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    plt.draw()

    start_time = timer()

    transform = texts[0].get_transform()

    orig_xy = [text.get_unitless_position() for text in texts]
    orig_xy_disp_coord = transform.transform(orig_xy)
    coords = get_2d_coordinates(texts)

    if isinstance(only_move, str):
        only_move = {"text": only_move, "static": only_move, "explode": only_move}

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
    if explode_radius > 0 and np.all(np.asarray(explode_force) > 0):
        explode_x, explode_y = explode(coords, static_coords, explode_radius)
        if "x" not in only_move["explode"]:
            explode_x = np.zeros_like(explode_x)
        if "y" not in only_move["explode"]:
            explode_y = np.zeros_like(explode_y)
        coords = apply_shifts(
            coords, -explode_x * explode_force[0], -explode_y * explode_force[1]
        )

    error = np.Inf

    # i_0 = 100
    # i = i_0
    # expand_start = 1.05, 1.5
    # expand_end = 1.05, 1.5
    # expand_steps = 100

    # expands = list(zip(np.linspace(expand_start[0], expand_end[0], expand_steps),
    #                 np.linspace(expand_start[1], expand_end[1], expand_steps)))
    ax_bbox = ax.patch.get_extents()

    if expand_axes:
        expand_axes_to_fit(coords, ax, transform)

    # i = 0
    while error > 0:
        # expand = expands[min(i, expand_steps-1)]
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
        pull_x, pull_y = pull_back(coords, orig_xy_disp_coord)

        text_shifts_x *= force_text[0]
        text_shifts_y *= force_text[1]
        static_shifts_x *= force_static[0]
        static_shifts_y *= force_static[1]
        pull_x *= force_pull[0]
        pull_y *= force_pull[1]

        if not any(["x" in val for val in only_move.values()]):
            pull_x = np.zeros_like(pull_x)
        if not any(["y" in val for val in only_move.values()]):
            pull_y = np.zeros_like(pull_y)

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

        shifts_x = text_shifts_x + static_shifts_x - pull_x
        shifts_y = text_shifts_y + static_shifts_y - pull_y

        coords = apply_shifts(coords, shifts_x, shifts_y)
        if ensure_inside_axes:
            coords = force_into_bbox(coords, ax_bbox)

        if timer() - start_time > time_lim:
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
                posB=text_mid, posA=target, patchB=text, *args, **kwargs, **ap
            )
            ax.add_patch(arrowpatch)

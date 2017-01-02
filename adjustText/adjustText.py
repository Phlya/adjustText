from __future__ import division
import sys
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
from operator import itemgetter

if sys.version_info >= (3, 0):
    xrange = range

def get_bboxes(objs, r, expand=(1.0, 1.0), ax=None):
    if ax is None:
        ax = plt.gca()
    return [i.get_window_extent(r).expanded(*expand).transformed(ax.\
                                          transData.inverted()) for i in objs]

def get_midpoint(bbox):
    cx = (bbox.x0+bbox.x1)/2
    cy = (bbox.y0+bbox.y1)/2
    return cx, cy

def get_points_inside_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
    x_in = np.logical_and(x>x1, x<x2)
    y_in = np.logical_and(y>y1, y<y2)
    return np.where(x_in & y_in)[0]

def overlap_bbox_and_point(bbox, xp, yp):
    cx, cy = get_midpoint(bbox)

    dir_x = np.sign(cx-xp)
    dir_y = np.sign(cy-yp)

    if dir_x == -1:
        dx = xp - bbox.xmax
    elif dir_x == 1:
        dx = xp - bbox.xmin
    else:
        dx = 0

    if dir_y == -1:
        dy = yp - bbox.ymax
    elif dir_y == 1:
        dy = yp - bbox.ymin
    else:
        dy = 0
    return dx, dy

def move_texts(texts, delta_x, delta_y, bboxes=None, renderer=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if bboxes is None:
        if renderer is None:
            r = ax.get_figure().canvas.get_renderer()
        else:
            r = renderer
        bboxes = get_bboxes(texts, r, (1, 1))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for i, (text, dx, dy) in enumerate(zip(texts, delta_x, delta_y)):
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        if x1 + dx < xmin:
            dx = 0
        if x2 + dx > xmax:
            dx = 0
        if y1 + dy < ymin:
            dy = 0
        if y2 + dy > ymax:
            dy = 0

        x, y = text.get_position()
        newx = x + dx
        newy = y + dy
        text.set_position((newx, newy))

def optimally_align_text(x, y, texts, expand=(1., 1.), add_bboxes=[],
                         renderer=None, ax=None,
                         direction='xy'):
    """
    For all text objects find alignment that causes the least overlap with
    points and other texts and apply it
    """
    if ax is None:
        ax = plt.gca()
    if renderer is None:
        r = ax.get_figure().canvas.get_renderer()
    else:
        r = renderer
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    bboxes = get_bboxes(texts, r, expand)
    if 'x' not in direction:
        ha = ['']
    else:
        ha = ['center', 'left', 'right']
    if 'y' not in direction:
        va = ['']
    else:
        va = ['bottom', 'top', 'center']
    alignment = list(product(ha, va))
    for i, text in enumerate(texts):
        counts = []
        for h, v in alignment:
            if h:
                text.set_ha(h)
            if v:
                text.set_va(v)
            bbox = text.get_window_extent(r).expanded(*expand).\
                                       transformed(ax.transData.inverted())
            c = len(get_points_inside_bbox(x, y, bbox))
            intersections = [bbox.intersection(bbox, bbox2) for bbox2 in
                             bboxes+add_bboxes]
            intersections = sum([abs(b.width*b.height) if b is not None else 0
                                 for b in intersections])
            # Check for out-of-axes position
            bbox = text.get_window_extent(r).transformed(ax.transData.inverted())
            x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            if x1 < xmin or x2 > xmax or y1 < ymin or y2 > ymax:
                axout = 1
            else:
                axout = 0
            counts.append((axout, c, intersections))
        a, value = min(enumerate(counts), key=itemgetter(1))
        if 'x' in direction:
            text.set_ha(alignment[a][0])
        if 'y' in direction:
            text.set_va(alignment[a][1])
        bboxes[i] = text.get_window_extent(r).expanded(*expand).\
                                       transformed(ax.transData.inverted())
    return texts

def repel_text(texts, renderer=None, ax=None, expand=(1.2, 1.2),
               only_use_max_min=False, move=False):
    """
    Repel texts from each other while expanding their bounding boxes by expand
    (x, y), e.g. (1.2, 1.2) would multiply width and height by 1.2.
    Requires a renderer to get the actual sizes of the text, and to that end
    either one needs to be directly provided, or the axes have to be specified,
    and the renderer is then got from the axes object.
    """
    if ax is None:
        ax = plt.gca()
    if renderer is None:
        r = ax.get_figure().canvas.get_renderer()
    else:
        r = renderer
    bboxes = get_bboxes(texts, r, expand)
    xmins = [bbox.xmin for bbox in bboxes]
    xmaxs = [bbox.xmax for bbox in bboxes]
    ymaxs = [bbox.ymax for bbox in bboxes]
    ymins = [bbox.ymin for bbox in bboxes]

    overlaps_x = np.zeros((len(bboxes), len(bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)
    for i, bbox1 in enumerate(bboxes):
        overlaps = get_points_inside_bbox(xmins*2+xmaxs*2, (ymins+ymaxs)*2,
                                             bbox1) % len(bboxes)
        overlaps = np.unique(overlaps)
        for j in overlaps:
            bbox2 = bboxes[j]
            x, y = bbox1.intersection(bbox1, bbox2).size
            overlaps_x[i, j] = x
            overlaps_y[i, j] = y
            direction = np.sign(bbox1.extents - bbox2.extents)[:2]
            overlap_directions_x[i, j] = direction[0]
            overlap_directions_y[i, j] = direction[1]

    move_x = overlaps_x*overlap_directions_x
    move_y = overlaps_y*overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(np.abs(delta_x) + np.abs(delta_y))
    if move:
        move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
    return delta_x, delta_y, q

def repel_text_from_bboxes(add_bboxes, texts, renderer=None, ax=None,
                           expand=(1.2, 1.2), only_use_max_min=False,
                           move=False):
    """
    Repel texts from other objects' bboxes while expanding their (texts')
    bounding boxes by expand (x, y), e.g. (1.2, 1.2) would multiply width and
    height by 1.2.
    Requires a renderer to get the actual sizes of the text, and to that end
    either one needs to be directly provided, or the axes have to be specified,
    and the renderer is then got from the axes object.
    """
    if ax is None:
        ax = plt.gca()
    if renderer is None:
        r = ax.get_figure().canvas.get_renderer()
    else:
        r = renderer

    bboxes = get_bboxes(texts, r, expand)

    overlaps_x = np.zeros((len(bboxes), len(add_bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)

    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(add_bboxes):
            try:
                x, y = bbox1.intersection(bbox1, bbox2).size
                direction = np.sign(bbox1.extents - bbox2.extents)[:2]
                overlaps_x[i, j] = x
                overlaps_y[i, j] = y
                overlap_directions_x[i, j] = direction[0]
                overlap_directions_y[i, j] = direction[1]
            except AttributeError:
                pass

    move_x = overlaps_x*overlap_directions_x
    move_y = overlaps_y*overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(np.abs(delta_x) + np.abs(delta_y))
    if move:
        move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
    return delta_x, delta_y, q

def repel_text_from_points(x, y, texts, renderer=None, ax=None,
                           expand=(1.2, 1.2), move=False):
    """
    Repel texts from all points specified by x and y while expanding their
    (texts'!) bounding boxes by expandby  (x, y), e.g. (1.2, 1.2)
    would multiply both width and height by 1.2. In the case when the text
    overlaps a point, but there is no definite direction for movement, moves
    in random direction by 40% of it's width and/or height depending on
    Requires a renderer to get the actual sizes of the text, and to that end
    either one needs to be directly provided, or the axes have to be specified,
    and the renderer is then got from the axes object.
    """
    assert len(x) == len(y)
    if ax is None:
        ax = plt.gca()
    if renderer is None:
        r = ax.get_figure().canvas.get_renderer()
    else:
        r = renderer
    bboxes = get_bboxes(texts, r, expand)

    move_x = np.zeros((len(bboxes), len(x)))
    move_y = np.zeros((len(bboxes), len(x)))
    for i, bbox in enumerate(bboxes):
        xy_in = get_points_inside_bbox(x, y, bbox)
        for j in xy_in:
            xp, yp = x[j], y[j]
            dx, dy = overlap_bbox_and_point(bbox, xp, yp)

            move_x[i, j] = dx
            move_y[i, j] = dy

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)
    q = np.sum(np.abs(delta_x) + np.abs(delta_y))
    if move:
        move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
    return delta_x, delta_y, q

def repel_text_from_axes(texts, ax=None, bboxes=None, renderer=None,
                         expand=None):
    if ax is None:
        ax = plt.gca()
    if renderer is None:
        r = ax.get_figure().canvas.get_renderer()
    else:
        r = renderer
    if expand is None:
        expand = (1, 1)
    if bboxes is None:
        bboxes = get_bboxes(texts, r, expand=expand)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        dx, dy = 0, 0
        if x1 < xmin:
            dx = xmin - x1
        if x2 > xmax:
            dx = xmax - x2
        if y1 < ymin:
            dy = ymin - y1
        if y2 > ymax:
            dy = ymax - y2
        if dx or dy:
            x, y = texts[i].get_position()
            newx, newy = x + dx, y + dy
            texts[i].set_position((newx, newy))
    return texts

def adjust_text(texts, x=None, y=None, add_objects=None, ax=None,
                expand_text=(1.2, 1.2), expand_points=(1.2, 1.2),
                expand_objects=(1.2, 1.2), expand_align=(0.9, 0.9),
                autoalign='xy',  va='center', ha='center',
                force_text=0.5, force_points=0.5, force_objects=0.5,
                lim=100, precision=0,
                only_move={}, text_from_text=True, text_from_points=True,
                save_steps=False, save_prefix='', save_format='png',
                add_step_numbers=True, draggable=True,
                *args, **kwargs):
    """
    Iteratively adjusts the locations of texts. First moves all texts that are
    outside the axes limits inside. Then in each iteration moves all texts away
    from each other and from points. In the end hides texts and substitutes
    them with annotations to link them to the respective points.

    Args:
        texts (list): a list of text.Text objects to adjust
        x (seq): x-coordinates of points to repel from; if not provided only
            uses text coordinates
        y (seq): y-coordinates of points to repel from; if not provided only
            uses text coordinates
        add_objects (list): a list of additional matplotlib objects to avoid;
            they must have a .get_window_extent() method
        ax (obj): axes object with the plot; if not provided is determined by
            plt.gca()
        expand_text (seq): a tuple/list/... with 2 numbers (x, y) to expand
            texts when repelling them from each other; default (1.2, 1.2)
        expand_points (seq): a tuple/list/... with 2 numbers (x, y) to expand
            texts when repelling them from points; default (1.2, 1.2)
        expand_objects (seq): a tuple/list/... with 2 numbers (x, y) to expand
            texts when repelling them from points; default (1.2, 1.2)
        expand_align (seq): a tuple/list/... with 2 numbers (x, y) to expand
            texts when autoaligning texts; default (1., 1.)
        autoalign: If 'xy', the best alignment of all texts will be
            determined in all directions automatically before running the
            iterative adjustment; if 'x' will only align horizontally, if 'y' -
            vertically; overrides va and ha
        va (str): vertical alignment of texts
        ha (str): horizontal alignment of texts
        force_text (float): the repel force from texts is multiplied by this
            value; default 0.5
        force_points (float): the repel force from points is multiplied by this
            value; default 0.5
        force_objects (float): same as other forces, but for repelling
            additional objects
        lim (int): limit of number of iterations
        precision (float): up to which sum of all overlaps along both x and y
            to iterate; may need to increase for complicated situations;
            default 0, so no overlaps with anything.
        only_move (dict): a dict to restrict movement of texts to only certain
            axis. Valid keys are 'points' and 'text', for each of them valid
            values are 'x', 'y' and 'xy'. This way you can forbid moving texts
            along either of the axes due to overlaps with points, but let it
            happen if there is an overlap with texts: only_move={'points':'y',
            'text':'xy'}. Default: None, so everything is allowed.
        text_from_text (bool): whether to repel texts from each other; default
            True
        text_from_points (bool): whether to repel texts from points; default
            True; can helpful to switch of in extremely crouded plots
        save_steps (bool): whether to save intermediate steps as images;
            default False
        save_prefix (str): a path and/or prefix to the saved steps; default ''
        save_format (str): a format to save the steps into; default 'png
        *args and **kwargs: any arguments will be fed into plt.annotate after
            all the optimization is done just for plotting
        add_step_numbers (bool): whether to add step numbers as titles to the
            images of saving steps
        draggable (bool): whether to make the annotations draggable; default
            True
    """
    if ax is None:
        ax = plt.gca()
    r = ax.get_figure().canvas.get_renderer()
    orig_xy = [text.get_position() for text in texts]
    orig_x = [xy[0] for xy in orig_xy]
    orig_y = [xy[1] for xy in orig_xy]
    if x is None:
        if y is None:
            x, y = orig_x, orig_y
        else:
            raise ValueError('Please specify both x and y, or neither')
    if y is None:
        raise ValueError('Please specify both x and y, or neither')
    if add_objects is None:
        text_from_objects = False
        add_bboxes = []
    else:
        try:
            add_bboxes = get_bboxes(add_objects, r)
        except:
            raise ValueError("Can't get bounding boxes from add_objects - is'\
                             it a flat list of matplotlib objects?")
            return
        text_from_objects = True
    for text in texts:
        text.set_va(va)
        text.set_ha(ha)
    if save_steps:
        if add_step_numbers:
            plt.title('0a')
        plt.savefig(save_prefix+'0a.'+save_format, format=save_format)
    if autoalign:
        if autoalign is not True:
            texts = optimally_align_text(x, y, texts, expand=expand_align,
                                         add_bboxes=add_bboxes,
                                         direction=autoalign, renderer=r,
                                         ax=ax)
        else:
            texts = optimally_align_text(orig_x, orig_y, texts,
                                         expand=expand_align,
                                         direction='xy',
                                         add_bboxes=add_bboxes, renderer=r,
                                         ax=ax)

    if save_steps:
        if add_step_numbers:
            plt.title('0b')
        plt.savefig(save_prefix+'0b.'+save_format, format=save_format)
    texts = repel_text_from_axes(texts, ax, renderer=r, expand=expand_points)
    history = [np.inf]*5
    for i in xrange(lim):
        q1, q2 = np.inf, np.inf

        if text_from_text:
            d_x_text, d_y_text, q1 = repel_text(texts, renderer=r, ax=ax,
                                                expand=expand_text)
        else:
            d_x_text, d_y_text, q1 = [0]*len(texts), [0]*len(texts), 0

        if text_from_points:
            d_x_points, d_y_points, q2 = repel_text_from_points(x, y, texts,
                                                   ax=ax, renderer=r,
                                                   expand=expand_points)
        else:
            d_x_points, d_y_points, q2 = [0]*len(texts), [0]*len(texts), 0

        if text_from_objects:
            d_x_objects, d_y_objects, q3 = repel_text_from_bboxes(add_bboxes,
                                                                  texts,
                                                             ax=ax, renderer=r,
                                                         expand=expand_objects)
        else:
            d_x_objects, d_y_objects, q3 = [0]*len(texts), [0]*len(texts), 0

        if only_move:
            if 'text' in only_move:
                if 'x' not in only_move['text']:
                    d_x_text = np.zeros_like(d_x_text)
                if 'y' not in only_move['text']:
                    d_y_text = np.zeros_like(d_y_text)
            if 'points' in only_move:
                if 'x' not in only_move['points']:
                    d_x_points = np.zeros_like(d_x_points)
                if 'y' not in only_move['points']:
                    d_y_points = np.zeros_like(d_y_points)
            if 'objects' in only_move:
                if 'x' not in only_move['objects']:
                    d_x_objects = np.zeros_like(d_x_objects)
                if 'y' not in only_move['objects']:
                    d_y_objects = np.zeros_like(d_y_objects)
        dx = (np.array(d_x_text) * force_text +
              np.array(d_x_points) * force_points +
              np.array(d_x_objects) * force_objects)
        dy = (np.array(d_y_text) * force_text +
              np.array(d_y_points) * force_points +
              np.array(d_y_objects) * force_objects)
        q = round(q1+q2+q3, 5)
        if q > precision and q < np.max(history):
            history.pop(0)
            history.append(q)
            move_texts(texts, dx, dy,
                       bboxes = get_bboxes(texts, r, (1, 1)), ax=ax)
            if save_steps:
                if add_step_numbers:
                    plt.title(i+1)
                plt.savefig(save_prefix+str(i+1)+'.'+save_format,
                            format=save_format)
        else:
            break

    for j, text in enumerate(texts):
        a = ax.annotate(text.get_text(), xy = (orig_xy[j]),
                    xytext=text.get_position(), *args, **kwargs)
        a.__dict__.update(text.__dict__)
        if draggable:
            a.draggable()
        texts[j].remove()
    if save_steps:
        if add_step_numbers:
            plt.title(i+1)
        plt.savefig(save_prefix+str(i+1)+'.'+save_format, format=save_format)
    return i+1

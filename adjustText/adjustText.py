from __future__ import division
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

def get_bboxes(texts, r, expand):
    return [i.get_window_extent(r).expanded(*expand).transformed(plt.gca().\
                                          transData.inverted()) for i in texts]

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
        x1, y1, x2, y2 = bboxes[i].extents
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
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

def repel_text(texts, renderer=None, ax=None, expand=(1.2, 1.2),
               only_use_max_min=False, move=False):
    """
    Repel texts from each other while expanding their bounding boxes by expand
    (x, y), e.g. (1.2, 1.2) would multiply both width and height by 1.2.
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

    overlaps_x = np.zeros((len(bboxes), len(bboxes)))
    overlaps_y = np.zeros((len(bboxes), len(bboxes)))
    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes[i+1:]):
            j += i
            j += 1
            try:
                x, y = bbox1.intersection(bbox1, bbox2).size
                overlaps_x[i, j] = x
                overlaps_y[i, j] = y
            except AttributeError:
                pass
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)
    inds, cols = np.where(overlaps_x!=0)
        
    for i in range(len(inds)):
        i, j = inds[i], cols[i]
        direction = np.sign(bboxes[i].extents - bboxes[j].extents)[:2]
        overlap_directions_x[i, j] = direction[0]
        overlap_directions_y[i, j] = direction[1]
        
    move_x = overlaps_x*overlap_directions_x
    move_y = overlaps_y*overlap_directions_y

    delta_x = move_x.sum(axis=1)-move_x.sum(axis=0)
    delta_y = move_y.sum(axis=1)-move_y.sum(axis=0)
    
    q = np.sum(np.abs(delta_x) + np.abs(delta_y))
    if move:
        move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
    return delta_x, delta_y, q

def get_midpoint(bbox):
    cx = (bbox.x0+bbox.x1)/2
    cy = (bbox.y0+bbox.y1)/2
    return cx, cy

def repel_text_from_points(x, y, texts, renderer=None, ax=None,
                           prefer_move='y', expand=(1.2, 1.2), move=False):
    """
    Repel texts from all points specified by x and y while expanding their
    (texts'!) bounding boxes by expandby  (x, y), e.g. (1.2, 1.2)
    would multiply both width and height by 1.2. In the case when the text
    overlaps a point, but there is no definite direction for movement (read,
    the point is in the very center), moves in random direction by 40% of it's
    width and/or height depending on prefer_move: 'x' moves along x, 'y' -
    along 'y', 'xy' - along both.
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
        for j, (xp, yp) in enumerate(zip(x, y)):
            if bbox.contains(xp, yp):
                cx, cy = get_midpoint(bbox)

                dir_x = np.sign(round(cx-xp, 3))
                dir_y = np.sign(round(cy-yp, 3))

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
                
                if dx == 0 and dy == 0:
                    if 'x' in prefer_move:
                        dx = bbox.width*0.4*np.random.choice([-1, 1])
                    if 'y' in prefer_move:
                        dy = bbox.height*0.4*np.random.choice([-1, 1])

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
        x1, y1, x2, y2 = bboxes[i].extents
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
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

def pull_text_to_respective_points(x, y, texts, renderer=None, ax=None,
                                   fraction=0.1, expand=(1.2, 1.2)):
    """
    Probably is never useful.
    """
    if ax is None:
        ax = plt.gca()
    if renderer is None:
        r = ax.get_figure().canvas.get_renderer()
    else:
        r = renderer
    bboxes = [i.get_window_extent(r).expanded(*expand).transformed(plt.gca().\
                                          transData.inverted()) for i in texts]

    for i, (bbox, xp, yp) in enumerate(zip(bboxes, x, y)):
        if not bbox.contains(xp, yp):
            cx, cy = get_midpoint(bbox)
            dx = cx - xp
            dy = cy - yp
            x, y = texts[i].get_position()
            if dx/x > dy/y:
                newx = x - dx * fraction
            else:
                newx = x
            if dy/y > dx/x:
                newy = y - dy * fraction
            else:
                newy = y
            texts[i].set_position((newx, newy))
    return texts

def adjust_text(x, y, texts, ax=None, expand_text = (1.2, 1.2),
                expand_points=(1.2, 1.2), prefer_move = 'xy', 
                force_text=0.5, force_points=1.0, lim=100, precision=0.1,
                pullback_fraction=0.0, only_move={},
                ha = 'center', va = 'center',
                text_from_text=True,
                text_from_points=True, save_steps=False, save_prefix='',
                save_format='png', add_step_numbers=True, draggable=True,
                *args, **kwargs):
    """
    Iteratively adjusts the locations of texts. In each iteration first moves
    all texts away from each other, then all texts away from points. In the end
    hides texts and substitutes them with annotations to link them to the
    rescpective points.
    
    Args:
        x (seq): x-coordinates of labelled points
        y (seq): y-coordinates of labelled points
        texts (list): a list of text.Text objects to adjust
        ax (obj): axes object with the plot; if not provided is determined by
            plt.gca()
        expand_text (seq): a tuple/list/... with 2 numbers (x, y) to expand
            texts when repelling them from each other; default (1.2, 1.2)
        expand_points (seq): a tuple/list/... with 2 numbers (x, y) to expand
            texts when repelling them from points; default (1.2, 1.2)
        prefer_move (str or seq(str, str)): specifies where to move the texts
            (along 'x', 'y' or both - 'xy') when unsure, e.g. just away from
            the respective point. Direction along each axis is random, thus
            introduces unpredictability (except when np.seed is set); default
            'xy'
        force_text (float): the repel force from texts is multiplied by this
            value; default 0.5
        force_points (float): the repel force from points is multiplied by this
            value; default 1.0
        lim (int): limit of number of iterations
        precision (float): up to which sum of all overlaps along both x and y
            to iterate
        pullback_fraction (float): a fraction of distance between each text and
            its corresponding point to pull the text back to it; probably never
            useful and should stay 0
        ha (str): horizontal alignment of the texts ("left", "center" or
            "right"). Has a strong effect in the very first cycle; default
            "center"
        va (str): vertical alignment of the texts ("bottom", "center" or
            "top").  Has a strong effect in the very first cycle; default
            "center"
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
    for text in texts:
        text.set_horizontalalignment(ha)
        text.set_verticalalignment(va)
    if save_steps:
        if add_step_numbers:
            plt.title(0)
        plt.savefig(save_prefix+'0.'+save_format, format=save_format)
    repel_text_from_axes(texts, ax, renderer=r)
    for i in range(lim):
        q1, q2 = np.inf, np.inf
        if text_from_text:
            d_x_text, d_y_text, q1 = repel_text(texts, renderer=r, ax=ax,
                                                expand=expand_text)
        else:
            d_x_text, d_y_text, q1 = [0]*len(texts), [0]*len(texts), 0
        if text_from_points:
            d_x_points, d_y_points, q2 = repel_text_from_points(x, y, texts,
                                                   ax=ax, renderer=r,
                                                   expand=expand_points,
                                                   prefer_move=prefer_move)
        else:
            d_x_points, d_y_points, q1 = [0]*len(texts), [0]*len(texts), 0
        if only_move:
            if 'x' not in only_move['text']:
                d_x_text = np.zeros_like(d_x_text)
            if 'y' not in only_move['text']:
                d_y_text = np.zeros_like(d_y_text)
            if 'x' not in only_move['points']:
                d_x_points = np.zeros_like(d_x_points)
            if 'y' not in only_move['points']:
                d_y_points = np.zeros_like(d_y_points)
        dx = np.array(d_x_text) + np.array(d_x_points)
        dy = np.array(d_y_text) + np.array(d_y_points)
        move_texts(texts, dx*force_text, dy*force_points,
                   bboxes = get_bboxes(texts, r, (1, 1)), ax=ax)
        if save_steps:
            if add_step_numbers:
                plt.title(i+1)
            plt.savefig(save_prefix+str(i+1)+'.'+save_format,
                        format=save_format)
        q = np.array([q1, q2])[np.array([q1, q2])<np.inf]
        if i>=5 and np.all(q <= precision):
            break
        
    for j, text in enumerate(texts):
        a = ax.annotate(text.get_text(), xy = (x[j], y[j]),
                    xytext=text.get_position(),
                    horizontalalignment=ha,
                    verticalalignment=va, *args, **kwargs)
        if draggable:
            a.draggable()
        texts[j].set_visible(False)
    if save_steps:
        if add_step_numbers:
            plt.title(i+1)
        plt.savefig(save_prefix+str(i+1)+'.'+save_format, format=save_format)

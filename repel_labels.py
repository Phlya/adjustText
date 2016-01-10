import numpy as np
from matplotlib import pyplot as plt

def repel_text_annotaions(x, y, texts):
        bboxes = [i.get_window_extent(r).transformed(plt.gca().transData.inverted()) for i in texts]

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

        delta_x = move_x.sum(axis=1)
        delta_y = move_y.sum(axis=1)
        print delta_x
        q = np.sum(np.abs(delta_x) + np.abs(delta_y))

        for i, text in enumerate(texts):
                x, y = text.get_position()
                newx = x + delta_x[i]
                newy = y + delta_y[i]
                text.set_position((newx, newy))
        return texts, q

def iteratively_repel_text_annotaions(x, y, texts, lim=100):
        history = [np.inf]*3
        for i in range(lim):
                print i
                texts, q = repel_text_annotaions(x, y, texts)
                if q > max(history):
                        print 'Done at sum of overlapping distances', q
                        break
                history.pop(0)
                history.append(q)
        for i, text in enumerate(texts):
                if (xs[i], ys[i]) != text.get_position():
                        plt.annotate(text.get_text(), xy = (x[i], y[i]), xytext=text.get_position(),
                                                 arrowprops=dict(arrowstyle='->', color='red'), bbox={'pad':0, 'alpha':0})
                        texts[i].set_visible(False)

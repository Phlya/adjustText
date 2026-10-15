import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adjustText import adjust_text, get_bboxes_pathcollection


@pytest.mark.parametrize("scale", ["linear", "log"])
@pytest.mark.parametrize("empty", [False, True])
def test_multiple_scatter_collections_match_explicit_bboxes(scale, empty):
    fig, ax = plt.subplots()
    try:
        ax.set_xscale(scale)
        first = ax.scatter([1, 2], [1, 2], s=[200, 400], marker="s")
        second = ax.scatter([] if empty else [3, 4], [] if empty else [2, 1], s=300)
        ax.set(xlim=(0.5, 5), ylim=(0, 3))
        positions = [(1.05, 1.05), (2.1, 2.05)]
        texts = [ax.text(x, y, "label") for x, y in positions]
        fig.canvas.draw()
        expected_boxes = get_bboxes_pathcollection(first, ax) + get_bboxes_pathcollection(second, ax)
        adjust_text(texts, objects=expected_boxes, ax=ax, iter_lim=10)
        expected = [text.get_position() for text in texts]
        for text in texts:
            text.remove()
        texts = [ax.text(x, y, "label") for x, y in positions]
        before = [collection.get_offsets().copy() for collection in (first, second)]
        adjust_text(texts, objects=[first, second], ax=ax, iter_lim=10)
        np.testing.assert_allclose([text.get_position() for text in texts], expected)
        for collection, offsets in zip((first, second), before):
            np.testing.assert_array_equal(collection.get_offsets(), offsets)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["single", "list", "empty-list"])
def test_empty_scatter_objects(kind):
    fig, ax = plt.subplots()
    try:
        collection = ax.scatter([], [])
        objects = {"single": collection, "list": [collection], "empty-list": []}[kind]
        texts = [ax.text(0.5, 0.5, "label")]
        adjust_text(texts, objects=objects, ax=ax, iter_lim=2)
        assert np.isfinite(texts[0].get_position()).all()
    finally:
        plt.close(fig)


def test_labels_avoid_marker_bounds_after_collection_updates():
    from matplotlib.transforms import Bbox

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    try:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        collections = [ax.scatter([x], [0.5], s=900, marker="s") for x in [0.3, 0.7]]
        for ypos in [0.5, 0.7]:
            for collection, xpos in zip(collections, [0.3, 0.7]):
                collection.set_offsets([[xpos, ypos]])
            texts = [ax.text(x, ypos, "label") for x in [0.3, 0.7]]
            adjust_text(
                texts, objects=collections, ax=ax, iter_lim=100,
                force_pull=0, force_text=0, force_explode=0, force_static=1,
                avoid_self=False,
            )
            # A square marker of area 900 pt^2 has a 30-point side.
            half_side = 15 * fig.dpi / 72
            for collection in collections:
                x, y = ax.transData.transform(collection.get_offsets()[0])
                marker = Bbox.from_extents(x - half_side, y - half_side, x + half_side, y + half_side)
                for text in texts:
                    assert not text.get_window_extent().overlaps(marker)
            for text in texts:
                text.remove()
    finally:
        plt.close(fig)


def test_one_collection_in_list_matches_direct_collection():
    fig, ax = plt.subplots()
    try:
        collection = ax.scatter([1, 2], [1, 2], s=[100, 300])
        outputs = []
        for objects in [collection, [collection], (collection,)]:
            text = ax.text(1.05, 1.05, "label")
            adjust_text([text], objects=objects, ax=ax, iter_lim=10)
            outputs.append(text.get_position())
            text.remove()
        np.testing.assert_allclose(outputs, [outputs[0]] * 3)
    finally:
        plt.close(fig)

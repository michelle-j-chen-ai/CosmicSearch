"""In-memory atlas: coordinates, categorical colorings, and exact kNN.

The whole artifact is resident. At 250k rows that is ~2MB of coordinates, ~50MB
of PCA vectors and ~40MB of metadata strings, so the app never reads S3 on the
request path -- the only S3 traffic after startup is presigning MP4 URLs.

Coordinates and color indices are served as raw little-endian binary rather than
JSON: 250k points is ~2MB as a Float32Array versus ~12MB of JSON text that also
has to be parsed on the main thread.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

# Fields offered as categorical colorings in the UI. `dt` exposes drive date
# (dataset composition over time); `run_uuid` shows whether a region is one
# drive, which is the tell for a map keyed on nuisance variables rather than
# semantics.
COLOR_FIELDS = ("dt", "run_uuid")

# Categories beyond this are merged into an "other" bucket. Past a few dozen
# hues are not distinguishable anyway, and run_uuid has thousands of values.
MAX_CATEGORIES = 24

# Grid resolution per axis for the density coloring.
DENSITY_BINS = 256


class AtlasStore:
    def __init__(self, path: Path) -> None:
        table = pq.read_table(path)
        self.count = table.num_rows
        self.x = table["x"].to_numpy(zero_copy_only=False).astype(np.float32)
        self.y = table["y"].to_numpy(zero_copy_only=False).astype(np.float32)
        self.chunk_id = table["chunk_id"].to_pylist()
        self.run_uuid = table["run_uuid"].to_pylist()
        self.dt = table["dt"].to_pylist()
        self.chunk_start_unix = table["chunk_start_unix"].to_pylist()
        self.source_media_uri = table["source_media_uri"].to_pylist()

        # Row-normalized so a dot product is cosine similarity, which is the
        # metric the retrieval path these embeddings are evaluated by uses.
        pca = np.stack(table["pca"].to_numpy(zero_copy_only=False)).astype(np.float32)
        self.pca = pca / np.maximum(np.linalg.norm(pca, axis=1, keepdims=True), 1e-12)

        self._colorings = {field: self._build_coloring(field) for field in COLOR_FIELDS}
        self._colorings["density"] = self._build_density_coloring()
        LOGGER.info(
            "atlas loaded: %d points, pca dim %d, colorings %s",
            self.count,
            self.pca.shape[1],
            sorted(self._colorings),
        )

    @property
    def positions(self) -> bytes:
        """Interleaved xy as Float32Array, consumed directly by deck.gl."""
        return np.column_stack([self.x, self.y]).astype("<f4").tobytes()

    def coloring(self, field: str) -> tuple[bytes, list[str]]:
        """Per-point category index as Uint8Array, plus the legend labels."""
        assert field in self._colorings, f"unknown color field {field!r}"
        indices, legend = self._colorings[field]
        return indices.tobytes(), legend

    @property
    def color_fields(self) -> list[str]:
        return sorted(self._colorings)

    def _build_coloring(self, field: str) -> tuple[np.ndarray, list[str]]:
        values = getattr(self, field)
        unique, counts = np.unique(np.asarray(values, dtype=object), return_counts=True)
        # Keep the most populous categories: they are the ones with enough
        # points on screen for a hue to mean anything.
        ranked = unique[np.argsort(-counts)][: MAX_CATEGORIES - 1]
        legend = sorted(str(v) for v in ranked)
        lookup = {label: i for i, label in enumerate(legend)}
        other = len(legend)
        indices = np.fromiter(
            (lookup.get(str(v), other) for v in values), dtype=np.uint8, count=len(values)
        )
        return indices, legend + ["other"]

    def _build_density_coloring(self) -> tuple[np.ndarray, list[str]]:
        """Quantile-bucketed local point count.

        Overplotting hides density: 250k points in a few thousand pixels means a
        sparse tail and a saturated core look identical. Quantile buckets (rather
        than linear ones) keep every bucket populated whatever the shape.
        """
        counts, x_edges, y_edges = np.histogram2d(
            self.x, self.y, bins=DENSITY_BINS, range=[[-1, 1], [-1, 1]]
        )
        # `np.digitize` returns 1..len(edges); subtract 1 and clip to land inside
        # the histogram, since the max coordinate sits exactly on the last edge.
        xi = np.clip(np.digitize(self.x, x_edges) - 1, 0, DENSITY_BINS - 1)
        yi = np.clip(np.digitize(self.y, y_edges) - 1, 0, DENSITY_BINS - 1)
        per_point = counts[xi, yi]

        buckets = 8
        edges = np.quantile(per_point, np.linspace(0, 1, buckets + 1)[1:-1])
        indices = np.digitize(per_point, edges).astype(np.uint8)
        legend = [f"q{i + 1}" for i in range(buckets)]
        return indices, legend

    def detail(self, index: int) -> dict:
        assert 0 <= index < self.count, f"index {index} out of range"
        return {
            "index": index,
            "chunk_id": self.chunk_id[index],
            "run_uuid": self.run_uuid[index],
            "dt": self.dt[index],
            "chunk_start_unix": self.chunk_start_unix[index],
            "source_media_uri": self.source_media_uri[index],
            "x": float(self.x[index]),
            "y": float(self.y[index]),
        }

    def neighbors(self, index: int, k: int) -> list[dict]:
        """Exact kNN in PCA space, ranked by cosine similarity.

        Deliberately not a 2D-coordinate lookup. UMAP preserves local
        neighbourhoods but distorts everything else, so "what is actually near
        this clip" must be answered in the projected embedding space; the map is
        only the navigation surface.
        """
        assert 0 <= index < self.count, f"index {index} out of range"
        similarities = self.pca @ self.pca[index]
        # k+1 because the query point is its own nearest neighbour.
        top = np.argpartition(-similarities, min(k, self.count - 1))[: k + 1]
        top = top[np.argsort(-similarities[top])]
        return [
            {**self.detail(int(i)), "similarity": float(similarities[i])}
            for i in top
            if int(i) != index
        ][:k]

    def lasso(self, polygon: list[list[float]], limit: int) -> list[int]:
        """Point indices inside a polygon, nearest-to-centroid first.

        Ordering by centroid distance rather than arbitrarily means the returned
        sample is representative of the middle of the selection instead of its
        edge.
        """
        inside = np.flatnonzero(_points_in_polygon(self.x, self.y, np.asarray(polygon)))
        if len(inside) == 0:
            return []
        centroid = np.asarray(polygon).mean(axis=0)
        distances = (self.x[inside] - centroid[0]) ** 2 + (self.y[inside] - centroid[1]) ** 2
        return [int(i) for i in inside[np.argsort(distances)][:limit]]


def _points_in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray casting over all points at once."""
    inside = np.zeros(len(x), dtype=bool)
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        # Edge straddles the point's horizontal ray, and the crossing is to the
        # right of the point.
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (x < crossing_x)
    return inside

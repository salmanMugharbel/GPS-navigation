"""Road-network model and a synthetic city generator.

Vertices  = road intersections
Edges     = road segments (directed, so one-way streets are possible)
Weights   = travel time in TENTHS of a second (integers, so every algorithm
            gives bit-for-bit identical answers and results are easy to compare)
"""
import random
from dataclasses import dataclass, field

INF = float("inf")


class Graph:
    """Directed weighted graph stored as forward and reverse adjacency lists."""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]    # adj[u]  = [(v, w), ...]  edges u -> v
        self.radj = [[] for _ in range(n)]   # radj[v] = [(u, w), ...]  edges u -> v, seen from v
        self.coords = [(0, 0)] * n           # (x, y) in metres, used for geographic partitioning
        self.m = 0

    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
        self.radj[v].append((u, w))
        self.m += 1


@dataclass
class Route:
    """Result of a shortest-path query."""
    distance: float                         # travel time in tenths of a second (INF = unreachable)
    path: list = field(default_factory=list)  # list of vertex ids from source to destination
    settled: int = 0                        # vertices settled = "search space" size

    @property
    def seconds(self):
        return self.distance / 10


def path_length(g, path):
    """Travel time of a vertex path (cheapest parallel edge is used). Raises if an edge is missing."""
    total = 0
    for a, b in zip(path, path[1:]):
        ws = [w for v, w in g.adj[a] if v == b]
        if not ws:
            raise ValueError(f"no edge {a} -> {b}")
        total += min(ws)
    return total


def generate_city(rows, cols, seed=42, block_m=100, oneway_ratio=0.08, highway_every=12):
    """Build a rows x cols street grid.

    * local streets: 30-50 km/h, random traffic slow-down 1.0x - 2.2x per direction
    * every `highway_every`-th row/column is a fast road (90 km/h, light traffic)
    * `oneway_ratio` of local streets are one-way
    Travel time = distance / speed * traffic, stored in tenths of a second.
    """
    rng = random.Random(seed)
    g = Graph(rows * cols)
    for r in range(rows):
        for c in range(cols):
            g.coords[r * cols + c] = (c * block_m, r * block_m)

    def travel_time(highway):
        speed = 90 if highway else rng.choice((30, 40, 50))          # km/h
        traffic = rng.uniform(1.0, 1.4) if highway else rng.uniform(1.0, 2.2)
        return max(1, round(block_m / (speed / 3.6) * traffic * 10))

    def add_segment(u, v, highway):
        if not highway and rng.random() < oneway_ratio:
            a, b = (u, v) if rng.random() < 0.5 else (v, u)
            g.add_edge(a, b, travel_time(False))
        else:
            g.add_edge(u, v, travel_time(highway))
            g.add_edge(v, u, travel_time(highway))

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                add_segment(u, u + 1, r % highway_every == 0)
            if r + 1 < rows:
                add_segment(u, u + cols, c % highway_every == 0)
    return g


def random_queries(g, k, seed=7, min_fraction=0.5):
    """k (source, destination) pairs that are far apart (>= min_fraction of the map diagonal)."""
    rng = random.Random(seed)
    xs = [p[0] for p in g.coords]
    ys = [p[1] for p in g.coords]
    diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    pairs = []
    while len(pairs) < k:
        s, t = rng.randrange(g.n), rng.randrange(g.n)
        dx = g.coords[s][0] - g.coords[t][0]
        dy = g.coords[s][1] - g.coords[t][1]
        if (dx * dx + dy * dy) ** 0.5 >= min_fraction * diag:
            pairs.append((s, t))
    return pairs

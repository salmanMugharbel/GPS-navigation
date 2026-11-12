"""Evaluate several candidate routes at the same time (one candidate per worker task).

A candidate is "go from s to t via intersection v" for v sampled from a corridor around the
straight line s-t.  Each candidate costs two shortest-path searches, and candidates are
completely independent, so they spread over all cores.  The fastest candidate wins.
The result is an excellent route that is guaranteed valid, and exactly optimal whenever
one of the sampled intersections lies on a true shortest path.
"""
import multiprocessing as mp
import random

from .bidirectional import bidirectional_dijkstra
from .graph import INF

_G = None


def _init_worker(g):
    global _G
    _G = g


def _evaluate(task, g=None):
    g = g or _G
    s, via, t = task
    a = bidirectional_dijkstra(g, s, via)
    b = bidirectional_dijkstra(g, via, t)
    if a.distance == INF or b.distance == INF:
        return (INF, via, [])
    return (a.distance + b.distance, via, a.path + b.path[1:])


def corridor_vias(g, s, t, k, seed=0, width=0.25):
    """k intersections whose projection falls between s and t and that lie within `width` x |st| of the line."""
    rng = random.Random(seed)
    (sx, sy), (tx, ty) = g.coords[s], g.coords[t]
    dx, dy = tx - sx, ty - sy
    length2 = dx * dx + dy * dy or 1
    vias, tries = [], 0
    while len(vias) < k and tries < 200 * k:
        tries += 1
        v = rng.randrange(g.n)
        if v in (s, t):
            continue
        px, py = g.coords[v]
        proj = ((px - sx) * dx + (py - sy) * dy) / length2
        off = abs((px - sx) * dy - (py - sy) * dx) / (length2 ** 0.5)
        if 0.1 <= proj <= 0.9 and off <= width * length2 ** 0.5:
            vias.append(v)
    return vias


class CandidateEvaluator:
    def __init__(self, g, processes=None):
        self.g = g
        self.pool = mp.Pool(processes, initializer=_init_worker, initargs=(g,))

    def best_route(self, s, t, k=16, seed=0):
        """Parallel: returns (best (time, via, path), all results)."""
        tasks = [(s, v, t) for v in corridor_vias(self.g, s, t, k, seed)]
        results = self.pool.map(_evaluate, tasks, chunksize=1)
        return min(results, key=lambda r: r[0]), results

    def best_route_sequential(self, s, t, k=16, seed=0):
        """Same work on one core, for timing comparison."""
        tasks = [(s, v, t) for v in corridor_vias(self.g, s, t, k, seed)]
        results = [_evaluate(task, self.g) for task in tasks]
        return min(results, key=lambda r: r[0]), results

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

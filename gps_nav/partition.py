"""Geographic partitioning: split the map into px x py regions, one job per processor.

Precomputation (parallel, one region per task):
    * boundary vertices = vertices with a road segment that crosses into another region
    * for every boundary vertex b, run Dijkstra INSIDE its region to get the travel time to
      every other boundary vertex of that region  -> "overlay" edges
    * plus the original region-crossing road segments
Query s -> t:
    1. Dijkstra inside s's region from s (reaches the region's boundary vertices)
    2. reverse Dijkstra inside t's region from t
    3. Dijkstra on the small overlay graph between the two
    (if s and t share a region we also keep the purely local route)
The answer equals the ordinary shortest distance; only the work is divided.
"""
import heapq
import multiprocessing as mp
import time

from .graph import INF


def _restricted_dijkstra(adj, region, r, src, must_settle):
    """Dijkstra that never leaves region r. Stops once all of `must_settle` are final."""
    dist = {src: 0}
    pq = [(0, src)]
    done = {}
    need, got = len(must_settle), 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done[u] = d
        if u in must_settle:
            got += 1
            if got == need:
                break
        for v, w in adj[u]:
            if region[v] != r:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return done


def _region_overlay(g, region, bset, r):
    """Overlay edges that start in region r: {b: [(b2, w), ...]}."""
    edges = {}
    for b in bset[r]:
        lst = [(v, w) for v, w in g.adj[b] if region[v] != r]            # crossing roads
        done = _restricted_dijkstra(g.adj, region, r, b, bset[r])
        lst.extend((b2, done[b2]) for b2 in bset[r] if b2 != b and b2 in done)
        edges[b] = lst
    return edges


_G = _REGION = _BSET = None


def _init_worker(g, region, bset):
    global _G, _REGION, _BSET
    _G, _REGION, _BSET = g, region, bset


def _work(r):
    return _region_overlay(_G, _REGION, _BSET, r)


class GeoPartition:
    def __init__(self, g, px=2, py=2, processes=None):
        """processes=1 builds sequentially; otherwise regions are processed by a worker pool."""
        self.g = g
        self.k = px * py
        xs = [c[0] for c in g.coords]
        ys = [c[1] for c in g.coords]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        self.region = [
            min(px - 1, int((x - x0) / (x1 - x0 + 1) * px)) + px * min(py - 1, int((y - y0) / (y1 - y0 + 1) * py))
            for x, y in g.coords
        ]
        self.bset = [set() for _ in range(self.k)]
        for u in range(g.n):
            for v, _ in g.adj[u]:
                if self.region[v] != self.region[u]:
                    self.bset[self.region[u]].add(u)
                    self.bset[self.region[v]].add(v)
        t0 = time.perf_counter()
        if processes == 1:
            parts = [_region_overlay(g, self.region, self.bset, r) for r in range(self.k)]
        else:
            with mp.Pool(processes, initializer=_init_worker, initargs=(g, self.region, self.bset)) as pool:
                parts = pool.map(_work, range(self.k), chunksize=1)
        self.build_seconds = time.perf_counter() - t0
        self.overlay = {}
        for p in parts:
            self.overlay.update(p)
        self.boundary_count = sum(len(b) for b in self.bset)

    def distance(self, s, t):
        """Shortest travel time (tenths of a second) from s to t, or INF."""
        g, region = self.g, self.region
        rs, rt = region[s], region[t]
        must = set(self.bset[rs]) | ({t} if rs == rt else set())
        done_s = _restricted_dijkstra(g.adj, region, rs, s, must)
        best = done_s.get(t, INF) if rs == rt else INF
        done_t = _restricted_dijkstra(g.radj, region, rt, t, self.bset[rt])
        label, pq = {}, []
        for b in self.bset[rs]:
            if b in done_s:
                label[b] = done_s[b]
                pq.append((done_s[b], b))
        heapq.heapify(pq)
        while pq:
            d, u = heapq.heappop(pq)
            if d >= best:
                break
            if d > label[u]:
                continue
            if region[u] == rt and u in done_t and d + done_t[u] < best:
                best = d + done_t[u]
            for v, w in self.overlay.get(u, ()):
                nd = d + w
                if nd < label.get(v, INF):
                    label[v] = nd
                    heapq.heappush(pq, (nd, v))
        return best

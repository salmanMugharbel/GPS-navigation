"""Contraction Hierarchies (CH): preprocessing that adds shortcut edges, then very fast queries.

Preprocessing: repeatedly "contract" the least important vertex x.  For every pair
(u -> x -> v) we add a shortcut u -> v with weight w(u,x)+w(x,v), unless a "witness" path
u ~> v that avoids x is at least as short.  Every vertex gets a rank (its contraction order).

Query: run Dijkstra from the source using only edges that go to a HIGHER rank, and a second
Dijkstra from the destination over reversed edges, also only upwards.  The best common
vertex gives the shortest distance.  Both searches are tiny compared with plain Dijkstra.
"""
import heapq
import time

from .graph import INF, Route

MAX_WITNESS_SETTLED = 400   # witness searches are cut off here (extra shortcuts are still correct)


class ContractionHierarchy:
    def __init__(self, g):
        t0 = time.perf_counter()
        self.n = g.n
        self.via = {}                       # (u, v) -> contracted vertex a shortcut skips over
        self.shortcuts_added = 0
        self._build(g)
        self.build_seconds = time.perf_counter() - t0

    # ---------------------------------------------------------------- preprocessing
    def _find_shortcuts(self, x, out, inn, contracted):
        ins = [(u, w) for u, w in inn[x].items() if not contracted[u]]
        outs = [(v, w) for v, w in out[x].items() if not contracted[v]]
        found = []
        if not ins or not outs:
            return ins, outs, found
        max_out = max(w for _, w in outs)
        for u, wu in ins:
            limit = wu + max_out
            dist = {u: 0}
            pq = [(0, u)]
            done = 0
            while pq and done < MAX_WITNESS_SETTLED:
                d, a = heapq.heappop(pq)
                if d > dist[a]:
                    continue
                if d > limit:
                    break
                done += 1
                for b, w in out[a].items():
                    if b == x or contracted[b]:
                        continue
                    nd = d + w
                    if nd < dist.get(b, INF):
                        dist[b] = nd
                        heapq.heappush(pq, (nd, b))
            for v, wv in outs:
                if v != u and dist.get(v, INF) > wu + wv:      # no witness => need shortcut
                    found.append((u, v, wu + wv))
        return ins, outs, found

    def _build(self, g):
        n = g.n
        out = [dict() for _ in range(n)]
        inn = [dict() for _ in range(n)]
        for u in range(n):
            for v, w in g.adj[u]:
                if v != u and w < out[u].get(v, INF):
                    out[u][v] = w
                    inn[v][u] = w
        contracted = [False] * n
        deleted_nb = [0] * n
        rank = [0] * n

        def priority(x):
            ins, outs, found = self._find_shortcuts(x, out, inn, contracted)
            return (len(found) - len(ins) - len(outs)) + deleted_nb[x]

        pq = [(priority(x), x) for x in range(n)]
        heapq.heapify(pq)
        r = 0
        while pq:
            _, x = heapq.heappop(pq)
            if contracted[x]:
                continue
            ins, outs, found = self._find_shortcuts(x, out, inn, contracted)
            p = (len(found) - len(ins) - len(outs)) + deleted_nb[x]
            if pq and p > pq[0][0]:                    # lazy update: priority got worse
                heapq.heappush(pq, (p, x))
                continue
            for u, v, w in found:
                if w < out[u].get(v, INF):
                    out[u][v] = w
                    inn[v][u] = w
                    self.via[(u, v)] = x
                    self.shortcuts_added += 1
            contracted[x] = True
            rank[x] = r
            r += 1
            for nb in {u for u, _ in ins} | {v for v, _ in outs}:
                deleted_nb[nb] += 1
        self.rank = rank
        # upward graphs: forward search follows edges to higher rank; backward search follows
        # reversed edges to higher rank
        self.up_out = [[] for _ in range(n)]
        self.up_in = [[] for _ in range(n)]
        for u in range(n):
            for v, w in out[u].items():
                if rank[v] > rank[u]:
                    self.up_out[u].append((v, w))
                else:
                    self.up_in[v].append((u, w))

    # ---------------------------------------------------------------- query
    @staticmethod
    def _upward_search(src, up):
        dist, parent = {src: 0}, {src: -1}
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in up[u]:
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, parent

    def _unpack(self, a, b, path):
        """Expand a (possibly shortcut) edge a->b into original vertices, appending to path."""
        stack = [(a, b)]
        while stack:
            u, v = stack.pop()
            x = self.via.get((u, v))
            if x is None:
                path.append(v)
            else:
                stack.append((x, v))
                stack.append((u, x))

    def query(self, s, t):
        if s == t:
            return Route(0, [s], 0)
        df, pf = self._upward_search(s, self.up_out)
        db, pb = self._upward_search(t, self.up_in)
        best, meet = INF, -1
        for v, d in df.items():
            o = db.get(v)
            if o is not None and d + o < best:
                best, meet = d + o, v
        settled = len(df) + len(db)
        if meet == -1:
            return Route(INF, [], settled)
        up_edges, v = [], meet                     # s ... meet (forward, upward edges)
        while pf[v] != -1:
            up_edges.append((pf[v], v))
            v = pf[v]
        up_edges.reverse()
        path = [s]
        for a, b in up_edges:
            self._unpack(a, b, path)
        v = meet                                   # meet ... t (original direction edges)
        while pb[v] != -1:
            self._unpack(v, pb[v], path)
            v = pb[v]
        return Route(best, path, settled)

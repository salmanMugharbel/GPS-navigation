"""Bidirectional Dijkstra: sequential version and a 2-process parallel version.

Search forward from the source and backward from the destination; stop when the two
searches have "met in the middle".  Each search only covers ~half the radius, so the
explored area (which grows with radius^2 on a map) is roughly halved or better.
"""
import heapq
import multiprocessing as mp

from .graph import INF, Route


def _build_path(meet, pf, pb):
    left, u = [], meet
    while u != -1:
        left.append(u)
        u = pf[u]
    left.reverse()
    u = pb[meet]
    while u != -1:
        left.append(u)
        u = pb[u]
    return left


def bidirectional_dijkstra(g, s, t):
    """Sequential bidirectional Dijkstra (alternates between the two searches)."""
    if s == t:
        return Route(0, [s], 0)
    adj, radj = g.adj, g.radj
    df, db = {s: 0}, {t: 0}
    pf, pb = {s: -1}, {t: -1}
    pqf, pqb = [(0, s)], [(0, t)]
    mu, meet, settled = INF, -1, 0
    while pqf and pqb:
        if pqf[0][0] + pqb[0][0] >= mu:       # searches have met: mu is optimal
            break
        if pqf[0][0] <= pqb[0][0]:
            d, u = heapq.heappop(pqf)
            if d > df[u]:
                continue
            settled += 1
            for v, w in adj[u]:
                nd = d + w
                if nd < df.get(v, INF):
                    df[v] = nd
                    pf[v] = u
                    heapq.heappush(pqf, (nd, v))
                    ob = db.get(v)
                    if ob is not None and nd + ob < mu:
                        mu, meet = nd + ob, v
        else:
            d, u = heapq.heappop(pqb)
            if d > db[u]:
                continue
            settled += 1
            for v, w in radj[u]:
                nd = d + w
                if nd < db.get(v, INF):
                    db[v] = nd
                    pb[v] = u
                    heapq.heappush(pqb, (nd, v))
                    of = df.get(v)
                    if of is not None and nd + of < mu:
                        mu, meet = nd + of, v
    if meet == -1:
        return Route(INF, [], settled)
    return Route(mu, _build_path(meet, pf, pb), settled)


# --------------------------------------------------------------------------------------
# Parallel version: forward search in process A, backward search in process B.
# They talk through shared memory (distance arrays + best-known path length "mu").
# --------------------------------------------------------------------------------------
def _worker(role, g, sf, sb, top_f, top_b, mu, meet, stop, lock, barrier, conn):
    mine, other = (sf, sb) if role == "f" else (sb, sf)
    top_mine, top_other = (top_f, top_b) if role == "f" else (top_b, top_f)
    graph_adj = g.adj if role == "f" else g.radj
    touched = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        s, t = msg
        for i in touched:                      # clear what the previous query wrote
            mine[i] = INF
        start = s if role == "f" else t
        mine[start] = 0
        touched = [start]
        dist, parent = {start: 0}, {start: -1}
        barrier.wait(60)                       # both sides initialised before anyone reads
        pq, settled = [(0, start)], 0
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            top_mine.value = d                 # lower bound on every key still in my heap
            # Stop rule: forward key + backward key >= best path found (both are lower bounds,
            # stale reads only make us stop later, never too early).
            if d + top_other.value >= mu.value:
                stop.value = 1
                break
            if stop.value:
                break
            settled += 1
            for v, w in graph_adj[u]:
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
                    touched.append(v)
                    with lock:                 # publish my label, read the other side's label
                        mine[v] = nd
                        ob = other[v]
                        if ob < INF and nd + ob < mu.value:
                            mu.value = nd + ob
                            meet.value = v
        top_mine.value = INF                   # I am done: never block the other side
        conn.send((parent, settled))


class ParallelBidirectional:
    """Two worker processes (forward / backward) that are started once and reused.

    Use as a context manager:   with ParallelBidirectional(g) as pb: pb.query(s, t)
    """

    def __init__(self, g):
        ctx = mp.get_context()
        self.n = g.n
        self.sf = ctx.Array("d", [INF] * g.n, lock=False)
        self.sb = ctx.Array("d", [INF] * g.n, lock=False)
        self.top_f = ctx.Value("d", 0.0, lock=False)
        self.top_b = ctx.Value("d", 0.0, lock=False)
        self.mu = ctx.Value("d", INF, lock=False)
        self.meet = ctx.Value("i", -1, lock=False)
        self.stop = ctx.Value("b", 0, lock=False)
        lock, barrier = ctx.Lock(), ctx.Barrier(2)
        self.conns, self.procs = [], []
        for role in ("f", "b"):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(role, g, self.sf, self.sb, self.top_f, self.top_b,
                      self.mu, self.meet, self.stop, lock, barrier, child_conn),
                daemon=True,
            )
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def query(self, s, t):
        if s == t:
            return Route(0, [s], 0)
        self.mu.value, self.meet.value, self.stop.value = INF, -1, 0
        self.top_f.value = self.top_b.value = 0.0
        for c in self.conns:
            c.send((s, t))
        (pf, n_f), (pb, n_b) = (c.recv() for c in self.conns)
        if self.meet.value == -1:
            return Route(INF, [], n_f + n_b)
        return Route(self.mu.value, _build_path(self.meet.value, pf, pb), n_f + n_b)

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Sequential Dijkstra - the baseline every other method is compared against."""
import heapq

from .graph import INF, Route


def dijkstra(g, s, t):
    """Classic Dijkstra with a binary heap: O(E log V). Stops as soon as t is settled."""
    dist = [INF] * g.n
    prev = [-1] * g.n
    dist[s] = 0
    pq = [(0, s)]
    settled = 0
    adj = g.adj
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:          # stale heap entry
            continue
        settled += 1
        if u == t:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[t] == INF:
        return Route(INF, [], settled)
    path, u = [], t
    while u != -1:
        path.append(u)
        u = prev[u]
    path.reverse()
    return Route(dist[t], path, settled)

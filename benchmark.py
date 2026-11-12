"""Average timings over many random long-distance queries.

    python benchmark.py --rows 120 --cols 120 --queries 30
    python benchmark.py --rows 316 --cols 316 --queries 10 --skip-ch --skip-partition   # ~100 000 intersections
"""
import argparse
import os
import statistics
import time

from gps_nav import (INF, ContractionHierarchy, GeoPartition, ParallelBidirectional,
                     bidirectional_dijkstra, dijkstra, generate_city, random_queries)


def run(label, fn, queries, base):
    times, settled, ok = [], [], True
    for (s, t), b in zip(queries, base):
        t0 = time.perf_counter()
        r = fn(s, t)
        times.append((time.perf_counter() - t0) * 1000)
        d = r if isinstance(r, (int, float)) else r.distance
        if not isinstance(r, (int, float)):
            settled.append(r.settled)
        ok &= d == b.distance
    st = f"{statistics.mean(settled):>10,.0f}" if settled else f"{'-':>10}"
    print(f"| {label:42} | {statistics.mean(times):>10.2f} | {st} | {'yes' if ok else 'NO!'} |")
    return statistics.mean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ch", action="store_true")
    ap.add_argument("--skip-partition", action="store_true")
    a = ap.parse_args()

    g = generate_city(a.rows, a.cols, a.seed)
    queries = random_queries(g, a.queries, seed=a.seed)
    print(f"{g.n:,} intersections, {g.m:,} road segments, {a.queries} long random queries, "
          f"{os.cpu_count()} CPU cores\n")
    base = [dijkstra(g, s, t) for s, t in queries]
    queries, base = zip(*[(q, b) for q, b in zip(queries, base) if b.distance < INF])

    print("| Method                                     | avg ms/query | avg settled | correct |")
    print("|--------------------------------------------|-------------:|------------:|:-------:|")
    t_dij = run("Dijkstra (sequential)", lambda s, t: dijkstra(g, s, t), queries, base)
    t_bi = run("Bidirectional Dijkstra (sequential)", lambda s, t: bidirectional_dijkstra(g, s, t), queries, base)
    with ParallelBidirectional(g) as pb:
        pb.query(*queries[0])
        t_par = run("Bidirectional Dijkstra (2 processes)", pb.query, queries, base)
    if not a.skip_ch:
        ch = ContractionHierarchy(g)
        t_ch = run("Contraction hierarchy query", ch.query, queries, base)
    if not a.skip_partition:
        gp = GeoPartition(g, 2, 2)
        run("Geographic partition query (2x2)", gp.distance, queries, base)
    print(f"\nSpeed-up vs Dijkstra: bidirectional {t_dij / t_bi:.2f}x, "
          f"2-process bidirectional {t_dij / t_par:.2f}x"
          + (f", CH query {t_dij / t_ch:.0f}x" if not a.skip_ch else ""))
    if not a.skip_ch:
        print(f"CH preprocessing: {ch.build_seconds:.1f}s, {ch.shortcuts_added:,} shortcuts")
    if not a.skip_partition:
        gp1 = GeoPartition(g, 2, 2, processes=1)
        print(f"Partition preprocessing (2x2 regions): 1 process {gp1.build_seconds:.2f}s, "
              f"{os.cpu_count()} processes {gp.build_seconds:.2f}s "
              f"-> {gp1.build_seconds / gp.build_seconds:.2f}x")


if __name__ == "__main__":
    main()

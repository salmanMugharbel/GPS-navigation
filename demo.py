"""Find one route with every method and print a side-by-side comparison.

    python demo.py                      # 60 x 60 city (3 600 intersections)
    python demo.py --rows 100 --cols 100
"""
import argparse
import time

from gps_nav import (INF, CandidateEvaluator, ContractionHierarchy, GeoPartition,
                     ParallelBidirectional, bidirectional_dijkstra, dijkstra,
                     generate_city, path_length)


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    g = generate_city(a.rows, a.cols, a.seed)
    s, t = 0, g.n - 1                                   # opposite corners of the city
    print(f"City: {g.n:,} intersections, {g.m:,} road segments. Route {s} -> {t}\n")

    rows = []
    base, ms = timed(dijkstra, g, s, t)
    rows.append(("1. Dijkstra (sequential baseline)", base.distance, base.settled, ms))

    r, ms = timed(bidirectional_dijkstra, g, s, t)
    rows.append(("2. Bidirectional Dijkstra (sequential)", r.distance, r.settled, ms))

    with ParallelBidirectional(g) as pb:
        pb.query(s, t)                                  # warm-up
        r, ms = timed(pb.query, s, t)
    rows.append(("3. Bidirectional Dijkstra (2 processes)", r.distance, r.settled, ms))

    ch = ContractionHierarchy(g)
    print(f"Contraction hierarchy built in {ch.build_seconds:.2f}s, {ch.shortcuts_added:,} shortcuts added")
    r, ms = timed(ch.query, s, t)
    rows.append(("4. Contraction hierarchy query", r.distance, r.settled, ms))

    gp = GeoPartition(g, 2, 2)
    print(f"Geographic partition (2x2 regions, parallel) built in {gp.build_seconds:.2f}s, "
          f"{gp.boundary_count:,} boundary intersections")
    d, ms = timed(gp.distance, s, t)
    rows.append(("5. Geographic partition query", d, None, ms))

    print(f"\n{'Method':42} {'Travel time':>12} {'Settled':>9} {'Time':>10}   Correct")
    for name, dist, settled, ms in rows:
        ok = "yes" if dist == base.distance else "NO!"
        st = f"{settled:,}" if settled is not None else "-"
        print(f"{name:42} {dist / 10:>10.1f} s {st:>9} {ms:>8.2f} ms   {ok}")

    print("\nParallel candidate evaluation (16 'via' intersections, 2 workers):")
    with CandidateEvaluator(g) as ce:
        ce.best_route(s, t, k=2)                        # warm-up
        (par, _), tp = timed(ce.best_route, s, t, 16)
        (seq, _), ts = timed(ce.best_route_sequential, s, t, 16)
    gap = (par[0] - base.distance) / base.distance * 100
    print(f"  best candidate: {par[0] / 10:.1f} s via intersection {par[1]} "
          f"({gap:.1f}% above the optimum {base.distance / 10:.1f} s)")
    print(f"  1 core: {ts:.0f} ms   |   2 workers: {tp:.0f} ms   |   speed-up {ts / tp:.2f}x")
    assert path_length(g, par[2]) == par[0]
    print("\nRoute (first 12 intersections):", base.path[:12], "...")


if __name__ == "__main__":
    main()

"""Every method must return exactly the same travel time as plain Dijkstra."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gps_nav import (INF, CandidateEvaluator, ContractionHierarchy, GeoPartition,  # noqa: E402
                     ParallelBidirectional, bidirectional_dijkstra, dijkstra,
                     generate_city, path_length, random_queries)


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = generate_city(30, 30, seed=11)
        rng = random.Random(1)
        cls.queries = random_queries(cls.g, 25, seed=2) + [
            (rng.randrange(cls.g.n), rng.randrange(cls.g.n)) for _ in range(25)]
        cls.base = [dijkstra(cls.g, s, t) for s, t in cls.queries]

    def check_paths(self, route, s, t):
        if route.distance < INF:
            self.assertEqual(route.path[0], s)
            self.assertEqual(route.path[-1], t)
            self.assertEqual(path_length(self.g, route.path), route.distance)

    def test_sequential_bidirectional(self):
        for (s, t), b in zip(self.queries, self.base):
            r = bidirectional_dijkstra(self.g, s, t)
            self.assertEqual(r.distance, b.distance)
            self.check_paths(r, s, t)

    def test_parallel_bidirectional(self):
        with ParallelBidirectional(self.g) as pb:
            for (s, t), b in zip(self.queries, self.base):
                r = pb.query(s, t)
                self.assertEqual(r.distance, b.distance)
                self.check_paths(r, s, t)

    def test_contraction_hierarchy(self):
        ch = ContractionHierarchy(self.g)
        for (s, t), b in zip(self.queries, self.base):
            r = ch.query(s, t)
            self.assertEqual(r.distance, b.distance)
            self.check_paths(r, s, t)

    def test_geographic_partition(self):
        for px, py, procs in ((2, 2, 2), (3, 1, 1)):
            gp = GeoPartition(self.g, px, py, processes=procs)
            for (s, t), b in zip(self.queries, self.base):
                self.assertEqual(gp.distance(s, t), b.distance)

    def test_candidates_are_valid_and_never_better_than_optimal(self):
        with CandidateEvaluator(self.g, processes=2) as ce:
            for (s, t), b in list(zip(self.queries, self.base))[:5]:
                best, _ = ce.best_route(s, t, k=8)
                self.assertGreaterEqual(best[0], b.distance)
                self.assertEqual(path_length(self.g, best[2]), best[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

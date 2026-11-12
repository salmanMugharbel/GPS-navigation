# GPS Navigation - Sequential vs Parallel Shortest Paths

Parallel Programming, Task 4 - *"Find the shortest route from your location to your destination."*

The road network is a graph: **vertices** = road intersections, **edges** = road segments,
**weights** = travel time (distance, speed limit and traffic). Pure Python 3 - no third-party packages.

## What is implemented

| Assignment step | Module | What it does |
|---|---|---|
| Sequential approach | `gps_nav/dijkstra.py` | Dijkstra with a binary heap, O(E log V) |
| 1. Preprocess shortcuts (contraction hierarchies) | `gps_nav/contraction.py` | Contracts vertices in importance order, adds shortcut edges, answers queries with two tiny upward searches |
| 2. Divide the map geographically among processors | `gps_nav/partition.py` | Splits the map into px x py regions; a worker pool precomputes each region's boundary-to-boundary travel times in parallel; queries combine them on a small overlay graph |
| 3. Bidirectional Dijkstra, forward + backward at the same time | `gps_nav/bidirectional.py` | `bidirectional_dijkstra` (sequential) and `ParallelBidirectional` (forward search in one process, backward search in another, sharing distances and the best path length through shared memory) |
| 4. Evaluate several path candidates on multiple cores | `gps_nav/candidates.py` | Samples "via" intersections around the straight line source-destination; each candidate route (source -> via -> destination) is evaluated by a pool worker; the fastest wins |

Supporting code: `gps_nav/graph.py` (graph class + synthetic city generator with local streets, fast
highways, traffic and one-way streets), `demo.py`, `benchmark.py`, `tests/`.

## How to run

```bash
python demo.py                                   # one route, all methods side by side
python benchmark.py --rows 120 --cols 120 --queries 30
python benchmark.py --rows 316 --cols 316 --queries 10 --skip-ch --skip-partition   # ~100 000 intersections
python -m unittest discover -s tests -v          # correctness tests
```
Works on Windows, macOS and Linux (Python 3.8+).

## Correctness

`tests/test_correctness.py` compares every method with plain Dijkstra on 50 random queries
(travel times are integers, so the comparison is exact) and checks that every returned path is a real
chain of roads whose total time equals the reported time. All tests pass.

## Results (measured in the development sandbox: 2 CPU cores - re-run on your own machine and use your numbers)

**14,400 intersections / 55,043 road segments, 30 long random queries**

| Method | avg ms/query | avg vertices settled | correct |
|---|---:|---:|:--:|
| Dijkstra (sequential) | 13.71 | 11,740 | yes |
| Bidirectional Dijkstra (sequential) | 17.89 | 8,186 | yes |
| Bidirectional Dijkstra (2 processes) | 16.25 | 7,995 | yes |
| **Contraction hierarchy query** | **0.38** | **242** | yes |
| Geographic partition query (2x2) | 17.93 | - | yes |

* Contraction hierarchy: **36x faster per query** than Dijkstra, after 13.1 s preprocessing (69,218 shortcuts).
* Geographic partition preprocessing (2x2 regions): 1.71 s on 1 process vs 1.05 s on 2 processes (**1.64x**).

**99,856 intersections / 383,648 road segments (the size mentioned in the assignment), 10 long queries**

| Method | avg ms/query | avg vertices settled |
|---|---:|---:|
| Dijkstra (sequential) | 147.51 | 85,077 |
| Bidirectional Dijkstra (sequential) | 146.34 | 64,409 |
| Bidirectional Dijkstra (2 processes) | 142.86 | 63,248 |

**Parallel candidate evaluation** (`demo.py`, 16 candidates, 2 workers, 3,600 intersections):
1 core 96 ms vs 2 workers 58 ms = **1.67x**; best candidate was 0.1 % above the true optimum.

## What the numbers mean (discussion for the report)

* **Bidirectional search shrinks the search space, but by less than half here.** It settled about 25-30 %
  fewer vertices than Dijkstra. The ideal "half the area" argument assumes uniform roads; our map has fast
  highways, so the forward and backward searches both flow along the same highways and overlap.
* **In pure Python, fewer vertices did not mean less time.** Python dictionaries and process
  communication cost more per vertex than the list-based Dijkstra, and the two searches must synchronise
  on every improved label (a lock). With true shared-memory threads (C++, Java, Rust) the parallel
  bidirectional search would give a real speed-up; Python's GIL is the reason we use processes.
* **Contraction hierarchies are by far the biggest win** (36x per query): the expensive work is moved
  into preprocessing, which is done once per map, not once per route. This is why real navigation
  systems use it. Preprocessing is inherently sequential (each contraction depends on the previous ones).
* **Dividing the map geographically parallelises the preprocessing well** (1.64x on 2 cores, regions are
  independent) but its queries are not faster than Dijkstra in this simple version: it is a trade-off
  between preprocessing in parallel and query time.
* **Candidate evaluation scales nicely** (1.67x on 2 cores) because the candidates are fully
  independent - no communication needed. The price is that the answer is "best of the candidates", not
  guaranteed optimal.
* **Amdahl's law**: the fraction that cannot be parallelised (priority-queue order in a single search,
  combining results, process start-up) limits the speed-up; extra cores help only the independent parts.

## Project layout

```
gps_nav/
  graph.py          Graph, Route, city generator, query generator
  dijkstra.py       sequential baseline
  bidirectional.py  sequential + 2-process parallel bidirectional Dijkstra
  contraction.py    contraction hierarchies
  partition.py      geographic partition + parallel precomputation
  candidates.py     parallel evaluation of candidate routes
demo.py             single-route comparison
benchmark.py        timing over many queries
tests/              correctness tests
```

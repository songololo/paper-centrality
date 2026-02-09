# Package Algorithm Audit: Closeness Centrality Implementations

Researched 2026-02-09. Checked actual source code and documentation for each package.

## Summary Table

| Package                          | Closeness function       | Formula                                                          | Cutoff/radius support                                            | Notes                                                                                                                                                 |
| -------------------------------- | ------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DepthmapX**                    | Angular Integration      | `N^2 / sum(d)` (Improved Closeness)                              | Yes (radius parameter)                                           | Matches Turner/Hillier description                                                                                                                    |
| **Place Syntax Tool**            | Integration              | `N^2 / sum(d)` (Improved Closeness)                              | Yes                                                              | Wraps Depthmap-like logic                                                                                                                             |
| **cityseer**                     | `node_centrality`        | `N^2 / sum(d)` (Improved Closeness)                              | Yes (distances parameter)                                        | Also provides Gravity and Harmonic                                                                                                                    |
| **sDNA**                         | Mean Distance (Farness)  | `sum(d) / N` (Normalised Farness)                                | Yes (radius)                                                     | Also provides NQPD: `sum(W * f(d))` (gravity-like). Does **NOT** implement Improved or Harmonic Closeness.                                            |
| **UNA (Urban Network Analysis)** | Closeness                | `1 / sum(d)` (non-normalised Closeness)                          | Yes (search radius)                                              | Also provides Gravity Index: `sum(O * f(d))`. Does **NOT** implement Improved or Harmonic Closeness. Uses the sign-reversal formula.                  |
| **igraph**                       | `closeness()`            | `(n-1) / sum(d)` (Normalised Closeness)                          | Yes (`cutoff` parameter)                                         | Also provides `harmonic_centrality()` = `sum(1/d)`. Default closeness is the formulation the paper critiques.                                         |
| **tidygraph / sfnetworks**       | `centrality_closeness()` | Wraps igraph directly                                            | Yes (via igraph `cutoff`)                                        | Same behaviour as igraph                                                                                                                              |
| **NetworkX**                     | `closeness_centrality()` | `((n-1)/(N-1)) * (n-1)/sum(d)` with `wf_improved=True` (default) | **No** cutoff parameter; requires explicit subgraph construction | Also provides `harmonic_centrality()` = `sum(1/d)`. The `wf_improved` Wasserman-Faust scaling reduces to Normalised Closeness on connected subgraphs. |
| **momepy**                       | Wraps NetworkX           | Same as NetworkX                                                 | Adds ego-graph radius support                                    | Downstream of NetworkX                                                                                                                                |
| **OSMnx**                        | No centrality functions  | N/A                                                              | N/A                                                              | Users call NetworkX directly                                                                                                                          |

## Detailed Findings

### DepthmapX

- Implements angular segment integration as described by Turner (2008) and Hillier et al. (2012).
- Formula: node count divided by mean depth, reciprocal taken = `N^2 / sum(d)`.
- This is the simplified Improved Closeness (Wasserman-Faust without the global normalisation).

### Place Syntax Tool

- Implements integration following Depthmap conventions.
- Same `N^2 / sum(d)` formulation.

### cityseer

- Python package by Gareth Simons.
- Provides Improved Closeness (`N^2 / sum(d)`), Gravity (`sum(exp(-beta*d))`), and Harmonic Closeness (`sum(1/d)`).
- Built-in distance threshold support.

### sDNA

- Outputs **Mean Distance** (Normalised Farness): `sum(d) / N`.
- Also outputs **NQPD** (Network Quantity Penalised by Distance): a gravity-like measure `sum(W * f(d))` where `f` is a decay function.
- Does **NOT** output Improved Closeness or Harmonic Closeness.
- The Mean Distance output is farness-like (higher = less central), not closeness-like.

### Urban Network Analysis (UNA) toolbox

- Implements **non-normalised Closeness**: `1 / sum(d)`.
- This is the formula that exhibits **sign reversal** under localised analysis (the core problem the paper identifies).
- Also implements a **Gravity Index**: `sum(O_j * f(d_ij))` where `O_j` are opportunities and `f` is a distance decay function.
- Does **NOT** implement Improved Closeness or Harmonic Closeness.

### igraph

- `closeness()` computes `(n-1) / sum(d)` where `n` is the number of reachable nodes = **Normalised Closeness**.
- Accepts a `cutoff` parameter for distance-limited computation.
- Also provides `harmonic_centrality()` = `sum(1/d)`, which is appropriate for localised analysis.
- The default `closeness()` is the formulation the paper identifies as problematic for localised analysis.

### tidygraph / sfnetworks

- R packages that wrap igraph.
- `centrality_closeness()` calls igraph's `closeness()` directly.
- Same `(n-1)/sum(d)` = Normalised Closeness.
- Same `cutoff` parameter available.

### NetworkX

- `closeness_centrality()` with `wf_improved=True` (the default):
  - Formula: `((n-1)/(N-1)) * (n-1) / sum(d)`
  - The `(n-1)/(N-1)` factor is a Wasserman-Faust scaling for disconnected graphs.
  - On a connected subgraph where `n = N`, this reduces to `(N-1) / sum(d)` = **Normalised Closeness**.
- **No** built-in cutoff/radius parameter. Localised analysis requires constructing ego graphs manually.
- Also provides `harmonic_centrality()` = `sum(1/d)`.
- Underpins higher-level packages: momepy, OSMnx.

### momepy

- Urban morphology Python package.
- Wraps NetworkX for centrality computation.
- Adds convenience for ego-graph construction (radius support).
- Same underlying formulas as NetworkX.

### OSMnx

- Network download and analysis package.
- Does not provide its own centrality functions.
- Users compute centrality by calling NetworkX functions on OSMnx-generated graphs.

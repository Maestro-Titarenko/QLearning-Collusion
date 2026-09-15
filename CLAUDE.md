# Q-learning Algorithmic Collusion: Replication

Calvano, Calzolari, Denicolò & Pastorello (2020), "Artificial
Intelligence, Algorithmic Pricing, and Collusion", *American Economic
Review* 110(10): 3267–3297. The original PDF is at `/root/.claude/uploads/.../calvanoetal2020...pdf`.

## 1. Economic environment (paper Section II.A)

n differentiated products + 1 outside good. Logit demand for product i at time t:

    q_i,t = exp((a_i - p_i,t) / mu) / [ sum_j exp((a_j - p_j,t) / mu) + exp(a_0 / mu) ]     (5)

- a_i: vertical quality index of product i
- a_0: (inverse) aggregate demand index for the outside good
- mu: degree of horizontal differentiation; mu -> 0 degenerates to perfect
  substitutes (Bertrand paradox)

Per-period profit: pi_i,t = (p_i,t - c_i) * q_i,t, where c_i is marginal cost
(fixed costs are normalized to zero as long as the firm stays in the market).

## 2. Discretizing the action space (Section II.B)

For each parameter set, first numerically solve the static (one-shot) game for
the Bertrand-Nash price vector p^N and the joint-profit-maximizing (monopoly)
price vector p^M (both scalars in the symmetric case). The feasible price set A
is m equally spaced points on [p^N - xi*(p^M - p^N), p^M + xi*(p^M - p^N)].

Baseline parameters: xi = 0.1, m = 15.

**Key implementation optimization**: profit pi_i(a) depends only on the discrete
action combination a = (a_1,...,a_n) (the combination of price indices), not on
the state. So the profit matrix for all m^n price combinations can be
precomputed once before a session starts (225 = 15x15 cells for the symmetric
duopoly). At runtime each period is just a table lookup — no need to recompute
exp(). This is the core trick that makes the whole simulation fast.

## 3. State and memory (Section II.C)

s_t = {p_{t-1}, ..., p_{t-k}}, where k is the memory length (baseline k=1, i.e.
only last period's prices for all players are remembered). State space size
|S| = m^(n*k), action space size |A| = m.

Symmetric duopoly baseline: |S| = 15^2 = 225, |A| = 15. Each agent's Q matrix is
225 x 15.

## 4. Q-learning (Section I.A general definition, Section II applied to the repeated game)

Q-function definition (eq. 3):

    Q(s,a) = E[pi | s,a] + delta * E[ max_{a'} Q(s',a') | s,a ]

Update rule (eq. 4), updating only the visited (s_t, a_t) cell each period:

    Q_{t+1}(s,a) = (1 - alpha) * Q_t(s,a) + alpha * [ pi_t + delta * max_a Q_t(s_{t+1}, a) ]     if s=s_t, a=a_t
    Q_{t+1}(s,a) = Q_t(s,a)                                                                       otherwise

- alpha in [0,1]: learning rate, baseline grid [0.025, 0.25]
- delta: discount factor, baseline 0.95

## 5. Exploration strategy (Section II.D)

Epsilon-greedy, with the exploration rate decaying exponentially over time
(eq. 7):

    epsilon_t = exp(-beta * t)

Larger beta means faster decay. Baseline "representative experiment" point:
alpha = 0.15, beta = 4e-6 (this is the parameter combination the paper uses
repeatedly for its figures starting in Section IV, not the midpoint of the
grid).

## 6. Q-matrix initialization (eq. 8)

At t=0, Q_{i,0}(s, a_i) is set to the discounted expected profit from choosing
a_i "assuming rivals price uniformly at random":

    Q_{i,0}(s, a_i) = [ sum_{a_{-i} in A^{n-1}} pi_i(a_i, a_{-i}) ] / [ (1-delta) * |A|^{n-1} ]

The initial state s_0 is drawn at random at the start of each session.

## 7. Convergence criterion (Section III.B)

For each player and each state, convergence is declared once the optimal action
a_i,t(s) = argmax_a Q_{i,t}(a,s) has stayed unchanged for 100,000 consecutive
periods. A session is stopped if it still hasn't converged after 1,000,000,000
periods (the paper's cap; for replication a lower cap can be used, recording
the non-convergence rate).

Tie-break rule: pick the lowest price on ties (under the symmetric game the
action set is pure strategies, and play can oscillate near an infeasible
target price).

## 8. Baseline parameterization (Section II.E, all taken from the paper)

    n = 2 (symmetric duopoly)
    c_i = 1
    a_i - c_i = 1   =>  a_i = 2
    a_0 = 0
    mu = 1/4
    delta = 0.95
    m = 15
    xi = 0.1
    k = 1 (one-period memory)

Acceptance benchmarks (numbers from the paper itself; use these as sanity
checks in unit tests):
- Price-cost margin at the static Bertrand-Nash equilibrium ≈ 47%
- Margin under full collusion (monopoly) ≈ twice the above, about 94%

"Representative experiment" (the parameter set used for the large batch of
figures starting in Section IV):
    alpha = 0.15, beta = 4e-6   (everything else as in the baseline)

## 9. Key output metric

Profit gain (eq. 9):

    Delta = (pi_bar - pi^N) / (pi^M - pi^N)

pi_bar is the average per-firm profit after convergence, pi^N is the static
Bertrand-Nash profit, and pi^M is the monopoly (joint-profit-maximizing)
profit. Delta=0 corresponds to perfect competition, Delta=1 to perfect
collusion.

Key numbers reported in the paper (used to check whether replicated results
are in a reasonable range):
- Over the full (alpha, beta) grid (100x100), Delta falls between 70%-90%
  (Figure 1)
- Table 1 "All" column: mean Delta = 0.849, Nash equilibrium frequency =
  0.505, average on-path Q-loss = 0.002
- Delta ≈ 0.75 for n=3 (after adjusting beta), Delta ≈ 0.56 for n=4 (on the
  baseline beta grid)
- Asymmetric costs (Table 4, c_1=1 fixed, c_2 lowered from 1 to 0.25): Delta
  declines slowly from 0.849 to 0.713

## 10. Project structure and development conventions

    src/environment.py   logit demand, profit matrix, numerical Nash/monopoly price solvers
    src/qlearning.py     Q-matrix initialization, update rule, epsilon-greedy, single-session driver
    src/simulate.py      multi-session driver, convergence detection, summary statistics
    configs/*.yaml        parameter sets (corresponding to different experiments in the paper)
    experiments/*.py      run specific experiments, output to results/
    tests/                unittest tests (this environment has no PyPI access, can't install
                           pytest, so everything uses the standard-library unittest)
    analysis/              matplotlib scripts reproducing the paper's figures

Development order: first verify that environment.py's numerical solutions
match the benchmark numbers in Section 9, then verify that qlearning.py
converges correctly on a small MDP with a known closed-form solution, and only
then run the duopoly game. Before running a large-scale grid experiment,
first validate that the summary statistics fall in a reasonable range at the
representative parameter point with a small number of sessions (tens to a
few hundred), and only then decide whether to scale up.

**Environment constraint**: the sandbox this Claude session runs in has no
PyPI access (network policy blocks requests to
pypi.org/files.pythonhosted.org, returning 403), so numba can't be installed.
The hot loop is written in plain NumPy first; if performance isn't enough,
consider: (a) lockstep-vectorizing multiple sessions along an array dimension
(since every session has identical structure, sessions can be advanced as a
batch dimension with NumPy array operations instead of a Python loop over
1000 independent sessions), or (b) installing numba/joblib for real JIT +
multiprocess parallelism in an environment that does have network access.

## 11. Phase 2 progress notes (2026-09-11)

Empirically, the pure-NumPy (no-JIT) implementation reaches about 100-110k
periods/second in this sandbox (using the "table lookup only + check only
whether the greedy action of the just-updated state changed" optimization in
`src/simulate.py`). The sandbox only has 2 CPU cores, so sessions are run in
parallel with `joblib.Parallel(n_jobs=2)`; a single session takes about 9-10
seconds on average (needing roughly 1.7-2 million periods to converge, slower
than the "grid midpoint alpha=0.125, beta=1e-5 converges in about 850k periods
on average" mentioned in the paper's main text, because the representative
point's beta=4e-6 is smaller than the grid midpoint, so exploration decays
more slowly).

`experiments/run_representative.py` appends results to
`results/representative_experiment.jsonl` session by session (resumable: seeds
already run are automatically skipped), and can be called in batches with
different `--start --n_sessions` to run more sessions.

Results from 140 sessions (compared against the paper's Table 1 "All"
column):

| Metric | Replicated | Paper |
|---|---|---|
| Mean Δ | 0.861 | 0.849 |
| Δ std. dev. | 0.102 | 0.112 |
| Convergence rate | 100% | close to 100% |
| Share with cycle length = 1 (constant price) | 66.4% | 64.3% |
| Share with cycle length = 2 | 21.4% | 23.8% |
| Share with cycle length ≥ 3 | 12.1% | 11.9% |

Note that the paper's Table 1 "All" column aggregates the entire 100×100
grid, whereas we only ran a single parameter point (the representative
experiment); the two numbers matching this closely is partly luck and
doesn't mean every point on the grid would match this well — reproducing
the Figure 1/2 style grid heatmap will require actually sweeping the grid to
verify this properly.

`src/simulate.py`'s `run_session()` supports a `record_every` argument that
records a windowed average-profit trace during training, used to plot a
learning curve similar to the paper's Figure 10 (see
`analysis/plot_representative.py`).

## 12. Phase 3 progress notes (2026-09-11): anatomy of collusion / deviation-punishment analysis

`src/anatomy.py` implements the core operation from Section IV of the paper:
given a converged limit policy (fixed, no longer exploring), exogenously force
one player to play the static best response at some state s0 on the limit
cycle at τ=1 (the best response to the rival's current price, not an
arbitrary price cut), and let both players resume their normal policies from
τ=2 onward. Because the system becomes a deterministic finite automaton once
the policy is fixed, the price path is guaranteed to eventually enter some
cycle (pigeonhole principle), so discounted profits are computed exactly
(transient segment + analytic sum over the cyclic segment), not approximated
by truncation.

`experiments/run_anatomy.py` trains 30 sessions (seeds 0-29, same alpha/beta
as Phase 2, but this time keeps `SessionResult.policy` for reuse), and for
each session runs a deviation analysis at every state on the converged limit
cycle and for both deviator identities. Within a session, results are first
averaged over "cycle starting point x deviator identity" (matching the paper's
footnote: "counted as one observation for this session"), then averaged
across sessions.

Results (compared against the paper's Table 3 / main text):

| Metric | Replicated | Paper |
|---|---|---|
| Mean discounted-profit change from deviating | -2.82% | about -3% to -4% |
| Share of deviations that are unprofitable | 98.9% | >95% |

The impulse-response plot from `analysis/plot_anatomy.py`
(`results/anatomy_impulse_response.png`) closely matches the shape of the
paper's Figure 4: price drops sharply at τ=1, bottoms out at τ=2 (slightly
below the τ=1 level — the "overshooting" the paper describes), then gradually
recovers, returning to near the long-run price by about τ=9-10 — exactly the
"finite price war + gradual return" stick-and-carrot pattern the paper
emphasizes, rather than a permanent grim-trigger punishment.

## 13. Phase 4 progress notes (2026-09-11): robustness subset — asymmetric costs + n=3/4

**Asymmetric costs (matching the paper's Table 4)**: `experiments/run_asymmetric.py`,
c1=1 fixed, c2 in {1.0, 0.875, 0.75, 0.625, 0.5, 0.25}, **crucially with
a1=a2=2 held constant so only costs are asymmetric** (the first version of the
code incorrectly kept a_2 = c2+1 to preserve a-c=1, which made the whole game
mathematically equivalent to a level shift of the baseline's prices, leaving
firm 2's market share stuck at exactly 0.5 — this itself turned out to be a
good self-check: if an asymmetry variable has no effect on any outcome, the
model is wrong somewhere). Each c2 point was run with 15 sessions:

| c2 | Firm 2's Nash market share (replicated/paper) | Δ (replicated) | Δ (paper) |
|---|---|---|---|
| 1.000 | 0.500 / 0.500 | 0.866 | 0.849 |
| 0.875 | 0.545 / 0.545 | 0.781 | 0.841 |
| 0.750 | 0.588 / 0.588 | 0.806 | 0.812 |
| 0.625 | 0.627 / 0.627 | 0.752 | 0.781 |
| 0.500 | 0.662 / 0.662 | 0.750 | 0.759 |
| 0.250 | 0.722 / 0.722 | 0.682 | 0.713 |

Market shares are the analytic solution and match the paper's numbers exactly
to three decimal places (this part doesn't depend on Q-learning training at
all — it's purely a property of the Bertrand-Nash equilibrium, and it
verifies that environment.py's asymmetric solver is correct). The downward
trend of Δ with increasing asymmetry matches the paper, and the magnitudes
are close; the small sample size (15 sessions vs. the paper's 1000) adds
noise but no systematic bias. Plot: `results/asymmetric_experiment.png`.

**Number of firms (matching the paper's Section V.A)**: `experiments/run_n_players.py`,
still using the representative alpha=0.15, beta=4e-6 (the paper states that
"the robustness analysis uses this same set of values throughout"). n=3's
state space is 15x n=2's (3375 vs. 225), and periods needed to converge rise
from about 2 million to about 3 million; n=4's state space is another 15x
larger (50625), and periods needed to converge rise to over 10 million, with
single-session training time going from about 10 seconds at n=2 to about
150-175 seconds at n=4. Ran 20 sessions for n=3 and 8 sessions for n=4 (the
small n=4 sample is purely a time-budget constraint, not a convergence
failure — all 8 converged):

| n | Δ (replicated, # sessions) | Δ (paper's main text) |
|---|---|---|
| 2 | 0.861 (140) | 0.849 |
| 3 | 0.629 (20) | 0.64 |
| 4 | 0.567 (8) | 0.56 |

All three points fall almost exactly on the paper's numbers; plot:
`results/n_players_comparison.png`.

**Not yet done**: a larger-scale grid heatmap (see Phase 5 below — only a
20×20 grid has been run so far, versus the paper's original 100×100 scale);
other robustness checks such as demand shocks and entry/exit (paper Section
V.C-D, not yet implemented); `anatomy.py`'s static-best-response lookup is
currently hardcoded for n=2 and hasn't been generalized to n=3/4 (this
generalization would need to happen first if deviation analysis is wanted for
n=3/4 as well).

## 14. Phase 5 progress notes (2026-09-11): migrating to the UNC Longleaf cluster + grid sweep

The local sandbox only has 2 CPUs and can't install numba, so the Figure 1/2
style 100×100 grid heatmap (which needs 100×100×several independent training
runs) is computationally infeasible there. This phase migrated the project to
UNC Research Computing's Longleaf cluster (RHEL 9, SLURM scheduler, `general`
partition nodes with 24+ cores each). Migration approach: GitHub as the single
source of truth (push to
`github.com/Maestro-Titarenko/QLearning-Collusion` first, then `git clone` on
Longleaf) rather than manually copying files. The environment uses the
`python/3.12.4` module plus a venv (`requirements.txt`:
numpy/scipy/joblib/matplotlib); SSH uses a dedicated ed25519 key
(`~/.ssh/id_ed25519_longleaf`) set up with `ssh-copy-id` for passwordless
login, so VS Code Remote-SSH and automation scripts can connect directly.

Added `experiments/run_grid.py` + `slurm/run_grid.slurm`: work is split by
SLURM job array, with **each array task handling one fixed alpha value**; within
a task, joblib uses however many cores `--cpus-per-task` requests to run all
beta × session combinations for that alpha. The resumability granularity is
the (alpha_idx, beta_idx, seed) triple, written to
`results/grid/alpha_{idx:03d}.jsonl`; if a task is killed or times out,
resubmitting the same `--array` automatically skips combinations that are
already done. `analysis/plot_grid.py` aggregates these files into a Figure
1-style heatmap (single-hue sequential color scale, light-to-dark blue for Δ
from 0 to 1).

**The beta grid range is an assumption that still needs verifying**:
CLAUDE.md didn't record the paper's exact beta grid endpoints, so the
`[1e-6, 2e-5]` range used here (log-spaced) was back-derived from two known
anchor points — "grid midpoint alpha=0.125, beta=1e-5" (Section 11) and the
representative-experiment point beta=4e-6 — as a reasonable range, not a
number taken directly from the paper. This should be checked against the
paper later.

The first real run used a 20×20 grid (20 alpha values, 20 beta values), 20
sessions per cell, 8000 sessions total, with `max_periods` set to 2 million
(much lower than the paper's 1 billion, trading accuracy for a manageable
wall-clock budget). All 20 array tasks `COMPLETED`, and each task actually
took only 6-11 minutes (well under the requested 6-hour cap):

| Metric | Result |
|---|---|
| Cells completed | 269 / 400 |
| Mean Δ over completed cells | 0.822 (paper's Table 1 "All" column: 0.849) |
| Share falling in the paper's reported Figure 1 range of [0.70, 0.90] | 96.3% |

The 131 empty cells are concentrated in the low-beta column (slow exploration
decay) — none of the sessions at these points converged within 2 million
periods, a direct consequence of `max_periods` being set too tight, not an
algorithm problem. Plot: `results/grid_heatmap.png`; its shape (higher Δ where
beta is larger and alpha is moderate) is qualitatively consistent with the
paper's Figure 1.

**To scale this up further**: increase `N_ALPHA`/`N_BETA`/`N_SESSIONS` in
`slurm/run_grid.slurm` and update the `sbatch --array` upper bound to match
`N_ALPHA-1`; for the low-beta column to converge fully, `--max_periods` would
need to be increased substantially (likely close to the paper's own scale),
and each task's `--time` cap would need to be relaxed accordingly.

**Follow-up run (2026-09-14): 10×10 grid, max_periods=10M**. Traded grid
resolution for per-session convergence budget: a coarser 10×10 grid (10
alpha values, 10 beta values, same range as before) but with `max_periods`
raised 5x to 10 million, to test whether the empty low-beta cells above were
really just a `max_periods` budget problem rather than something structural.
The old 20×20/2M run's data was archived to `results/grid_20x20_2M/` (and its
heatmap to `results/grid_heatmap_20x20_2M.png`) rather than deleted, since
its (alpha_idx, beta_idx) numbering doesn't line up with the new 10-point
grid — the resumability check keys off index tuples, not actual alpha/beta
values, so reusing the same `results/grid/` directory for a different grid
shape would have silently corrupted which combinations get treated as
"already done." All 10 array tasks `COMPLETED` in 5-12 minutes each (well
under the 8-hour cap):

| Metric | 20×20, max_periods=2M | 10×10, max_periods=10M |
|---|---|---|
| Cells completed | 269 / 400 (67%) | 100 / 100 (100%) |
| Mean Δ over completed cells | 0.822 | 0.807 |
| Share in the paper's [0.70, 0.90] range | 96.3% | 98.0% |

Confirms the hypothesis: every cell converges once given enough periods —
this was purely a `max_periods` budget effect, not a sign that some
(alpha, beta) combinations fail to converge in principle. The mean Δ ticking
down slightly (0.822 -> 0.807) makes sense too: the newly-converged cells are
concentrated at low beta, and low beta (slow exploration decay) is exactly
where the paper's own Figure 1 shows lower Δ, so including them pulls the
average down a bit rather than up. Current heatmap:
`results/grid_heatmap.png` (100% filled, no gaps).

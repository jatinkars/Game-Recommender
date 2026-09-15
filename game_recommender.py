"""
Game Recommender: why offline metrics lie
=========================================

An implicit-feedback recommender over player-game playtime, plus the evaluation
setup that decides whether its reported numbers mean anything.

Two things sink most offline recommender results:

  1. RANDOM TRAIN/TEST SPLIT. Player behaviour is a time series. Splitting
     interactions at random puts a player's future in the training set and asks
     the model to predict their past. Metrics look great and do not survive
     deployment.

  2. NO POPULARITY BASELINE. Recommending the top-selling games to everyone is
     a strong strategy on any skewed catalogue. A model that cannot beat it has
     learned popularity, not preference — and precision@k will not tell you
     which happened.

This compares three models under both split strategies, and reports catalogue
coverage alongside accuracy because a recommender that only ever surfaces the
top 20 titles is useless to a storefront even when its precision is high.

Data is simulated with known latent preferences so model quality can be judged
against a ground truth that real playtime logs never expose.

Run:
    python game_recommender.py
    python game_recommender.py --players 8000 --games 500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

SEED = 20260914
GENRES = ["FPS", "TPS", "RPG", "Strategy", "Sports", "Simulation"]


# ---------------------------------------------------------------------------
# Simulated catalogue and players
# ---------------------------------------------------------------------------


@dataclass
class Catalog:
    n_players: int = 4000
    n_games: int = 300
    n_genres: int = len(GENRES)
    sessions_per_player: Tuple[int, int] = (8, 40)
    popularity_alpha: float = 1.1   # power-law skew; lower = more head-heavy


def build_world(cfg: Catalog, rng: np.random.Generator):
    """
    Players have latent genre affinities. Games have genre loadings and an
    intrinsic popularity drawn from a power law, which is what real storefront
    catalogues look like: a few titles absorb most playtime.
    """
    player_taste = rng.dirichlet(np.ones(cfg.n_genres) * 0.6, cfg.n_players)
    game_genre = rng.dirichlet(np.ones(cfg.n_genres) * 0.4, cfg.n_games)
    popularity = rng.pareto(cfg.popularity_alpha, cfg.n_games) + 1.0
    popularity /= popularity.sum()
    # Games launch throughout the year. A title cannot be played before release,
    # and attention spikes at launch then decays -- the dynamic that makes a
    # random split leak: it lets the model see post-cutoff releases.
    release = rng.uniform(0, 300, cfg.n_games)
    release[rng.random(cfg.n_games) < 0.45] = 0.0   # back catalogue
    return player_taste, game_genre, popularity, release


def generate_interactions(cfg: Catalog, rng: np.random.Generator):
    """
    Each interaction is (player, game, timestamp). Choice probability mixes
    genuine taste match with raw popularity — the same entanglement that makes
    offline evaluation hard on real data.
    """
    taste, genre, pop, release = build_world(cfg, rng)
    affinity = taste @ genre.T                       # (players, games)
    base = 0.65 * (affinity / affinity.sum(1, keepdims=True)) + 0.35 * pop

    rows: List[Tuple[int, int, float]] = []
    lo, hi = cfg.sessions_per_player
    for p in range(cfg.n_players):
        k = int(rng.integers(lo, hi))
        times = np.sort(rng.uniform(0, 365, k))
        chosen: set = set()
        for t in times:
            available = release <= t
            if not available.any():
                continue
            # launch spike decaying with weeks since release
            recency = np.exp(-np.maximum(t - release, 0) / 60.0)
            w = base[p] * available * (0.45 + 0.55 * recency)
            for g in chosen:
                w[g] = 0.0
            tot = w.sum()
            if tot <= 0:
                continue
            g = int(rng.choice(cfg.n_games, p=w / tot))
            chosen.add(g)
            rows.append((p, g, float(t)))

    arr = np.array(rows, dtype=float)
    return arr, taste, genre


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def random_split(inter: np.ndarray, rng, test_frac: float = 0.2):
    """Shuffle every interaction. Leaks the future. Included to show the damage."""
    idx = rng.permutation(len(inter))
    cut = int(len(inter) * (1 - test_frac))
    return inter[idx[:cut]], inter[idx[cut:]]


def temporal_split(inter: np.ndarray, test_frac: float = 0.2):
    """
    Global time cutoff. Train on the past, predict the future — the only split
    that matches how the model would actually be used.
    """
    cutoff = np.quantile(inter[:, 2], 1 - test_frac)
    return inter[inter[:, 2] <= cutoff], inter[inter[:, 2] > cutoff]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def to_matrix(train: np.ndarray, n_players: int, n_games: int) -> np.ndarray:
    M = np.zeros((n_players, n_games), dtype=np.float32)
    M[train[:, 0].astype(int), train[:, 1].astype(int)] = 1.0
    return M


def rec_popularity(M: np.ndarray, k: int) -> np.ndarray:
    """Recommend the globally most-played games to everyone. The baseline."""
    counts = M.sum(0)
    top = np.argsort(-counts)[: k + 50]
    return np.tile(top, (M.shape[0], 1))


def rec_item_cf(M: np.ndarray, k: int) -> np.ndarray:
    """Item-item collaborative filtering with cosine similarity."""
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    N = M / norms
    sim = N.T @ N                      # (games, games)
    np.fill_diagonal(sim, 0.0)
    scores = M @ sim
    scores[M > 0] = -np.inf            # never re-recommend an owned title
    return np.argsort(-scores, axis=1)[:, : k + 50]


def rec_als(M: np.ndarray, k: int, factors: int = 24, iters: int = 12,
            reg: float = 0.08, alpha: float = 12.0, seed: int = SEED) -> np.ndarray:
    """
    Implicit-feedback matrix factorization (alternating least squares).

    Confidence weighting: an observed interaction is evidence of preference,
    but an unobserved one is only weak evidence of dislike. Treating zeros as
    hard negatives is the standard mistake with implicit data.
    """
    rng = np.random.default_rng(seed)
    n_u, n_i = M.shape
    X = rng.normal(0, 0.1, (n_u, factors))
    Y = rng.normal(0, 0.1, (n_i, factors))
    C = 1.0 + alpha * M                      # confidence
    P = (M > 0).astype(np.float32)           # preference
    I = np.eye(factors)

    for _ in range(iters):
        YtY = Y.T @ Y
        for u in range(n_u):
            Cu = C[u]
            A = YtY + (Y.T * (Cu - 1.0)) @ Y + reg * I
            b = (Y.T * Cu) @ P[u]
            X[u] = np.linalg.solve(A, b)
        XtX = X.T @ X
        for i in range(n_i):
            Ci = C[:, i]
            A = XtX + (X.T * (Ci - 1.0)) @ X + reg * I
            b = (X.T * Ci) @ P[:, i]
            Y[i] = np.linalg.solve(A, b)

    scores = X @ Y.T
    scores[M > 0] = -np.inf
    return np.argsort(-scores, axis=1)[:, : k + 50]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def evaluate(recs: np.ndarray, test: np.ndarray, M_train: np.ndarray,
             n_games: int, k: int = 10) -> Dict[str, float]:
    truth: Dict[int, set] = {}
    for p, g, _ in test:
        truth.setdefault(int(p), set()).add(int(g))

    precs, recs_at_k, ndcgs = [], [], []
    surfaced = set()

    for p, relevant in truth.items():
        # drop anything already seen in training, then take top k
        seen = set(np.flatnonzero(M_train[p]))
        ranked = [int(g) for g in recs[p] if int(g) not in seen][:k]
        if not ranked:
            continue
        surfaced.update(ranked)
        hits = [1.0 if g in relevant else 0.0 for g in ranked]

        precs.append(sum(hits) / k)
        recs_at_k.append(sum(hits) / max(1, len(relevant)))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(k, len(relevant))))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        f"precision@{k}": float(np.mean(precs)),
        f"recall@{k}": float(np.mean(recs_at_k)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "coverage": len(surfaced) / n_games,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=4000)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    cfg = Catalog(n_players=args.players, n_games=args.games)
    inter, _, _ = generate_interactions(cfg, rng)
    print(f"\n{len(inter):,} interactions | {cfg.n_players:,} players | "
          f"{cfg.n_games} games")

    bar = "-" * 74
    models = {
        "Popularity (baseline)": rec_popularity,
        "Item-item CF": rec_item_cf,
        "Implicit MF (ALS)": rec_als,
    }

    for split_name, (tr, te) in {
        "RANDOM SPLIT  (leaks the future)": random_split(inter, rng),
        "TEMPORAL SPLIT  (honest)": temporal_split(inter),
    }.items():
        M = to_matrix(tr, cfg.n_players, cfg.n_games)
        print(f"\n{split_name}")
        print(bar)
        print(f"{'model':<24}{'precision@%d' % args.k:>14}{'recall@%d' % args.k:>12}"
              f"{'ndcg@%d' % args.k:>10}{'coverage':>12}")
        for name, fn in models.items():
            r = fn(M, args.k)
            m = evaluate(r, te, M, cfg.n_games, args.k)
            print(f"{name:<24}{m[f'precision@{args.k}']:>14.4f}"
                  f"{m[f'recall@{args.k}']:>12.4f}{m[f'ndcg@{args.k}']:>10.4f}"
                  f"{m['coverage']:>12.1%}")
    print(bar)
    print("""
Read the two tables against each other, not each on its own.

The random split reports better numbers for every model. None of that lift is
real: it comes from training on interactions that happened after the ones being
predicted. Any recommender result quoted without naming its split strategy
should be treated as unverified.

Coverage is the column most write-ups omit. A model can post competitive
precision while surfacing a tiny slice of the catalogue, which is worthless to a
storefront that needs the long tail discovered. Accuracy and coverage trade off,
and picking a point on that trade-off is a product decision, not a modelling one.
""")


if __name__ == "__main__":
    main()

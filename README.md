# Game Recommender: why offline metrics lie
 
An implicit-feedback recommender over player–game interactions, and the
evaluation setup that decides whether its numbers mean anything.
 
## The two failures this is built to expose
 
**Random train/test splits leak the future.** Player behaviour is a time series
and a game catalogue changes — titles launch, attention spikes, interest decays.
Splitting interactions at random lets the model train on events that happened
*after* the ones it is asked to predict, including games that had not launched
at prediction time.
 
**No popularity baseline.** Recommending the best-selling titles to everyone is
a strong strategy on any skewed catalogue. A model that cannot beat it has
learned popularity, not preference, and precision@k will not tell you which.
 
## Results
 
4,000 players, 300 games, 94,770 interactions. Games launch throughout the year
with a decaying launch spike; 45% are back catalogue.
 
**Random split — leaks the future**
 
| model | precision@10 | recall@10 | ndcg@10 | coverage |
|---|---|---|---|---|
| Popularity (baseline) | 0.0706 | 0.1526 | 0.1827 | 6.7% |
| Item-item CF | 0.0719 | 0.1551 | 0.1841 | 51.3% |
| Implicit MF (ALS) | 0.0516 | 0.1096 | 0.0914 | 99.3% |
 
**Temporal split — honest**
 
| model | precision@10 | recall@10 | ndcg@10 | coverage |
|---|---|---|---|---|
| Popularity (baseline) | 0.0348 | 0.0742 | 0.0749 | 6.7% |
| Item-item CF | 0.0360 | 0.0770 | 0.0765 | 46.7% |
| Implicit MF (ALS) | 0.0291 | 0.0625 | 0.0469 | 91.7% |
 
### A random split roughly doubles reported precision
 
Popularity goes 0.0348 → 0.0706, item-item CF 0.0360 → 0.0719. None of that
lift is real. Any recommender result quoted without naming its split strategy
should be treated as unverified.
 
### Collaborative filtering barely beats popularity on accuracy
 
0.0360 against 0.0348 — about 3%. On accuracy alone the model is not worth
shipping over a top-sellers list.
 
### The actual win is coverage
 
Item-item CF surfaces 46.7% of the catalogue against popularity's 6.7% — seven
times the discovery for a 3% accuracy gain. For a storefront that needs the long
tail found, that is the whole argument, and it is invisible if you report
precision alone.
 
### ALS trades accuracy for reach
 
Lowest precision, 91.7% coverage. Whether that is the right point on the
trade-off is a product decision, not a modelling one.
 
## Implementation notes
 
**Confidence weighting in ALS.** An observed interaction is evidence of
preference; an unobserved one is weak evidence of dislike. Treating zeros as
hard negatives is the standard mistake with implicit feedback, so the model
weights confidence as `1 + α·interaction`.
 
**Owned titles are masked** from recommendations before ranking, and anything
seen in training is dropped before the top-k cut. Skipping this inflates every
metric.
 
**Why simulated data.** Latent player preferences are set explicitly, so model
behaviour can be judged against a ground truth that real playtime logs never
expose. The pipeline runs unchanged on a real interaction table of
`(player, game, timestamp)`.
 
## Running it
 
```bash
pip install -r requirements.txt
python game_recommender.py
python game_recommender.py --players 8000 --games 500
```
 
## Limitations
 
- **No cold start.** New players and new games are not handled; both are the
  hard part in production.
- **Binary implicit signal.** Playtime magnitude is discarded; a 2-hour and a
  200-hour title count the same.
- **No session context or sequence.** A sequential model would use order, which
  matters for games.
- **Single global time cutoff.** Per-player leave-last-n is stricter.
- **No online validation.** Offline metrics rank models; only an A/B test
  measures whether recommendations change behaviour.
## Files
 
| File | Purpose |
|---|---|
| `game_recommender.py` | Simulation, three models, both splits, metrics |
| `requirements.txt` | numpy |

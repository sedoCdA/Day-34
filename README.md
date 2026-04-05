### AM + PM Take-Home Assignment | Clustering, PCA & Week 6 Comprehensive Review

Assignment Structure

```
D34_Clustering_PCA_Assignment.ipynb
│
├── AM Part A — K-Means & DBSCAN on Iris
│   ├── load_and_scale_iris()
│   ├── run_kmeans()
│   ├── align_cluster_labels()        [Hungarian algorithm for label matching]
│   ├── cluster_comparison_table()
│   ├── compute_clustering_metrics()  [ARI + NMI]
│   ├── tune_dbscan()                 [grid search over eps, min_samples]
│   ├── run_dbscan()
│   └── plot_cluster_comparison()     [3-panel: true, K-Means, DBSCAN]
│
├── AM Part B — Hierarchical Clustering
│   ├── run_hierarchical()
│   ├── plot_dendrogram()
│   └── comparison_summary DataFrame  [all 3 methods]
│
├── AM Part C — Interview Questions
│   ├── Q1: K-Means greedy / K-Means++ (markdown)
│   ├── Q2: kmeans_from_scratch()     [NumPy, centroid shift convergence]
│   └── Q3: Silhouette 0.25 analysis + plot_elbow_and_silhouette()
│
├── AM Part D — Clustering Analogy (markdown, verified)
│
├── PM Part A — 12-Algorithm Reference
│   ├── WEEK6_REFERENCE dict
│   ├── print_week6_reference()
│   ├── benchmark_on_wine()           [3 algorithms verified on Wine]
│   └── Algorithm Selection Flowchart (markdown)
│
├── PM Part B — Image Compression
│   ├── create_test_image()
│   ├── compress_image_pca()          [per-channel, MSE, PSNR, ratio]
│   └── plot_pca_compression()        [side-by-side grid]
│
├── PM Part C — Interview Questions
│   ├── Q1: Full ML pipeline (markdown)
│   ├── Q2: weekly_model_comparison() [5 models, PCA option, CV DataFrame]
│   └── Q3: PCA accuracy drop analysis (markdown)
│
└── PM Part D — Week 6 Study Guide (markdown)
```

### 1.1 Algorithm

K-Means partitions n points into K clusters by minimizing **within-cluster sum of squared distances (inertia)**:

```
J = Σₖ Σᵢ∈Cₖ ||xᵢ − μₖ||²
```

**Steps:**
1. Initialize K centroids (randomly or via K-Means++)
2. **Assignment:** Each point → nearest centroid (Euclidean distance)
3. **Update:** Recompute each centroid as the mean of its assigned points
4. Repeat steps 2–3 until convergence (centroid shift < tolerance)

### 1.2 K-Means++ Initialization

Standard random initialization can produce poor local minima. K-Means++ spreads initial centroids:

```
P(x is next centroid) ∝ D(x)²
```
where D(x) is the distance from x to the nearest already-selected centroid. This ensures starting centroids cover the data space, yielding faster convergence and better solutions.

### 1.3 Greedy Nature and Local Minima

K-Means is **greedy** — each assignment step picks the locally optimal cluster without reconsidering previous steps. This guarantees convergence but only to a **local minimum**. Different initializations → different results.

Mitigation: Run K-Means multiple times (`n_init=10`) and keep the solution with the lowest inertia.

### 1.4 Choosing Optimal K

**Elbow Method:** Plot inertia vs K. The "elbow" where inertia stops dropping sharply suggests the optimal K.

**Silhouette Method:** Plot silhouette score vs K. Choose K with the highest score.

```python
inertias = [KMeans(n_clusters=k, n_init=10).fit(X).inertia_ for k in range(2, 11)]
sil_scores = [silhouette_score(X, KMeans(n_clusters=k).fit_predict(X)) for k in range(2, 11)]
```

### 1.5 K-Means from Scratch

```python
def kmeans_from_scratch(X, k, max_iter=100, tol=1e-4, random_state=42):
    if k < 1 or k > len(X):
        raise ValueError(f"k must be between 1 and {len(X)}, got {k}")

    rng = np.random.RandomState(random_state)
    centroids = X[rng.choice(len(X), k, replace=False)].astype(float)
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        diff      = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))
        new_labels = distances.argmin(axis=1)

        new_centroids = np.array([
            X[new_labels == c].mean(axis=0) if (new_labels == c).any()
            else centroids[c]
            for c in range(k)
        ])

        if np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max() < tol:
            break

        centroids, labels = new_centroids, new_labels

    inertia = sum(
        ((X[labels == c] - centroids[c]) ** 2).sum()
        for c in range(k) if (labels == c).any()
    )
    return labels, centroids, inertia
```

### 1.6 Limitations

| Limitation | Solution |
|---|---|
| Requires specifying K | Elbow + Silhouette analysis |
| Assumes spherical clusters | Use GMM or DBSCAN |
| Sensitive to outliers | Use median-based k-medoids |
| Sensitive to initialization | K-Means++, multiple runs |
| Requires scaling | Always apply StandardScaler |

---

## 2. DBSCAN

### 2.1 Core Concepts

DBSCAN classifies each point as:

- **Core point:** Has at least `min_samples` points within radius `eps`
- **Border point:** Within `eps` of a core point but not a core point itself
- **Noise point:** Neither core nor border — labeled as `-1`

```
Core point:   ●  (dense neighborhood)
Border point: ○  (reachable from core)
Noise:        ×  (isolated)

[eps radius around core point]
   ● ○ ○
  ○ ● ○ ×
   ○ ● ○
```

### 2.2 Key Parameters

**eps (ε):** Neighborhood radius. Too small → everything is noise. Too large → everything merges into one cluster.

**min_samples:** Minimum points to form a core. Higher = more conservative = more noise points. Rule of thumb: `min_samples ≥ D + 1` where D is the number of dimensions.

### 2.3 Choosing eps — k-Distance Plot

```python
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances  = np.sort(distances[:, -1])[::-1]
plt.plot(k_distances)
# Look for the "elbow" — use that distance as eps
```

### 2.4 DBSCAN vs K-Means

| Criterion | K-Means | DBSCAN |
|---|---|---|
| K required | Yes | No |
| Cluster shape | Spherical only | Any shape |
| Outlier handling | No | Yes (-1 label) |
| Scalability | O(nKt) | O(n log n) with index |
| Sensitivity to params | K | eps, min_samples |
| Cluster size | Tends to equalize | Variable |

---

## 3. Hierarchical Clustering

### 3.1 Algorithm (Agglomerative)

Bottom-up approach:
1. Each point starts as its own cluster (n clusters)
2. Compute pairwise distances between all clusters
3. Merge the two closest clusters
4. Repeat until desired number of clusters reached

### 3.2 Linkage Methods

| Linkage | Distance Formula | Best For |
|---|---|---|
| Ward | Minimizes within-cluster variance | General purpose, produces compact clusters |
| Complete | Max distance between clusters | Tends to produce balanced clusters |
| Average | Mean distance between all pairs | Balanced, robust to outliers |
| Single | Min distance between clusters | Elongated chains; sensitive to outliers |

### 3.3 Dendrogram

The dendrogram shows the full merge history. To select K clusters, draw a horizontal line at the appropriate height — count how many vertical lines it crosses.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30)
```

### 3.4 Comparison: K-Means vs DBSCAN vs Hierarchical

| Aspect | K-Means | DBSCAN | Hierarchical |
|---|---|---|---|
| K required | Yes | No | Yes (cut level) |
| Shape | Spherical | Any | Any |
| Memory | O(nK) | O(n) | O(n²) |
| Scalability | Large n | Medium n | Small n (<10K) |
| Interpretability | Centroids | Core/noise | Dendrogram |

---

## 4. PCA — Principal Component Analysis

### 4.1 Mathematical Foundation

PCA finds orthogonal directions (principal components) that capture maximum variance:

1. Center: `X_c = X - mean(X)`
2. Covariance: `Σ = X_cᵀ X_c / (n-1)`
3. Eigen-decomposition: `Σ = V Λ Vᵀ`
4. Sort eigenvectors by eigenvalue (descending)
5. Project: `X_reduced = X_c · V[:, :k]`

The first principal component (PC1) is the direction of maximum variance. PC2 is the direction of maximum remaining variance orthogonal to PC1.

### 4.2 Explained Variance

```python
pca = PCA().fit(X_scaled)
ev_ratio = pca.explained_variance_ratio_
cumulative = np.cumsum(ev_ratio)

# Retain 95% variance automatically
pca_95 = PCA(n_components=0.95)
X_reduced = pca_95.fit_transform(X_scaled)
n_components = pca_95.n_components_
```

### 4.3 Scree Plot

```python
plt.plot(range(1, len(ev_ratio)+1), ev_ratio, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
# Look for the elbow to choose K
```

### 4.4 When PCA Helps and Hurts

**Helps:**
- Removes multicollinearity
- Reduces noise from irrelevant features
- Speeds up downstream models
- 2D/3D visualization of high-D data

**Hurts:**
- Destroys interpretability (components are linear combinations)
- Discards low-variance features even if they are class-discriminative
- Accuracy may drop if discriminative signal is in minor components

### 4.5 PCA vs LDA

| Aspect | PCA | LDA |
|---|---|---|
| Objective | Maximize variance | Maximize class separation |
| Labels required | No (unsupervised) | Yes (supervised) |
| Components | min(n, p) | n_classes - 1 |
| Use case | Compression, viz | Feature extraction for classification |

### 4.6 Image Compression with PCA

```python
def compress_channel(channel, n_components):
    pca = PCA(n_components=n_components)
    compressed   = pca.fit_transform(channel)
    reconstructed = pca.inverse_transform(compressed)
    return reconstructed, pca.explained_variance_ratio_.sum()

# Compression ratio
original_size    = H * W * C
compressed_size  = n_components * (H + W) * C
ratio = original_size / compressed_size
```

---

## 5. Clustering Evaluation Metrics

### 5.1 External Metrics (require ground truth)

**Adjusted Rand Index (ARI)**

```
ARI = (RI - Expected_RI) / (Max_RI - Expected_RI)
```
Range: [-1, 1]. ARI=1: perfect match. ARI=0: random. ARI<0: worse than random.

**Normalized Mutual Information (NMI)**

```
NMI = I(Y; C) / sqrt(H(Y) * H(C))
```
Range: [0, 1]. Measures how much knowing clusters reduces uncertainty about true labels.

### 5.2 Internal Metrics (no ground truth needed)

**Silhouette Score**

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
- `a(i)`: mean distance within same cluster
- `b(i)`: mean distance to nearest other cluster
- Range: [-1, 1]. Above 0.5 is good; below 0.25 indicates weak structure.

| Score | Interpretation |
|---|---|
| 0.71–1.00 | Strong, well-separated clusters |
| 0.51–0.70 | Reasonable structure |
| 0.26–0.50 | Weak structure |
| ≤ 0.25 | No substantial structure found |

**Inertia (within-cluster SSE)**
Only use for comparing models with the same K — lower is better.

---

## 6. Week 6 — 12-Algorithm Reference

| # | Algorithm | One-Line | Key Params | Use Case |
|---|---|---|---|---|
| 1 | Logistic Regression | Sigmoid classifier | C, solver | Default linear baseline |
| 2 | Decision Tree | Recursive splits | max_depth, criterion | Interpretable rules |
| 3 | Random Forest | Bagged trees | n_estimators, max_features | Tabular baseline |
| 4 | AdaBoost | Sequential upweighting | n_estimators, lr | Weak-learner ensemble |
| 5 | XGBoost | Regularized boosting | max_depth, lr, subsample | Competition accuracy |
| 6 | LightGBM | Leaf-wise boosting | num_leaves, lr | Large-scale tabular |
| 7 | Voting | Majority/probability avg | voting, estimators | Easy ensemble gain |
| 8 | Stacking | Meta-learner on OOF | estimators, final_estimator | Maximum accuracy |
| 9 | SVM (RBF) | Max-margin + kernel | C, gamma | High-dim non-linear |
| 10 | KNN | Nearest-neighbor vote | n_neighbors, metric | Simple baseline |
| 11 | K-Means | Centroid clustering | n_clusters, init | Customer segmentation |
| 12 | DBSCAN | Density clustering | eps, min_samples | Anomaly detection |
| + | PCA | Variance projection | n_components | Dimensionality reduction |

---

## 7. Image Compression with PCA

**Why it works:** Natural images have high spatial correlation — nearby pixels are similar. PCA captures this redundancy in a few components.

**Compression ratio formula:**
```
Original size    = H × W × C  (pixels)
Compressed size  = n_components × (H + W) × C
Ratio            = Original / Compressed
```

| n_components | Typical PSNR | Use Case |
|---|---|---|
| 5 | Low (~20 dB) | Thumbnail / preview |
| 20 | Medium (~30 dB) | Web compression |
| 50 | Good (~35 dB) | Archival |
| 100 | High (~40 dB) | Near-lossless |

**PSNR:** `10 × log₁₀(MAX² / MSE)`. Higher = better quality. Above 30 dB is generally acceptable.

---

## 8. Interview Questions & Answers

### Q1 (AM) — K-Means Greedy + Local Minima + K-Means++

K-Means is greedy because each assignment step commits to the nearest centroid without global optimization. It always converges but only to a local minimum — the result depends on initialization.

K-Means++ fixes this by spreading initial centroids: each new centroid is chosen with probability proportional to its squared distance from existing centroids. This probabilistic seeding virtually eliminates poor initializations.

---

### Q2 (AM) — K-Means from Scratch

See [Section 1.5](#15-k-means-from-scratch) for full implementation.

Key design decisions:
- `np.argpartition` avoided here for clarity; `argmin` on distances matrix suffices for moderate n
- Centroid shift convergence criterion (`< tol`) is more stable than label change detection
- Handles empty cluster edge case by keeping the previous centroid

---

### Q3 (AM) — Silhouette Score 0.25 Investigation

0.25 is at the lower boundary of acceptable. Investigate:

1. **Wrong K:** Plot silhouette for K=2 to 10. The true optimal K may not be 5.
2. **Non-spherical clusters:** K-Means assumes convex, equally-sized clusters. Visualize with PCA 2D. If clusters are elongated or ring-shaped, switch to DBSCAN or GMM.
3. **Too many features:** In high dimensions, Euclidean distances become unreliable. Apply PCA to 10–20 dimensions first.
4. **No real cluster structure:** The data may be inherently unimodal. Check if silhouette peaks at K=2.
5. **Unscaled features:** Re-verify StandardScaler was applied.

---

### Q1 (PM) — Complete ML Pipeline (1000 samples, 200 features)

| Step | Action | Reason |
|---|---|---|
| EDA | Distribution, correlation, missing | Understand data before modeling |
| Scaling | StandardScaler | Required by SVM, KNN, LR, PCA |
| PCA (optional) | n_components=0.95 | Remove multicollinearity, reduce noise |
| Baseline | Logistic Regression | Fast, interpretable, check linear separability |
| Strong model | Random Forest | Non-linear, feature importance, robust |
| Best accuracy | XGBoost | If RF insufficient |
| Evaluation | Stratified 5-fold CV, F1-macro | Handles class imbalance |
| Deployment | sklearn Pipeline + joblib | Prevents data leakage, reproducible |

---

### Q3 (PM) — PCA Accuracy Drop (0.92 → 0.85)

**Reason 1:** PCA maximizes variance, not class discrimination. The discarded 5% variance may contain the most class-discriminative signal.

**Reason 2:** Low-variance features can be highly informative. A binary feature that perfectly separates two classes but has low variance (most values are 0) gets dropped by PCA.

**Reason 3:** Model hyperparameters tuned for 100D space are no longer optimal in 10D space. SVM's gamma, KNN's K, and regularization strength all change with dimensionality.

**Solution:** Use `LinearDiscriminantAnalysis` for supervised reduction, or `SelectKBest(mutual_info_classif, k=10)` to select features by discriminative power, not variance.

---

## 9. Algorithm Selection Flowchart

```
Is the task UNSUPERVISED?
├── Yes
│   ├── K is known, clusters are spherical?  → K-Means
│   ├── K unknown, non-spherical, outliers?  → DBSCAN
│   ├── Need hierarchy / dendrogram?         → Agglomerative (Ward)
│   └── Need to reduce dimensions?           → PCA
│
└── No (SUPERVISED)
    ├── Text / sparse features?              → LinearSVC / Logistic Regression
    ├── n < 500, p >> n?                     → SVM (Linear) / LR (L1)
    ├── n < 500, p < n?                      → SVM (RBF) + GridSearch
    ├── Need interpretability?               → Decision Tree
    ├── Need probabilities?                  → Logistic Regression
    ├── Robust non-linear baseline?          → Random Forest
    ├── Best accuracy (medium n)?            → XGBoost / LightGBM
    ├── Large n > 100K?                      → LightGBM / SGDClassifier
    └── Multiple strong models available?   → Stacking / Voting
```

**Edge Cases:**

| Situation | Correct Approach |
|---|---|
| Imbalanced classes | F1-macro; `class_weight='balanced'` |
| Time-series | `TimeSeriesSplit` — no future leakage |
| Multi-label | `MultiOutputClassifier` or `ClassifierChain` |
| Online/streaming | `SGDClassifier.partial_fit()` |
| Mixed feature types | `ColumnTransformer` or tree-based |
| Very sparse | Keep sparse; `MaxAbsScaler` |

---



# ibm_attrition_kmeans_criteria

**Purpose:** evaluates IBM numeric-feature K-Means candidates k=2..7.

**Selected solution:** k=2, SSE 29253.11, Silhouette 0.1529, Davies-Bouldin 2.3825.

**Interpretation:** the selected solution has weak separation and should be treated as exploratory segmentation.

**Limitation:** low Silhouette warns against strong substantive cluster claims.
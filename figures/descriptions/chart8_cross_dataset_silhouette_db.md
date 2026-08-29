# chart8_cross_dataset_silhouette_db

**Purpose:** compares selected K-Means solutions using Silhouette and Davies-Bouldin indices across the four analyses.

**Results:** IBM k=2, Silhouette 0.1529, DB 2.3825; Job Change k=3, 0.5773, 0.6639; Promotion k=5, 0.2565, 1.2695; GHRM k=2, 0.4363, 0.8611.

**Interpretation:** Job Change shows the clearest internal separation; IBM is weak.

**Limitation:** cluster quality indices measure geometric separation, not causal or substantive HR mechanisms.
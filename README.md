## Predicting Team Outcomes Based on Player Composition

#### Charles Williams, Jeffrey Vartabedian, and Ryan Peet

### Overview

This project aims to predict Major League Baseball team outcomes (regular season win totals) based on player composition using both unsupervised and supervised machine learning techniques.
By integrating player performance data, salary information, and demographic context, we explore how team structure and player archetypes contribute to regular season success.

### Project Workflow

The project consists of four Jupyter notebooks, which should be executed in the following order:

---

1. `data_merge.ipynb`

Aggregates and cleans data from multiple sources (Baseball Reference and Spotrac).

Standardizes naming conventions and merges player statistics, salaries, and team-level data.

---

2. `cluster_feature_selection.ipynb`

Performs feature selection on batting and pitching data using statistical and clustering-based methods (e.g., Predictive Attribute Dependence, Correlation Pruning, PCA).

Prunes redundant features to prepare datasets for unsupervised learning.

---

3. `unsupervised.ipynb`

Uses clustering algorithms (e.g., K-Means, PCA) to identify player archetypes for position players and pitchers.

---

4. `supervised.ipynb`

Builds and evaluates supervised models predicting team success metrics based on aggregated cluster compositions.

### Data Sources

#### **Baseball-Reference**
- **Format Used:** Copy-pasted tabular data stored in .csv files  
- **Variables of Interest:**

  **Position Players:**  
  `Name`, `Team`, `Age`, `2B`, `3B`, `HR`, `SO`, `BB`, `IBB`, `HBP`, `BA`, `OPS`, `SB`, `CS`, `SH`, `GIDP`, `WAR`  

  **Pitchers:**  
  `Name`, `Team`, `Age`, `W`, `ERA`, `ERA+`, `WHIP`, `FIP`, `GF`, `CG`, `SHO`, `SO/BB`, `IBB`, `BB9`, `BK`, `WP`, `HBP`, `WAR`  

- **Records Used:** 31,497 individual player-season records  
- **Time Period:** 2011–2024  

---

#### **Spotrac**
- **Format Used:** Copy-pasted tabular data stored in .csv files  
- **Variables of Interest:**  
  `Name`, `Team`, `Season`, `Salary`  
- **Records Used:** 15,382 player salary records  
- **Time Period:** 2011-2024  

---

#### **Wikipedia (U.S. Census Bureau)**
- **Format Used:** Copy-pasted tabular data in .csv format  
- **Variables of Interest:**  
  `Population (Millions)`  
- **Records Used:** 420 records  

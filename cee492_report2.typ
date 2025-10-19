= Deliverable 2: Exploratory Data Analysis and Predictive Modeling Plan

== 1. Exploratory Data Analysis

=== 1.1 Dataset Overview
The Concrete Compressive Strength dataset contains 1030 samples with 8 input features and one output variable. No missing values exist.

#table(
  columns: (1cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1cm, 1.5cm),
  align: center,
  table.header(
    [*Row*], [*Cement*], [*Slag*], [*FlyAsh*], [*Water*], [*SuperP*], [*CA*], [*FA*], [*Age*], [*Strength*]
  ),
  [1], [540.0], [0.0], [0.0], [162.0], [2.5], [1040.0], [676.0], [28], [79.98611076],
  [2], [540.0], [0.0], [0.0], [162.0], [2.5], [1055.0], [676.0], [28], [61.88736576],
  [3], [332.5], [142.5], [0.0], [228.0], [0.0], [932.0], [594.0], [270], [40.26953526],
  [4], [332.5], [142.5], [0.0], [228.0], [0.0], [932.0], [594.0], [365], [41.05277999],
  [5], [198.6], [132.4], [0.0], [192.0], [0.0], [978.4], [825.5], [360], [44.29607510]
)

=== 1.2 Summary Statistics
#table(
  columns: (3.5cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm),
  align: center,
  table.header(
    [*Feature*], [*Min*], [*Max*], [*Mean*], [*Median*], [*StdDev*]
  ),
  [Cement (component 1)(kg in a m³ mixture)], [102.0], [540.0], [281.17], [272.9], [104.51],
  [Blast Furnace Slag (component 2)(kg in a m³ mixture)], [0.0], [359.4], [73.9], [22.0], [86.28],
  [Fly Ash (component 3)(kg in a m³ mixture)], [0.0], [200.1], [54.19], [0.0], [64.0],
  [Water (component 4)(kg in a m³ mixture)], [121.75], [247.0], [181.57], [185.0], [21.36],
  [Superplasticizer (component 5)(kg in a m³ mixture)], [0.0], [32.2], [6.2], [6.35], [5.97],
  [Coarse Aggregate (component 6)(kg in a m³ mixture)], [801.0], [1145.0], [972.92], [968.0], [77.75],
  [Fine Aggregate (component 7)(kg in a m³ mixture)], [594.0], [992.6], [773.58], [779.51], [80.18],
  [Age (day)], [1], [365], [45.66], [28.0], [63.17]
)

=== 1.3 Data Visualizations
#figure(
  image("histograms.png", width: 80%),
  caption: [Histograms of input features]
)

#figure(
  image("scatter.png", width: 80%),
  caption: [Scatter plots of input features vs compressive strength]
)

#figure(
  image("corr.png", width: 80%),
  caption: [Correlation heatmap]
)

#figure(
  image("boxplot.png", width: 80%),
  caption: [Boxplots of input features]
)

#figure(
  image("pca.png", width: 80%),
  caption: [PCA plot of input features]
)

== 2. Predictive Modeling Plan

#list(
  [Baseline: Linear Regression],
  [Advanced: Decision Tree, Random Forest, Neural Networks],
  [Evaluation Metrics: R², RMSE, MAE],
  [Cross-validation: k-fold (k=5 or 10)],
  [Feature Importance Analysis]
)
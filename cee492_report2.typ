= Deliverable 2: Exploratory Data Analysis and Predictive Modeling Plan

== 1. Exploratory Data Analysis

=== 1.1 Dataset Overview
The Concrete Compressive Strength dataset contains 1030 samples with 8 input features and one output variable. No missing values exist.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
  table.header([*Row*],[*Cement*],[*Slag*],[*FlyAsh*],[*Water*],[*SuperP*],[*CA*],[*FA*],[*Age*],[*Strength*]),
1,540.0,0.0,0.0,162.0,2.5,1040.0,676.0,28,79.98611076
2,540.0,0.0,0.0,162.0,2.5,1055.0,676.0,28,61.887365759999994
3,332.5,142.5,0.0,228.0,0.0,932.0,594.0,270,40.269535256000005
4,332.5,142.5,0.0,228.0,0.0,932.0,594.0,365,41.052779992
5,198.6,132.4,0.0,192.0,0.0,978.4,825.5,360,44.296075096
)

=== 1.2 Summary Statistics
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  table.header([*Feature*],[*Min*],[*Max*],[*Mean*],[*Median*],[*StdDev*]),
Cement  (Kg/m3),102.0,540.0,281.17,272.9,104.51
Blast Furnace Slag  (Kg/m3),0.0,359.4,73.9,22.0,86.28
Fly Ash  (Kg/m3),0.0,200.1,54.19,0.0,64.0
Water  (Kg/m3),121.75,247.0,181.57,185.0,21.36
Superplasticizer  (Kg/m3),0.0,32.2,6.2,6.35,5.97
Coarse Aggregate  (Kg/m3),801.0,1145.0,972.92,968.0,77.75
Fine Aggregate (Kg/m3),594.0,992.6,773.58,779.51,80.18
Age (day),1,365,45.66,28.0,63.17
)

=== 1.3 Data Visualizations
#figure("histograms.png", caption: [Histograms of input features])
#figure("scatter.png", caption: [Scatter plots of input features vs compressive strength])
#figure("corr.png", caption: [Correlation heatmap])
#figure("boxplot.png", caption: [Boxplots of input features])
#figure("pca.png", caption: [PCA plot of input features])

== 2. Predictive Modeling Plan
- Baseline: Linear Regression
- Advanced: Decision Tree, Random Forest, Neural Networks
- Evaluation Metrics: R², RMSE, MAE
- Cross-validation: k-fold (k=5 or 10)
- Feature Importance Analysis

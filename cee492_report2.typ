#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Predicting Concrete Compressive Strength using Machine Learning],
  abstract: [
This project investigates the prediction of concrete compressive strength using machine learning models. We use the Concrete Compressive Strength dataset from the UCI Machine Learning Repository, which contains 1030 experimental data points with eight input variables (cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and curing age) and one output variable (compressive strength in MPa). Our study compares linear regression, decision tree regression, and neural networks to determine the most effective approach for modeling the nonlinear relationships that influence the development of concrete strength.
  ],

  authors: (
    (
      name: "Osama Ibrahim",
      department: [Department of Civil and Environmental Engineering],
      organization: [University of Illinois Urbana-Champaign],
      location: [Urbana, IL, USA],
      email: "osamani2@illinois.edu",
    ),
    (
      name: "Praneeth Shivashankarappa",
      department: [Department of Civil and Environmental Engineering],
      organization: [University of Illinois Urbana-Champaign],
      location: [Urbana, IL, USA],
      email: "ps104@illinois.edu",
    ),
    (
      name: "Kazi Ishat Mushfiq",
      department: [Department of Civil and Environmental Engineering],
      organization: [University of Illinois Urbana-Champaign],
      location: [Urbana, IL, USA],
      email: "mushfiq2@illinois.edu",
    ),
        (
      name: "Georg Bauer",
      department: [Department of Civil and Environmental Engineering],
      organization: [University of Illinois Urbana-Champaign],
      location: [Urbana, IL, USA],
      email: "georgb2@illinois.edu",
    ),
  ),
  index-terms: ("Concrete Compressive Strength", "Machine Learning", "Civil Engineering", "Regression Models"),
  bibliography: bibliography("refs.bib"),
)

= Part (2): Exploratory Data Analysis

This section presents the *Exploratory Data Analysis (EDA)* conducted on the *Concrete Compressive Strength Dataset*. The primary objective of this analysis was to understand the distribution, relationships, and statistical characteristics of the data through systematic cleaning, visualization, and interpretation steps. The structure follows the logical progression of the analysis as carried out in the notebook.


== Data Description and Cleaning
The Concrete Compressive Strength dataset originates from laboratory experiments conducted by Prof. I-Cheng Yeh, who donated it to the UCI Machine Learning Repository in 2007 to support research on high-performance concrete @yeh2007, @uci-dataset. The dataset records the quantities of concrete mix ingredients and the curing age, together with the corresponding compressive strength as the output variable. In total, it includes 1030 samples in a CSV file, with each row representing one concrete mix. All ingredient quantities are reported per cubic meter of concrete, and the compressive strength is expressed as a continuous variable in megapascals (MPa), representing the material’s capacity to withstand compressive loads. For this project, we will obtain the dataset directly from the UCI Machine Learning Repository @uci-dataset, ensuring a reliable source. The variables included are listed below.

The Concrete Compressive Strength Dataset originates from laboratory experiments conducted by Prof. I-Cheng Yeh, who donated it to the UCI Machine Learning Repository in 2007 to support research on high-performance concrete [@yeh2007; @uci-dataset]. The dataset records the quantities of concrete mix ingredients and the curing age, together with the corresponding compressive strength as the output variable. In total, it contains 1,030 observations with nine continuous numerical variables, each describing the material composition and performance characteristics of various concrete mixtures. All ingredient quantities are reported per cubic meter of concrete, and the compressive strength is expressed as a continuous variable in megapascals (MPa), representing the material’s capacity to withstand compressive loads. The target variable is the Concrete Compressive Strength (MPa), measured after a specified curing period (Age). For this project, the dataset is obtained directly from the UCI Machine Learning Repository [@u]()

#v(0.75em)

*Variables included in the dataset:*

- Cement (kg/m³)  
- Blast Furnace Slag (kg/m³)  
- Fly Ash (kg/m³)  
- Water (kg/m³)  
- Superplasticizer (kg/m³)  
- Coarse Aggregate (kg/m³)  
- Fine Aggregate (kg/m³)  
- Age (days, from 1–365)

#v(0.5em)

*Target variable:*

- Concrete Compressive Strength (MPa)

#v(0.5em)

All variables were confirmed to be numerical (`Float64`) and expressed in consistent units (kg/m³ for material quantities, days for age).

During data preprocessing:

- **25 duplicate rows** (≈2.4%) were identified and removed.
- **No missing values** were found.
- The dataset was thus reduced to **1,005 unique entries**, fully ready for statistical analysis and visualization.


== Sample Data Preview

Table I provides representative entries from the Concrete Compressive Strength dataset, illustrating the input features and the output variable. Each row corresponds to one concrete mix, with ingredient quantities and age shown alongside the measured compressive strength in MPa.

#figure(
  caption: [Representative Entries from the Concrete Compressive Strength Dataset],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    table.header(
      [#text(size: 6pt)[*Row*]],
      [#text(size: 6pt)[*Cement*]],
      [#text(size: 6pt)[*Slag*]],
      [#text(size: 6pt)[*FlyAsh*]],
      [#text(size: 6pt)[*Water*]],
      [#text(size: 6pt)[*SuperP.*]],
      [#text(size: 6pt)[*CA*]],
      [#text(size: 6pt)[*FA*]],
      [#text(size: 6pt)[*Age*]],
      [#text(size: 6pt)[*Strength*]],
    ),
    "1","540.0","0.0","0.0","162.0","2.5","1040.0","676.0","28","79.99",
    "2","540.0","0.0","0.0","162.0","2.5","1055.0","676.0","28","61.89",
    "3","332.5","142.5","0.0","228.0","0.0","932.0","594.0","270","40.27",
    "4","332.5","142.5","0.0","228.0","0.0","932.0","594.0","365","41.05",
    "5","198.6","132.4","0.0","192.0","0.0","978.0","825.0","360","44.30",
    "...","...","...","...","...","...","...","...","...","...",
    "...","...","...","...","...","...","...","...","...","...",
    "1028","148.5","139.4","108.6","192.7","6.1","892.4","780.0","28","23.70",
    "1029","159.1","186.7","0.0","175.6","11.3","989.6","788.9","28","32.77",
    "1030","260.9","100.5","78.3","200.6","8.6","864.5","761.5","28","32.40",
  )
)


== Summary Statistics

Descriptive statistics were computed for all variables to establish a preliminary understanding of their ranges and central tendencies. The table below summarizes the minimum, maximum, mean, and median values for each feature, providing an overview of the composition and variability within the concrete mixtures.

#figure(
  caption: [Summary statistics of concrete mix features],
  placement: none,
  table(
    columns: (3.5cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm),
    align: center,
    table.header(
      [Feature], [Min], [Max], [Mean], [Median], [N-Missing]
    ),
    [Cement (kg/m³)], [102.0], [540.0], [281.17], [272.9], [0],
    [Blast Furnace Slag (kg/m³)], [0.0], [359.4], [73.90], [22.0], [0],
    [Fly Ash (kg/m³)], [0.0], [200.1], [54.19], [0.0], [0],
    [Water (kg/m³)], [121.75], [247.0], [181.57], [185.0], [0],
    [Superplasticizer (kg/m³)], [0.0], [32.2], [6.20], [6.35], [0],
    [Coarse Aggregate (kg/m³)], [801.0], [1145.0], [972.92], [968.0], [0],
    [Fine Aggregate (kg/m³)], [594.0], [992.6], [773.58], [779.51], [0],
    [Age (day)], [1], [365], [45.66], [28.0], [0],
    [Concrete compressive strength (MPa)], [2.33], [82.60], [35.82], [34.44], [0]
  )
)

The summary statistics highlight several important characteristics of the dataset that help contextualize subsequent analyses. Key findings include:

- The dataset covers a wide range of mix proportions, from lean to rich cement contents.  
- The compressive strength spans from 2.3 to 82.6 MPa, with an average of around 35.8 MPa, indicating substantial variation in concrete performance.  
- Cement, water, and age exhibit the widest value ranges among all variables, reflecting diverse mix designs and curing durations. These parameters are also known to have the strongest theoretical influence on compressive strength, making them primary candidates for deeper analysis.


== Univariate Analysis
This section examines each variable in isolation to understand its individual distribution and statistical properties. By analyzing the spread, central tendency, and shape of each feature, we can identify potential outliers, skewness, and data patterns that may influence later multivariate analyses.


=== Box Plot Analysis of Compressive Strength
The box plot indicates that the median compressive strength of the concrete samples is approximately 35 MPa, aligning well with earlier summary statistics. The interquartile range (IQR) is relatively wide, suggesting that the dataset covers a broad spectrum of strength values, likely reflecting variation in mix proportions and curing conditions.

The right whisker extends noticeably farther than the left, implying a slight positive (right) skew in the distribution. This means that while most concrete samples cluster around the median, a few specimens achieve substantially higher strengths than the majority.

Additionally, the presence of outliers above the upper whisker indicates that some mixes produced exceptionally strong concrete. These cases may correspond to specific combinations of high cement content, optimal water-to-cement ratio, or longer curing times. Further analysis, such as correlating mix components with strength, helps explain these unusually high values.

## BOXPLOT


=== Histogram of Compressive Strength

The histogram of compressive strength values shows a distribution that is approximately unimodal, with most observations concentrated around 30–40 MPa.  
However, the distribution is not perfectly symmetric because it exhibits a noticeable positive skew.

The right tail extends farther than the left, indicating that while the majority of concrete samples achieve moderate compressive strengths, a few cases reach significantly higher values. This suggests that certain mix proportions or curing conditions occasionally produce concrete that is much stronger than the typical range.

Overall, the histogram supports the findings from the box plot: the data are slightly skewed to the right, with a handful of high-strength outliers pulling the mean above the median.

## HISTOGRAM 1


=== Input Variable Distributions

After analyzing the distribution of the target variable (compressive strength), the next step involves examining the input features. The histograms below illustrate the distributions of all mix components, revealing that most input variables are not normally distributed and display distinct skewness and concentration patterns across the dataset.

- **Cement:** Roughly uniform between 100–500 kg/m³, with a slight peak in the 150–300 kg/m³ range, indicating diverse mix designs from lean to rich cement contents.
- **Blast Furnace Slag:** Strongly right-skewed; most mixes contain little or no slag, suggesting it was used only selectively.
- **Fly Ash:** Heavily right-skewed; used as partial cement replacement in a minority of mixes.
- **Water:** Roughly normal distribution centered around 175–200 kg/m³, consistent with typical mix designs.
- **Superplasticizer:** Right-skewed; most mixes include very small dosages (<10 kg/m³), with a few outliers up to 30 kg/m³.
- **Coarse Aggregate:** Fairly uniform, concentrated around 900–1000 kg/m³, suggesting consistent proportions.
- **Fine Aggregate:** Approximately normal, centered near 750–800 kg/m³, typical for balanced mix ratios.
- **Age:** Extremely right-skewed. most tests were conducted within 50 days, with few extending beyond 300 days, meaning early-age data dominate the dataset.

## HISTOGRAM 2


=== Bivariate Analysis: Compressive Strength vs. Age

The relationship between compressive strength and curing age is of particular interest, as the project proposal identified the time-dependent development of concrete strength as a central aspect of investigation.  
Accordingly, a bivariate analysis was conducted to examine how compressive strength evolves with increasing age and to assess whether a consistent trend can be observed across the dataset.

## SCATTERPLOTT

The scatter plot reveals that extremely long curing periods (beyond ~350 days) tend to produce concrete with average rather than exceptionally high strengths.  
This could suggest that strength eventually plateaus or declines slightly over extended ageing times depending on curing conditions and mix design.

The maximum strengths are generally observed for specimens aged between 25 and 100 days, indicating that this period represents an optimal curing window for strength development.

There are also a few outlier cases where concrete achieved very high strengths (>60 MPa and >70 MPa) at ages exceeding 150 days. This implies that other factors, such as cement content or admixtures, also play a significant role.

The 0–50 day range shows a particularly wide variation in strength, emphasizing that age alone cannot fully explain the observed differences. While curing time clearly influences strength, the relationship is nonlinear and mediated by multiple interacting mix parameters.


== Correlation Analysis

Following the individual and pairwise analyses of key variables, a correlation heatmap was generated to provide a comprehensive overview of how all input factors, particularly the various concrete mix ingredients, relate to one another and to the target variable. This step helps identify the most influential predictors of compressive strength and potential interdependencies within the dataset.

## HEATMAP

The correlation analysis offers valuable insights into the interdependencies among input variables:

- **Cement content** shows the strongest positive correlation with strength (~0.5), confirming its role as the primary strength determinant.  
- **Age** exhibits a moderate positive correlation (~0.33), supporting the continuous but diminishing strength gain over time.  
- **Superplasticizer** correlates positively (~0.37) with strength, indicating improved compaction and hydration efficiency.  
- **Water content** correlates negatively (~–0.29) with strength and strongly negatively (~–0.66) with superplasticizer, aligning with the water-to-cement ratio principle, which states that higher water content increases porosity and thus weakens the hardened concrete matrix. 
- **Blast Furnace Slag** and **Fly Ash** show weak or slightly negative correlations, suggesting their contribution depends on dosage and curing time.  
- **Aggregate contents** show weak relationships with most other variables, implying consistent proportions across samples.

Overall, the dataset supports classical concrete mix design theory: Compressive strength is primarily governed by cement content, curing age, and the water-to-cement ratio, while supplementary materials play secondary roles.



== Multivariate Analysis: PairPlot Interpretation

Building on these correlation-based insights, a contour-based PairPlot was created to further explore nonlinear dependencies and multivariate interactions among the most influential variables: Cement (Kg/m³), Water (Kg/m³), Age (day), and Compressive Strength (MPa). Unlike the static correlation heatmap, the PairPlot additionally reveals nonlinear trends, distribution shapes, and clustering patterns, offering a deeper and more physical understanding of how these factors jointly influence concrete strength.

## PairePlot

The key insights derived from the contour-based PairPlot can be found below:

- **Cement vs. Strength:** The contours show a clear positive trend, indicating that higher cement content consistently leads to greater compressive strength.  
- **Water vs. Strength:** A distinct negative dependency is visible, where increasing water content results in lower strength, confirming the influence of the water-to-cement ratio.  
- **Age vs. Strength:** Strength increases rapidly during the first 100 days and then approaches a plateau, demonstrating the diminishing effect of prolonged curing.  
- **Cement vs. Water:** The contours reveal a mild inverse relationship, suggesting that mixtures were designed to maintain a controlled water-to-cement balance.






















== Project Proposal

=== Planned Approach
In this project, the Concrete Compressive Strength dataset will be analyzed from a data science perspective. Techniques learned in CEE 492 will be applied to perform exploratory data analysis, develop machine learning models, and investigate how each ingredient in the dataset influences the resulting compressive strength. This begins with ensuring the dataset is properly structured for analysis, followed by looking at the distributions of the variables and their relationships. We also plan to explore whether Principal Component Analysis provides useful insights into correlations among the input variables.

After the exploratory phase, we will develop predictive models for compressive strength. Multiple linear regression will serve as the baseline, against which we will compare more advanced approaches such as decision tree models and neural networks. Finally, we will test the trained models on the dataset and check how well they can predict compressive strength, using common accuracy measures.

=== Relevance
Compressive strength is one of the most important properties of a concrete mix, and being able to estimate it accurately is essential in civil engineering, as it can help engineers decide whether a given mixture will meet the required strength for a project. Laboratory testing is the standard approach, but it can be time-consuming and resource-intensive. Empirical equations from design codes can be used as an alternative, but they often lack accuracy and flexibility across different mix proportions and curing conditions, since concrete compressive strength is a highly nonlinear function of both age and ingredient proportions. This creates a need for faster, more reliable methods of prediction.

This project addresses that need by using machine learning to test how well different models can predict compressive strength. A good model can give faster and more consistent estimates of concrete strength for different mix proportions and curing ages, making it a useful tool in engineering practice. The project will also show which ingredients are most important and how curing age affects strength, helping to better understand the factors that influence concrete performance.

=== Deliverables
The deliverables include an exploratory analysis of the dataset, the development and testing of different models to predict compressive strength, a comparison of their performance, and a discussion of the results to show how mix design and curing age influence concrete strength.


= Part (2): Exploratory Data Analysis and Predictive Modeling Plan

== Exploratory Data Analysis

=== Dataset Overview
The Concrete Compressive Strength dataset contains 1030 samples with 8 input features and one output variable. As shown in Table (I). And a comprehensive desception of data was mentioned earlier in part (1-A) of the study.


=== Summary Statistics
In this part of the exploratory data analysis (EDA), a statistical analysis is conducted and summarized in Table (II). The dataset contains key input variables that influence the compressive strength of concrete, including the quantities of cement, supplementary cementitious materials (blast furnace slag and fly ash), water, superplasticizer, and aggregates, as well as the curing age. Table (II) summarizes the basic descriptive statistics, minimum, maximum, mean, median, and standard deviation, for each feature.

These statistics serve as a solid basis to see the distribution and variability of the dataset. For example, a wide range of cement content (102–540 kg/m³), as well as testing age (1–365 days) can be noticed, which reflect the diversity in the dataset in terms of mix design and testing age. This variability is beneficial for a broader forecasting of performance, as well as a reliable model learning that is not only exclusive to specific small-range dataset. Also, the standard deviations shows the heterogeneity among samples, which reflects a potential nonlinear relationships. This type of data overview is required as a starting point in any EDA, as it helps to see the data variability, assess the existence of outliers, and address the variables correlations. These findings will enhance the future analysis and the building of the model itself.

#figure(
  caption: [Summary statistics of concrete mix features],
  placement: none,
  table(
    columns: (3.5cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm, 1.2cm),
    align: center,
    table.header(
      [Feature], [Min], [Max], [Mean], [Median], [N-Missing]
    ),
    [Cement (kg/m³)], [102.0], [540.0], [281.17], [272.9], [0],
    [Blast Furnace Slag (kg/m³)], [0.0], [359.4], [73.90], [22.0], [0],
    [Fly Ash (kg/m³)], [0.0], [200.1], [54.19], [0.0], [0],
    [Water (kg/m³)], [121.75], [247.0], [181.57], [185.0], [0],
    [Superplasticizer (kg/m³)], [0.0], [32.2], [6.20], [6.35], [0],
    [Coarse Aggregate (kg/m³)], [801.0], [1145.0], [972.92], [968.0], [0],
    [Fine Aggregate (kg/m³)], [594.0], [992.6], [773.58], [779.51], [0],
    [Age (day)], [1], [365], [45.66], [28.0], [0],
    [Concrete compressive strength (MPa)], [2.33], [82.60], [35.82], [34.44], [0]
  )
)

=== Data Visualizations
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

== Predictive Modeling Plan

#list(
  [Baseline: Linear Regression],
  [Advanced: Decision Tree, Random Forest, Neural Networks],
  [Evaluation Metrics: R², RMSE, MAE],
  [Cross-validation: k-fold (k=5 or 10)],
  [Feature Importance Analysis]
)















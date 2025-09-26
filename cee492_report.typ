#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Predicting Concrete Compressive Strength using Machine Learning],
  abstract: [
      This project investigates the prediction of concrete compressive strength using machine learning models. We use the         Concrete Compressive Strength dataset from the UCI Machine Learning Repository, which contains 1,030 experimental           data points with eight input variables (cement, blast furnace slag, fly ash, water, superplasticizer, coarse                aggregate, fine aggregate, and curing age) and one output variable (compressive strength in MPa). Our study compares        linear regression, decision tree regression, and neural networks to determine the most effective approach for               capturing the nonlinear relationships that govern the development of concrete strength.
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

= Dataset Description
The Concrete Compressive Strength dataset originates from laboratory experiments conducted by Prof. I-Cheng Yeh, who donated it to the UCI Machine Learning Repository in 2007 to support research on high-performance concrete @yeh2007, @uci-dataset. It contains measurements of the concrete mix ingredients and the age of the concrete, paired with the corresponding compressive strength. In total, the dataset includes 1,030 data points, each representing one concrete sample. There are nine variables: eight input features (cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age) and one output variable (compressive strength in megapascals). All ingredient quantities are reported per cubic meter of concrete mix, and the compressive strength is recorded as a continuous variable representing the capacity of concrete to withstand loads that act to compress it.\

While strength typically increases with curing age, the relationship is highly nonlinear and strongly influenced by the proportions of the mix components. The dataset has no missing values, is provided in a structured tabular format (CSV file) with each row corresponding to a unique sample, and is widely recognized in both the civil engineering and machine learning communities as a benchmark for modeling concrete strength. For our project, we will obtain the dataset directly from the UCI Machine Learning Repository @uci-dataset, ensuring a reliable and well-documented source.\

*Variables included in the dataset:*
- Cement (kg/m³)  
- Blast Furnace Slag (kg/m³)  
- Fly Ash (kg/m³)  
- Water (kg/m³)  
- Superplasticizer (kg/m³)  
- Coarse Aggregate (kg/m³)  
- Fine Aggregate (kg/m³)  
- Age (days, from 1–365)\  

*Target variable:*
- Concrete Compressive Strength (MPa)


= Sample Data Preview

Table I shows a preview of the dataset, showing the first 10 rows, then skipping to the last 3.

#figure(
  caption: [Concrete Compressive Strength dataset],
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
    "6","266.0","114.0","0.0","228.0","0.0","932.0","670.0","90","47.03",
    "7","380.0","95.0","0.0","228.0","0.0","932.0","594.0","365","43.70",
    "8","380.0","95.0","0.0","228.0","0.0","932.0","594.0","28","36.45",
    "9","266.0","114.0","0.0","228.0","0.0","932.0","670.0","28","45.85",
    "10","475.0","0.0","0.0","228.0","0.0","932.0","594.0","28","39.29",
    "...","...","...","...","...","...","...","...","...","...",
    "...","...","...","...","...","...","...","...","...","...",
    "1028","148.5","139.4","108.6","192.7","6.1","892.4","780.0","28","23.70",
    "1029","159.1","186.7","0.0","175.6","11.3","989.6","788.9","28","32.77",
    "1030","260.9","100.5","78.3","200.6","8.6","864.5","761.5","28","32.40",
  )
)



= Project Proposal

== Planned Approach
In this project, the Concrete Compressive Strength dataset will be analyzed from a data science perspective. Techniques learned in CEE 492 will be applied to perform exploratory data analysis, develop machine learning models, and investigate how each ingredient in the dataset influences the resulting compressive strength. This begins with ensuring the dataset is properly structured for analysis, followed by examining the distributions of the variables and their relationships. For example, we will explore how curing age affects achieved strength. Principal Component Analysis (PCA) will also be employed to assess variance across the input variables and potentially reduce dimensionality.

After the exploratory phase, we will develop predictive models for compressive strength. Multiple linear regression will serve as the baseline, against which we will compare more advanced approaches such as decision tree regression and neural networks. Finally, the trained models will be validated using the available data, with performance evaluated through metrics such as mean squared error and the coefficient of determination (R-squared).

== Relevance
Compressive strength is one of the most important properties of a concrete mix, and understanding the relationship between mix design and strength has long been a central focus in civil engineering. Traditionally, strength has been estimated using empirical equations provided in design codes and technical literature. While useful, these equations can be limited in accuracy, time-consuming to apply, and often require laboratory testing for validation.

In this project, we will apply machine learning algorithms to evaluate their reliability in predicting compressive strength, with the goal of selecting the most suitable model for the dataset. Such a model offers a fast, efficient, and reliable alternative for estimating concrete strength across different mix proportions and curing ages. Our deliverables will include exploratory data analysis, the development and comparison of multiple predictive models, and a discussion of the results, demonstrating the application of data science techniques to a real-world civil engineering problem.

== Deliverables  
- Exploratory analysis  
- Predictive model development and comparison  
- Results discussion with insights into concrete mix design  

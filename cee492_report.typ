#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Predicting Concrete Compressive Strength using Machine Learning],
  abstract: [
    This project investigates the prediction of concrete compressive strength using machine learning models.
    We use the Concrete Compressive Strength dataset from the UCI Machine Learning Repository, which includes
    1,030 experimental data points with 8 input variables (cement, slag, fly ash, water, superplasticizer,
    coarse aggregate, fine aggregate, and curing age) and one output (compressive strength in MPa).
    Our study compares linear regression, decision tree regression, and neural networks to determine the
    most effective approach for capturing the nonlinear relationships governing concrete strength development.
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
Learning Repository @uci-dataset. Noting that, this data came from lab experiments done by Prof. I-Cheng Yeh. He donated it to UCI back in 2007. The whole point was to look into high-performance concrete @yeh2007. The dataset has measurements for all the ingredients in the concrete mix. It also includes the age of the concrete. Those get paired up with the final compressive strength. In total, there are 1030 data points. Each one is an instance. You get 9 variables overall. Eight of them are features, the input variables. Then there is one target, the output variable. No missing values at all for any of the instances. The data comes in a structured table, as CSV file. Every row stands for one concrete sample. Now, compressive strength for concrete turns into a regression problem. The goal here is to predict that compressive strength value. It's measured in megapascals. You pull that from the inputs.
*Variables included in the dataset:*
- Cement (kg/m³)  
- Blast Furnace Slag (kg/m³)  
- Fly Ash (kg/m³)  
- Water (kg/m³)  
- Superplasticizer (kg/m³)  
- Coarse Aggregate (kg/m³)  
- Fine Aggregate (kg/m³)  
- Age (days, from 1–365)  

*Target variable:*
- Concrete Compressive Strength (MPa)

All ingredient quantities are given per cubic meter of concrete mix. The compressive strength is the capacity of the concrete to withstand loads that tend to compress it (push it together). Concrete strength generally increases with age as the concrete cures, but the relationship is highly nonlinear and depends on the mix proportions. This dataset is well-known in the civil engineering and machine learning communities for modeling concrete strength and has been used in prior research. We will obtain the dataset from the UCI repository @uci-dataset, which ensures we have a reliable source of the data.


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
In this project, a concrete data set will be analyzed from a data science perspective. Techniques learnt during the CEE 492 class will be utilized to do exploratory data analysis, a Machine learning model, and look into how each ingredient from the dataset affects the output result of the compressive strength. That means getting a feel for the data distribution and how the variables connect with each other. 
First step will be making sure our data set is tidy enough for analysis, and then we will figure out and visualize the different main trends or patterns. For instance, we can figure out the relation between the mix age and its effect on achieved strength. To achieve that, principal component analysis (PCA) will be conducted as it will help to see the variance across the input variables, and may eliminate some dimensions if needed.
Once that's done, the exploratory tasks, we'll move on to building predictive models for the compressive strength. We'll kick things off with multiple linear regression as our baseline. After that, we'll stack it up against more complicated ones, say a decision tree regression model and maybe a neural network. 
Finally, the trained models will be examined against available data to validate the prediction of the model. Metrics, such as the mean squared error or R-squared, will be evaluated to assess the reliability of the model.

== Relevance

The compressive strength is one of the most important characteristics of a concrete mix, and figuring out the relation between the mix design and the strength was always of high importance in civil engineering over the past years.
The strength can be estimated based on available empirical equations in different codes available in the literature. However, these equations tend to be challenging at some point, and are not the fastest way to get accurate estimations. Also, Lab testing may be required to validate the results.
In this project, machine learning algorithms are created and checked for their reliability, and the most suitable model will be selected based on the dataset used.
This created model will help as a useful, fast and reliable method of estimating the mix strength with different parameters and different testing ages.
Our deliverables include data analysis. Developing and comparing multiple models. Discussing the results. It shows data science applied to real civil engineering problems.

== Deliverables  
- Exploratory analysis  
- Predictive model development and comparison  
- Results discussion with insights into concrete mix design  

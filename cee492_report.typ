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

Our project will use the Concrete Compressive Strength dataset, a public dataset from the UCI Machine Learning Repository @uci-dataset. This dataset was originally collected through laboratory experiments by Prof. I-Cheng Yeh (donated to UCI in 2007) to study high-performance concrete @yeh2007. It contains measurements of concrete mix ingredients and the age of the concrete, paired with the resulting compressive strength. There are 1030 data points (instances) in total, with 9 variables: 8 features (input variables) and 1 target (output variable), and no missing values. The data is provided in a structured table format (e.g. as an Excel/CSV file) with each row representing one concrete sample. The concrete compressive strength is a regression problem. the goal is to predict a continuous strength value (in megapascals) from the inputs. 

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

All ingredient quantities are given per cubic meter of concrete mix. The compressive strength is the capacity of the concrete to withstand loads that tend to compress it (push it together). Concrete strength generally increases with age as the concrete cures, but the relationship is highly nonlinear and depends on the mix proportions. This dataset is well-known in the civil engineering and machine learning community for modeling concrete strength, and it has been used in prior research. We will obtain the dataset from the UCI repository @uci-dataset, which ensures we have a reliable source of the data.



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
Our team plans to perform an in-depth analysis of this concrete dataset and develop predictive models for concrete compressive strength using various data science and machine learning techniques covered in the course. We will begin with exploratory data analysis to understand the data distribution and relationships between variables (for example, examining how each ingredient and age correlate with strength). We will ensure the data is in a tidy format and visualize key trends or patterns (such as how strength increases with age for different mixes). We may also apply techniques like principal component analysis (PCA) to explore variance in the ingredient combinations or to reduce dimensionality if needed, as a part of feature analysis. After this exploratory phase, we will build several predictive models for the compressive strength. In particular, we intend to apply multiple linear regression as a baseline model, and then compare it with more advanced models such as a decision tree regression model and a neural network model. These models will be trained on a portion of the data and tested on held-out data to evaluate their performance. We will use appropriate metrics (like mean squared error or R²) to compare how well the different approaches predict the concrete strength.

== Relevance

Predicting concrete compressive strength from its mix composition and age is a practical problem in civil engineering. Being able to accurately estimate the strength without lengthy laboratory tests can help engineers decide whether a given concrete mixture will meet the required strength for a project (for example, determining if the mix design is suitable for a building or infrastructure application). By comparing different machine learning algorithms on this problem, we can identify which modeling technique provides the best balance of accuracy and complexity for our dataset. We expect nonlinear models (like decision trees or neural networks) to potentially capture the complex relationships in the data better than a simple linear model, given that concrete strength depends on ingredients in a nonlinear way. Through this project, we will not only build a useful predictive tool for concrete strength, but also gain insight into the importance of each ingredient (feature importance) and how the curing age influences strength. In summary, our deliverables will include the data analysis, the development and comparison of multiple predictive models, and a discussion of results, helping demonstrate the application of data science techniques to a real-world Civil Engineering problem. 

== Deliverables  
- Exploratory analysis  
- Predictive model development and comparison  
- Results discussion with insights into concrete mix design  

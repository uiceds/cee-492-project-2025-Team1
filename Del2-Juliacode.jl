### Pluto Notebook: Deliverable 2 - Concrete Compressive Strength EDA and Modeling Plan

### 1️⃣ Import Packages
begin
    using CSV, DataFrames
    using Statistics
    using Plots
    using StatsPlots
    using StatsBase
    using MLJ
    using MultivariateStats
end

### 2️⃣ Load Dataset
begin
    dataset_path = "concrete_data.csv"  # Update path as needed
    df = CSV.read(dataset_path, DataFrame)
    first(df, 5)  # Preview first 5 rows
end

### 3️⃣ Summary Statistics
begin
    summary_stats = DataFrame(
        Feature = names(df)[1:end-1],
        Min = [minimum(df[!, col]) for col in names(df)[1:end-1]],
        Max = [maximum(df[!, col]) for col in names(df)[1:end-1]],
        Mean = [mean(df[!, col]) for col in names(df)[1:end-1]],
        Median = [median(df[!, col]) for col in names(df)[1:end-1]],
        StdDev = [std(df[!, col]) for col in names(df)[1:end-1]]
    )
    summary_stats
end

### 4️⃣ Histograms for Input Features
begin
    histogram_plot = plot(layout=(3,3), size=(900,700))
    for (i, col) in enumerate(names(df)[1:end-1])
        histogram!(histogram_plot, df[!, col], bins=20, title=col, subplot=i)
    end
    histogram_plot
end

### 5️⃣ Scatter Plots vs Compressive Strength
begin
    target = :Concrete_compressive_strength  # Adjust if your column name differs
    scatter_plot = plot(layout=(3,3), size=(900,700))
    for (i, col) in enumerate(names(df)[1:end-1])
        scatter!(scatter_plot, df[!, col], df[!, target], xlabel=col, ylabel="Strength (MPa)", title="$col vs Strength", subplot=i)
    end
    scatter_plot
end

### 6️⃣ Correlation Matrix Heatmap
begin
    corr_matrix = cor(Matrix(df))
    corr_heatmap = heatmap(corr_matrix, xticks=(1:9, names(df)), yticks=(1:9, names(df)), c=:coolwarm, title="Correlation Matrix")
    corr_heatmap
end

### 7️⃣ Boxplots to Detect Outliers
begin
    boxplot_plot = plot(layout=(3,3), size=(900,700))
    for (i, col) in enumerate(names(df)[1:end-1])
        boxplot!(boxplot_plot, df[!, col], title=col, subplot=i)
    end
    boxplot_plot
end

### 8️⃣ PCA Plot for Feature Exploration
begin
    X = convert(Matrix, df[:, 1:end-1])
    X_standardized = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    pca_model = fit(PCA, X_standardized; maxoutdim=2)
    X_pca = transform(pca_model, X_standardized)
    pca_plot = scatter(X_pca[:,1], X_pca[:,2], xlabel="PC1", ylabel="PC2", title="PCA of Input Features")
    pca_plot
end

### 9️⃣ Generate Deliverable 2 `.typ` File Dynamically
begin
    open("Deliverable2.typ", "w") do io
        write(io, """
= Deliverable 2: Exploratory Data Analysis and Predictive Modeling Plan

== 1. Exploratory Data Analysis

=== 1.1 Dataset Overview
The Concrete Compressive Strength dataset contains 1030 samples with 8 input features and one output variable (compressive strength in MPa). No missing values exist.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
  table.header([*Row*],[*Cement*],[*Slag*],[*FlyAsh*],[*Water*],[*SuperP*],[*CA*],[*FA*],[*Age*],[*Strength*]),
""")
        for row in first(df,5)
            write(io, join([string(row[i]) for i in 1:10], ",") * "\n")
        end

        write(io, """
)

=== 1.2 Summary Statistics
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  table.header([*Feature*],[*Min*],[*Max*],[*Mean*],[*Median*],[*StdDev*]),
""")
        for row in eachrow(summary_stats)
            write(io, join([row.Feature, string(row.Min), string(row.Max), string(round(row.Mean,digits=2)), string(round(row.Median,digits=2)), string(round(row.StdDev,digits=2))], ",") * "\n")
        end
        write(io, """
)

=== 1.3 Data Visualizations
#figure(caption: [Histograms of input features], plot = histogram_plot)
#figure(caption: [Scatter plots of input features vs compressive strength], plot = scatter_plot)
#figure(caption: [Correlation heatmap], plot = corr_heatmap)
#figure(caption: [Boxplots of input features], plot = boxplot_plot)
#figure(caption: [PCA plot of input features], plot = pca_plot)

== 2. Predictive Modeling Plan
- Baseline: Linear Regression
- Advanced: Decision Tree, Random Forest, Neural Networks
- Evaluation Metrics: R², RMSE, MAE
- Cross-validation: k-fold (k=5 or 10)
- Feature Importance Analysis
""")
    end
end

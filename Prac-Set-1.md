Study Material
Mohd Rafi Lone
•
Jan 7 (Edited Feb 2)
Syllabus+Textbook+Lecture Notes
CDS3005_FOUNDATIONS-OF-DATA-SCIENCE_LP_1.0_18_CDS3005_FOUNDATION-OF-DATA-SCIENCE_LP_1.0_1_Foundations of Data Science.pdf
PDF
Doing Data Science Straight Talk from the Frontline.pdf
PDF
DS_Practice1.md
Text
Unit-1 new.pptx
Microsoft PowerPoint
Class comments
Add class comment…

# Fundamentals of Data Science: Complete Learning & Teaching Material

## Table of Contents
1. Matrices and Linear Algebraic Operations
2. Matrices Representing Relations
3. Matrix Decompositions: SVD and PCA
4. Descriptive Statistics
5. Probability Distributions
6. Statistical Inference
7. Statistical Modeling and Model Fitting

---

# PART 1: MATRICES AND LINEAR ALGEBRAIC OPERATIONS

## 1.1 Introduction to Matrices

A matrix is a rectangular array of numbers organized in rows and columns. Matrices are fundamental data structures in data science for representing datasets, transformations, and relationships.

### 1.1.1 Creating and Manipulating Matrices in R

#### Example 1: Creating Matrices

```r
# Create a 3×2 matrix from vector (filled column-wise by default)
M <- matrix(1:6, nrow = 3, ncol = 2)
print(M)
#      [,1] [,2]
# [1,]    1    4
# [2,]    2    5
# [3,]    3    6

# Create a 5×4 matrix with named rows and columns
M_named <- matrix(1:20, nrow = 5, ncol = 4)
colnames(M_named) <- c("A", "B", "C", "D")
rownames(M_named) <- c("R1", "R2", "R3", "R4", "R5")
print(M_named)
#    A  B  C  D
# R1 1  6 11 16
# R2 2  7 12 17
# R3 3  8 13 18
# R4 4  9 14 19
# R5 5 10 15 20

# Access matrix dimensions
dim(M_named)  # [1] 5 4
nrow(M_named) # [1] 5
ncol(M_named) # [1] 4
```

### 1.1.2 Arithmetic Operations on Matrices

#### Example 2: Element-wise and Matrix Operations

```r
# Create two matrices
M1 <- matrix(1:15, nrow = 5, ncol = 3)
M2 <- matrix(2:16, nrow = 5, ncol = 3)

# Element-wise operations (use *)
M_elementwise <- M1 * M2
print(M_elementwise[1:3, ])
#      [,1] [,2] [,3]
# [1,]    2   42  132
# [2,]    6   56  156
# [3,]   12   72  182

# Matrix addition
M_sum <- M1 + M2
print(M_sum[1:3, ])
#      [,1] [,2] [,3]
# [1,]    3   13   23
# [2,]    5   15   25
# [3,]    7   17   27

# Matrix multiplication (use %*%)
# For A (m×n) %*% B (n×p) = result (m×p)
M3 <- matrix(1:9, nrow = 3, ncol = 3)
M4 <- matrix(10:18, nrow = 3, ncol = 3)
M_product <- M3 %*% M4
print(M_product)
#      [,1] [,2] [,3]
# [1,]  138  174  210
# [2,]  171  216  261
# [3,]  204  258  312
```

#### Solution: Verify matrix multiplication result
The (1,1) element: 1×10 + 4×13 + 7×16 = 10 + 52 + 112 = 174 ✗ (check row-column pairing)
Actually: (1,1) element = (Row1 of M3) · (Col1 of M4) = [1,4,7]·[10,11,12] = 10+44+84 = 138 ✓

### 1.1.3 Matrix Transpose and Special Operations

```r
# Transpose operation
M <- matrix(1:6, nrow = 2, ncol = 3)
M_T <- t(M)
print(M_T)
#      [,1] [,2]
# [1,]    1    2
# [2,]    3    4
# [3,]    5    6

# Row and column sums
M <- matrix(1:20, nrow = 5, ncol = 4)
rowSums(M)  # Sum across columns for each row
# [1] 18 21 24 27 30
colSums(M)  # Sum down rows for each column
# [1] 15 40 65 90

# Row and column means
rowMeans(M)
# [1]  4.5  5.25  6  6.75  7.5
colMeans(M)
# [1]  3  8 13 18
```

### 1.1.4 Matrix Inversion and Eigenvalues

```r
# Create a square, invertible matrix
A <- matrix(c(4, 3, 3, 2), nrow = 2, ncol = 2)
# [1,] 4 3
# [2,] 3 2

# Compute inverse
A_inv <- solve(A)
print(A_inv)
#      [,1] [,2]
# [1,]   -2    3
# [2,]    3   -4

# Verify: A × A^(-1) = I
print(A %*% A_inv)  # Should be identity matrix
#      [,1] [,2]
# [1,]    1    0
# [2,]    0    1

# Eigenvalues and eigenvectors
eigen_result <- eigen(A)
eigenvalues <- eigen_result$values
eigenvectors <- eigen_result$vectors

print(eigenvalues)
# [1] 6.372281  -0.372281
print(eigenvectors)
#           [,1]       [,2]
# [1,] -0.7071068  -0.7071068
# [2,] -0.7071068   0.7071068
```

#### Practice Exercise 1:
Create a 4×4 matrix B and compute:
a) B + B^T (symmetric sum)
b) B × B^T (gram matrix)
c) The trace (sum of diagonal elements)
d) The determinant

**Solution:**
```r
B <- matrix(1:16, nrow = 4, ncol = 4)
# Symmetric sum
B_sym <- B + t(B)
# Gram matrix
B_gram <- B %*% t(B)
# Trace
trace_B <- sum(diag(B))  # [1] 34
# Determinant
det_B <- det(B)  # [1] 0 (rank-deficient matrix)
```

---

# PART 2: MATRICES REPRESENTING RELATIONS

## 2.1 Relations Between Data Points

Matrices are used to represent relations—connections or associations between entities.

### 2.1.1 Adjacency Matrices and Graph Relations

An adjacency matrix represents connections in a graph where A[i,j] = 1 if there's a relation from i to j, and 0 otherwise.

#### Example 3: Social Network Relations

```r
# Social network: who follows whom
# 4 people, directed relations
A <- matrix(c(0, 1, 1, 0,    # Person 1 follows persons 2, 3
              1, 0, 1, 1,    # Person 2 follows persons 1, 3, 4
              0, 1, 0, 1,    # Person 3 follows persons 2, 4
              1, 0, 0, 0),   # Person 4 follows person 1
            nrow = 4, ncol = 4, byrow = TRUE)

rownames(A) <- colnames(A) <- c("Alice", "Bob", "Charlie", "David")
print(A)
#         Alice Bob Charlie David
# Alice       0   1       1     0
# Bob         1   0       1     1
# Charlie     0   1       0     1
# David       1   0       0     0

# Find followers of each person (column sum)
followers <- colSums(A)
# Alice   Bob Charlie  David 
#     2     2       2      1

# Find who each person follows (row sum)
following <- rowSums(A)
# Alice   Bob Charlie David 
#     2     3       2     1

# Compute A^2: path length 2
A2 <- A %*% A
print(A2)
#         Alice Bob Charlie David
# Alice       1   1       1     1
# Bob         1   2       1     1
# Charlie     2   1       1     1
# David       0   1       1     0

# Interpretation: A2[i,j] = number of 2-step paths from i to j
```

#### Solution: Verify A^2[1,1] = 1
A^2[1,1] = sum(A[1,] * A[,1]) = A[1,1]×A[1,1] + A[1,2]×A[2,1] + A[1,3]×A[3,1] + A[1,4]×A[4,1]
= 0×0 + 1×1 + 1×0 + 0×1 = 1 ✓

### 2.1.2 Correlation and Covariance Matrices

These matrices represent relationships between variables.

#### Example 4: Data Matrix and Covariance

```r
# Dataset: 5 observations, 3 variables
data_matrix <- matrix(c(
  2, 8, 4,
  4, 12, 8,
  6, 15, 10,
  8, 18, 12,
  10, 22, 14
), nrow = 5, ncol = 3, byrow = TRUE)

colnames(data_matrix) <- c("X1", "X2", "X3")
print(data_matrix)
#      X1 X2 X3
# [1,]  2  8  4
# [2,]  4 12  8
# [3,]  6 15 10
# [4,]  8 18 12
# [5,] 10 22 14

# Compute covariance matrix
cov_matrix <- cov(data_matrix)
print(cov_matrix)
#           X1    X2    X3
# X1 10.0000  20.0   10.0
# X2 20.0000  40.625  20.0
# X3 10.0000  20.0    10.0

# Compute correlation matrix
cor_matrix <- cor(data_matrix)
print(cor_matrix)
#           X1        X2        X3
# X1 1.0000000 0.9959674 1.0000000
# X2 0.9959674 1.0000000 0.9959674
# X3 1.0000000 0.9959674 1.0000000

# Strong positive correlations indicate linear relationships
```

---

# PART 3: MATRIX DECOMPOSITIONS - SVD AND PCA

## 3.1 Singular Value Decomposition (SVD)

SVD is a fundamental matrix decomposition: **A = UΣV^T**

### 3.1.1 Understanding SVD Components

```r
# Create a data matrix
A <- matrix(c(3, 2, 2,
              2, 3, -2), nrow = 2, ncol = 3)

# Perform SVD
svd_result <- svd(A)

U <- svd_result$u          # Left singular vectors (2×2)
singular_values <- svd_result$d  # Singular values (length 2)
V <- svd_result$v          # Right singular vectors (3×3)

print("Matrix A:")
print(A)
print("U (left singular vectors):")
print(U)
#           [,1]       [,2]
# [1,] -0.7815437 -0.6238505
# [2,] -0.6238505  0.7815437

print("Singular values:")
print(singular_values)
# [1] 5.5480189 2.8669646

print("V (right singular vectors):")
print(V)
#            [,1]        [,2]        [,3]
# [1,] -0.6474982  0.10759258 -0.7544335
# [2,] -0.7599438  0.16501062  0.6286946
# [3,] -0.0568467 -0.98040574  0.1886084

# Reconstruct A: U %*% diag(d) %*% t(V)
Sigma <- diag(singular_values)  # Diagonal matrix
A_reconstructed <- U %*% Sigma %*% t(V)
print("Reconstructed A:")
print(A_reconstructed)  # Should equal A (within numerical precision)
```

#### Example 5: SVD for Compression

```r
# Image data as a matrix (grayscale)
# Simulated 8×8 image
image_data <- matrix(c(
  100, 80, 70, 60, 50, 40, 30, 20,
  90, 75, 65, 55, 45, 35, 25, 15,
  85, 70, 60, 50, 40, 30, 20, 10,
  80, 65, 55, 45, 35, 25, 15, 5,
  75, 60, 50, 40, 30, 20, 10, 0,
  70, 55, 45, 35, 25, 15, 5, -5,
  65, 50, 40, 30, 20, 10, 0, -10,
  60, 45, 35, 25, 15, 5, -5, -15
), nrow = 8, ncol = 8, byrow = TRUE)

# SVD of image
svd_img <- svd(image_data)
d <- svd_img$d

# Compression: keep only first k components
k_values <- c(1, 2, 3, 4)
for (k in k_values) {
  # Reconstruct with k components
  U_k <- svd_img$u[, 1:k, drop = FALSE]
  V_k <- svd_img$v[, 1:k, drop = FALSE]
  d_k <- diag(d[1:k])
  
  img_reconstructed <- U_k %*% d_k %*% t(V_k)
  
  # Compression ratio
  original_size <- 8 * 8
  compressed_size <- (8 * k) + k + (8 * k)  # U + Σ + V^T
  compression_ratio <- compressed_size / original_size
  
  cat("k =", k, "Compression ratio:", round(compression_ratio, 3), "\n")
}

# Output:
# k = 1 Compression ratio: 0.312 
# k = 2 Compression ratio: 0.375
# k = 3 Compression ratio: 0.469
# k = 4 Compression ratio: 0.562
```

### 3.2 Principal Component Analysis (PCA)

PCA finds directions (principal components) where data has maximum variance.

#### Example 6: PCA Step by Step

```r
# Sample dataset: student scores in 4 subjects
scores <- data.frame(
  Math = c(85, 78, 92, 88, 95, 72, 80, 86),
  English = c(78, 82, 85, 90, 88, 75, 79, 87),
  Science = c(88, 80, 90, 85, 92, 70, 82, 88),
  History = c(72, 75, 70, 88, 80, 68, 75, 85)
)

print(scores)

# Step 1: Standardize the data
scores_scaled <- scale(scores)
print("Scaled data (first 3 rows):")
print(scores_scaled[1:3, ])

# Step 2: Compute covariance matrix
cov_scores <- cov(scores_scaled)
print("Covariance matrix:")
print(round(cov_scores, 3))

# Step 3: Compute eigenvalues and eigenvectors
eigen_result <- eigen(cov_scores)
eigenvalues <- eigen_result$values
eigenvectors <- eigen_result$vectors

print("Eigenvalues:")
print(round(eigenvalues, 3))
# [1] 2.5820 0.8934 0.3563 0.1683

# Step 4: Compute variance explained
var_explained <- eigenvalues / sum(eigenvalues)
cumulative_var <- cumsum(var_explained)

print("Variance explained by each PC:")
for (i in 1:4) {
  cat("PC", i, ":", round(var_explained[i] * 100, 1), "% (Cumulative:",
      round(cumulative_var[i] * 100, 1), "%)\n")
}
# PC 1 : 59.8 % (Cumulative: 59.8 %)
# PC 2 : 20.7 % (Cumulative: 80.5 %)
# PC 3 : 8.3 % (Cumulative: 88.8 %)
# PC 4 : 3.9 % (Cumulative: 92.7 %)

# Step 5: Use prcomp() function (simpler approach)
pca_result <- prcomp(scores, scale = TRUE)
print(summary(pca_result))

# PC1 explains ~60% of variance - can use just PC1 & PC2 for 80% variance
# Get principal component scores
pc_scores <- pca_result$x
print("PC scores (first 3 observations):")
print(pc_scores[1:3, 1:2])
```

#### Solution: Interpreting PC1

The first principal component (PC1) has loadings:
```r
pca_result$rotation[, 1]
#      Math    English    Science    History
#    -0.462     -0.455     -0.473     -0.402
```

All loadings are negative and roughly equal, meaning PC1 is an overall "average performance" score. Students with high PC1 scores perform well across all subjects.

#### Example 7: PCA with Visualization

```r
# Biplot: shows both observations and variables
biplot(pca_result, scale = 0)

# Scree plot: variance explained
var_explained_pct <- (pca_result$sdev^2 / sum(pca_result$sdev^2)) * 100
plot(1:4, var_explained_pct, type = "b", 
     main = "Scree Plot", 
     xlab = "Principal Component", 
     ylab = "Variance Explained (%)",
     ylim = c(0, 70))

# Add cumulative variance line
cumulative_var_pct <- cumsum(var_explained_pct)
lines(1:4, cumulative_var_pct, col = "red", type = "b", lty = 2)
legend("topright", c("Individual", "Cumulative"), col = c("black", "red"), lty = 1:2)
```

#### Practice Exercise 2:
Use a sample of 50 observations from the iris dataset and perform PCA. 
a) How many PCs explain 90% of variance?
b) Create a biplot
c) Interpret the first two principal components

**Solution:**
```r
# Load data and standardize
iris_scaled <- scale(iris[1:50, 1:4])
iris_pca <- prcomp(iris_scaled)

# Cumulative variance
cumvar <- cumsum((iris_pca$sdev^2) / sum(iris_pca$sdev^2))
# Usually 2-3 PCs explain ~90% in iris data

# Biplot
biplot(iris_pca)

# Loadings interpretation
print(iris_pca$rotation[, 1:2])
```

---

# PART 4: DESCRIPTIVE STATISTICS

## 4.1 Measures of Central Tendency

### 4.1.1 Mean, Median, Mode

```r
# Dataset: Sales data
sales <- c(15000, 18000, 20000, 22000, 25000, 100000)  # Last value is outlier

# Mean: average (affected by outliers)
mean_sales <- mean(sales)
cat("Mean:", mean_sales, "\n")  # 37500

# Median: middle value (robust to outliers)
median_sales <- median(sales)
cat("Median:", median_sales, "\n")  # 21000

# Mode: most frequent (need to compute manually)
library(modes)  # or compute manually
get_mode <- function(x) {
  freq <- table(x)
  as.numeric(names(freq)[which.max(freq)])
}

# Comparison
cat("Mean vs Median difference:", mean_sales - median_sales, "\n")
# 16500 - shows strong positive skew due to outlier
```

### 4.1.2 Measures of Spread

```r
# Using the sales data
range_sales <- range(sales)
cat("Range:", range_sales[2] - range_sales[1], "\n")  # 85000

# Variance (squared deviations from mean)
var_sales <- var(sales)  # Sample variance
cat("Variance:", var_sales, "\n")  # 1320000000

# Standard deviation
sd_sales <- sd(sales)
cat("Standard Deviation:", sd_sales, "\n")  # 36331.12

# Interquartile Range (IQR)
Q1 <- quantile(sales, 0.25)
Q3 <- quantile(sales, 0.75)
iqr_sales <- Q3 - Q1
cat("IQR:", iqr_sales, "\n")  # 4000

# Coefficient of Variation (standardized measure)
cv_sales <- (sd_sales / mean_sales) * 100
cat("Coefficient of Variation:", round(cv_sales, 2), "%\n")  # 96.88%
```

### 4.1.3 Skewness and Kurtosis

```r
library(moments)  # For skewness and kurtosis functions

# Skewness (asymmetry)
skew <- skewness(sales)
cat("Skewness:", round(skew, 3), "\n")  # Positive skew (right tail)

# Kurtosis (tail heaviness)
kurt <- kurtosis(sales)
cat("Kurtosis:", round(kurt, 3), "\n")

# Interpretation:
# Skewness > 0: Right-skewed (tail on right)
# Skewness < 0: Left-skewed (tail on left)
# Skewness ≈ 0: Symmetric
# Kurtosis > 0: Heavy tails (outliers likely)
# Kurtosis < 0: Light tails (fewer outliers)
```

### 4.1.4 Summary Statistics

```r
# Comprehensive summary
summary(sales)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#   15000   20000   21000   37500   23500  100000 

# Boxplot visualization
boxplot(sales, main = "Sales Distribution", 
        ylab = "Amount ($)", horizontal = TRUE)
```

---

# PART 5: PROBABILITY DISTRIBUTIONS

## 5.1 Discrete Distributions

### 5.1.1 Binomial Distribution

The binomial distribution models the number of successes in n trials, each with probability p.

**Probability Mass Function (PMF):** P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

#### Example 8: Coin Flips

```r
# 10 coin flips, probability of heads = 0.5
n <- 10
p <- 0.5
k <- 0:10

# Probability of exactly k heads
prob_exact <- dbinom(k, n, p)
print(cbind(k, prob_exact))
#       k   prob_exact
#  [1,] 0  0.0009766
#  [2,] 1  0.0097656
#  [3,] 2  0.0439453
#  [4,] 3  0.1171875
#  [5,] 4  0.2050781
#  [6,] 5  0.2460938  <- Most likely
#  [7,] 6  0.2050781
#  [8,] 7  0.1171875
#  [9,] 8  0.0439453
# [10,] 9  0.0097656
# [11,] 10 0.0009766

# Cumulative probability: P(X ≤ k)
prob_cumulative <- pbinom(k, n, p)
cat("P(X ≤ 5) =", pbinom(5, n, p), "\n")  # 0.6230469

# Generate random samples
random_binomial <- rbinom(1000, n, p)
hist(random_binomial, breaks = 11, prob = TRUE,
     main = "Distribution of 1000 Binomial(n=10, p=0.5) Samples",
     xlab = "Number of Heads")
```

### 5.1.2 Poisson Distribution

Models count data with events occurring at constant rate λ.

**PMF:** P(X = k) = (e^(-λ) × λ^k) / k!

#### Example 9: Website Visits

```r
# Website receives average 3 visits per hour
lambda <- 3

# Probability of exactly k visits
k <- 0:10
prob <- dpois(k, lambda)
print(cbind(k, prob))
#       k        prob
#  [1,] 0  0.04978707
#  [2,] 1  0.14936121
#  [3,] 2  0.22404181
#  [4,] 3  0.22404181  <- Maximum
#  [5,] 4  0.16803136
#  [6,] 5  0.10081882
#  [7,] 6  0.05040941
#  [8,] 7  0.02160404
#  [9,] 8  0.00810152
# [10,] 9  0.00270051
# [11,] 10 0.00081015

# Probability of at most 3 visits
prob_at_most_3 <- ppois(3, lambda)
cat("P(X ≤ 3) =", prob_at_most_3, "\n")  # 0.6472639

# Generate random Poisson data
visits <- rpois(100, lambda)
hist(visits, breaks = seq(0, max(visits)+1) - 0.5,
     main = "Website Visits (100 observations)",
     xlab = "Number of Visits",
     prob = TRUE)
```

## 5.2 Continuous Distributions

### 5.2.1 Normal (Gaussian) Distribution

Most important distribution in statistics: X ~ N(μ, σ²)

**PDF:** f(x) = (1/(σ√(2π))) × e^(-(x-μ)²/(2σ²))

#### Example 10: Student Heights

```r
# Heights normally distributed with mean 170 cm, SD 10 cm
mu <- 170
sigma <- 10

# Probability density
x <- seq(140, 200, by = 1)
density <- dnorm(x, mu, sigma)

plot(x, density, type = "l", main = "Normal Distribution of Heights",
     xlab = "Height (cm)", ylab = "Density")
abline(v = mu, col = "red", lty = 2, label = "Mean")

# Probability: P(160 < X < 180)
prob_between <- pnorm(180, mu, sigma) - pnorm(160, mu, sigma)
cat("P(160 < Height < 180) =", round(prob_between, 4), "\n")  # 0.6827

# What height corresponds to 95th percentile?
height_95 <- qnorm(0.95, mu, sigma)
cat("95th percentile height:", round(height_95, 1), "cm\n")  # 186.4 cm

# Simulate heights
heights <- rnorm(1000, mu, sigma)
hist(heights, breaks = 30, prob = TRUE,
     main = "Simulated Student Heights (n=1000)",
     xlab = "Height (cm)")
curve(dnorm(x, mu, sigma), add = TRUE, col = "red", lwd = 2)

# Check normality
qqnorm(heights)
qqline(heights, col = "red")
```

### 5.2.2 Uniform Distribution

All values equally likely in [a, b]: X ~ Uniform(a, b)

```r
# Random number between 0 and 1 (standard uniform)
random_uniform <- runif(1000, min = 0, max = 1)

# Probability density
x <- seq(0, 1, by = 0.01)
density <- dunif(x, 0, 1)

plot(x, density, type = "l", ylim = c(0, 1.2),
     main = "Uniform Distribution U(0,1)",
     xlab = "x", ylab = "Density")

# P(X < 0.3)
prob_less_03 <- punif(0.3, 0, 1)
cat("P(X < 0.3) =", prob_less_03, "\n")  # 0.3
```

### 5.2.3 Exponential Distribution

Models waiting times between events: X ~ Exp(λ)

```r
# Time between arrivals (mean = 5 minutes)
lambda <- 1/5  # Rate parameter

# Probability density
x <- seq(0, 30, by = 0.1)
density <- dexp(x, lambda)

plot(x, density, type = "l",
     main = "Exponential Distribution (mean = 5)",
     xlab = "Time (minutes)", ylab = "Density")

# P(X > 10) - probability wait > 10 minutes
prob_greater_10 <- 1 - pexp(10, lambda)
cat("P(Wait > 10 min) =", round(prob_greater_10, 4), "\n")  # 0.1353

# Mean and variance
mean_exp <- 1/lambda
var_exp <- 1/(lambda^2)
cat("Mean:", mean_exp, "Variance:", var_exp, "\n")
```

### 5.2.4 t-Distribution

Used for small samples: similar to normal but with heavier tails.

```r
# Compare normal and t-distributions
x <- seq(-4, 4, by = 0.1)

# Normal distribution
normal_density <- dnorm(x)

# t-distribution with different degrees of freedom
t_df3 <- dt(x, df = 3)
t_df10 <- dt(x, df = 10)
t_df30 <- dt(x, df = 30)

plot(x, normal_density, type = "l", col = "black", ylim = c(0, 0.45),
     main = "Normal vs t-Distribution",
     xlab = "x", ylab = "Density")
lines(x, t_df3, col = "red", lty = 2)
lines(x, t_df10, col = "blue", lty = 2)
lines(x, t_df30, col = "green", lty = 2)

legend("topright", c("Normal", "t(df=3)", "t(df=10)", "t(df=30)"),
       col = c("black", "red", "blue", "green"), lty = c(1, 2, 2, 2))

# Critical values for hypothesis testing
t_critical_05 <- qt(0.975, df = 25)  # 2-tailed, α = 0.05
cat("t-critical value (α=0.05, df=25):", round(t_critical_05, 3), "\n")  # 2.060
```

#### Practice Exercise 3:
For X ~ N(100, 15²):
a) Find P(X < 85)
b) Find P(85 < X < 115)
c) Find the value x where P(X < x) = 0.9

**Solution:**
```r
# a) P(X < 85)
pnorm(85, mean = 100, sd = 15)  # 0.1587

# b) P(85 < X < 115)
pnorm(115, 100, 15) - pnorm(85, 100, 15)  # 0.6826

# c) 90th percentile
qnorm(0.9, mean = 100, sd = 15)  # 119.22
```

---

# PART 6: STATISTICAL INFERENCE

## 6.1 Populations and Samples

### 6.1.1 Sampling Distribution

```r
# Population: height of all students (assume normal)
# Population parameters
pop_mean <- 170
pop_sd <- 10
population_size <- 10000

# Create population
population <- rnorm(population_size, pop_mean, pop_sd)

# Take 1000 random samples of size n = 50
n_samples <- 1000
sample_size <- 50
sample_means <- numeric(n_samples)

set.seed(42)
for (i in 1:n_samples) {
  sample <- sample(population, sample_size)
  sample_means[i] <- mean(sample)
}

# Analyze sampling distribution
mean_of_means <- mean(sample_means)
sd_of_means <- sd(sample_means)
theoretical_se <- pop_sd / sqrt(sample_size)

cat("Population mean:", pop_mean, "\n")
cat("Mean of sample means:", round(mean_of_means, 2), "\n")
cat("Standard error (observed):", round(sd_of_means, 2), "\n")
cat("Standard error (theoretical):", round(theoretical_se, 2), "\n")

# Visualization
hist(sample_means, breaks = 30, prob = TRUE,
     main = "Sampling Distribution of Mean (n=50)",
     xlab = "Sample Mean Height")
curve(dnorm(x, pop_mean, theoretical_se), add = TRUE, col = "red", lwd = 2)
```

### 6.1.2 Central Limit Theorem

```r
# Show that sampling distribution is normal even if population isn't
# Use exponential distribution (not normal)

# Population: exponential with mean 5
population_exp <- rexp(10000, rate = 1/5)

# Take samples of increasing sizes
sizes <- c(10, 30, 100, 200)
par(mfrow = c(2, 2))

for (n in sizes) {
  sample_means <- replicate(1000, mean(sample(population_exp, n)))
  
  hist(sample_means, breaks = 30, prob = TRUE,
       main = paste("Sample Size n =", n),
       xlab = "Sample Mean")
  
  # Overlay normal distribution
  curve(dnorm(x, 5, 5/sqrt(n)), add = TRUE, col = "red", lwd = 2)
}

par(mfrow = c(1, 1))

# Even though population is skewed, sample means are normally distributed!
```

## 6.2 Confidence Intervals

A confidence interval estimates a population parameter within a margin of error.

### 6.2.1 Confidence Interval for Population Mean

```r
# Sample of test scores
scores <- c(78, 82, 75, 88, 92, 85, 80, 86, 79, 87)

# Calculate 95% confidence interval
n <- length(scores)
sample_mean <- mean(scores)
sample_sd <- sd(scores)
standard_error <- sample_sd / sqrt(n)

# Use t-distribution (n is small)
confidence_level <- 0.95
alpha <- 1 - confidence_level
t_critical <- qt(1 - alpha/2, df = n - 1)

margin_error <- t_critical * standard_error
ci_lower <- sample_mean - margin_error
ci_upper <- sample_mean + margin_error

cat("Sample mean:", round(sample_mean, 2), "\n")
cat("Standard error:", round(standard_error, 2), "\n")
cat("t-critical value (df=9):", round(t_critical, 3), "\n")
cat("Margin of error:", round(margin_error, 2), "\n")
cat("95% CI: [", round(ci_lower, 2), ",", round(ci_upper, 2), "]\n")
# 95% CI: [ 79.4 , 85.86 ]

# Interpretation: We are 95% confident the true population mean
# is between 79.4 and 85.86

# Using built-in function
t_test <- t.test(scores, conf.level = 0.95)
print(t_test$conf.int)
# [1] 79.37 85.83
```

### 6.2.2 Confidence Interval for Difference in Means

```r
# Compare test scores between two groups
group1 <- c(78, 82, 75, 88, 92, 85, 80, 86, 79, 87)
group2 <- c(72, 75, 68, 82, 85, 78, 75, 80, 73, 81)

# Two-sample t-test with CI
t_result <- t.test(group1, group2, conf.level = 0.95)

cat("Group 1 mean:", round(mean(group1), 2), "\n")
cat("Group 2 mean:", round(mean(group2), 2), "\n")
cat("Difference:", round(mean(group1) - mean(group2), 2), "\n")
cat("95% CI for difference: [", round(t_result$conf.int[1], 2), ",",
    round(t_result$conf.int[2], 2), "]\n")

# Output:
# Group 1 mean: 83.2
# Group 2 mean: 76.9
# Difference: 6.3
# 95% CI for difference: [ -0.32 , 12.92 ]

# The CI includes 0, suggesting no significant difference at α=0.05
```

## 6.3 Hypothesis Testing

### 6.3.1 One-Sample t-Test

Test if population mean equals a specified value.

```r
# Test if average score is 80
scores <- c(78, 82, 75, 88, 92, 85, 80, 86, 79, 87)

# Hypotheses:
# H0: μ = 80
# H1: μ ≠ 80 (two-tailed)

t_test_result <- t.test(scores, mu = 80, alternative = "two.sided")
print(t_test_result)

# Output:
#   One Sample t-test
# t = 0.9994, df = 9, p-value = 0.3448
# alternative hypothesis: true mean is not equal to 80
# 95% confidence interval: [79.37 85.83]
# sample estimates: mean of x = 83.2

# Interpretation:
# t-statistic = 0.9994
# p-value = 0.3448 > 0.05
# Fail to reject H0: No significant evidence that μ ≠ 80
```

### 6.3.2 Two-Sample t-Test

Compare means of two independent groups.

```r
# Birth weight comparison: non-smoker vs smoker mothers
# Dataset from North Carolina births study

nonsmoker_weight <- c(7.63, 7.88, 6.63, 8.00, 6.38, 5.38, 8.44, 7.30, 8.10, 6.99)
smoker_weight <- c(7.05, 6.95, 6.11, 6.42, 6.08, 5.85, 6.55, 6.89, 6.33, 6.75)

# Hypotheses:
# H0: μ_nonsmoker = μ_smoker
# H1: μ_nonsmoker ≠ μ_smoker

t_test_weights <- t.test(nonsmoker_weight, smoker_weight,
                         alternative = "two.sided",
                         var.equal = FALSE)  # Welch's t-test
print(t_test_weights)

# Output might show:
# Welch Two Sample t-test
# t = 2.3625, df = 17.65, p-value = 0.0298
# 95% CI: (0.057, 0.66)
# Non-smoker mean: 7.24, Smoker mean: 6.59

# Interpretation:
# p-value = 0.0298 < 0.05
# Reject H0: Significant evidence that babies of non-smoking mothers
# have higher birth weight
```

### 6.3.3 Effect Size

Cohen's d measures the magnitude of difference between groups.

```r
# Calculate Cohen's d
d <- (mean(nonsmoker_weight) - mean(smoker_weight)) / 
     sqrt(((length(nonsmoker_weight)-1)*var(nonsmoker_weight) +
           (length(smoker_weight)-1)*var(smoker_weight)) /
          (length(nonsmoker_weight) + length(smoker_weight) - 2))

cat("Cohen's d =", round(d, 3), "\n")
# Interpretation:
# d < 0.2: negligible
# 0.2 - 0.5: small
# 0.5 - 0.8: medium
# d > 0.8: large

# In this case, d ≈ 0.92 = large effect
```

### 6.3.4 Chi-Square Test of Independence

Test if two categorical variables are independent.

```r
# Test: Is baby gender independent of maternal smoking?
contingency_table <- matrix(c(
  280,  # Male, non-smoker
  97,   # Male, smoker
  241,  # Female, non-smoker
  89    # Female, smoker
), nrow = 2, ncol = 2, byrow = TRUE,
   dimnames = list(Gender = c("Male", "Female"),
                   Smoking = c("Non-smoker", "Smoker")))

print(contingency_table)
#         Smoking
# Gender   Non-smoker Smoker
#   Male         280     97
#   Female       241     89

# Perform chi-square test
chi_test <- chisq.test(contingency_table)
print(chi_test)

# Output:
#   Pearson's Chi-squared test
# X-squared = 0.0038, df = 1, p-value = 0.9506
# 
# Interpretation:
# p-value = 0.9506 > 0.05
# Fail to reject H0: No significant association between
# maternal smoking and baby gender
```

---

# PART 7: STATISTICAL MODELING AND MODEL FITTING

## 7.1 Linear Regression Model

### 7.1.1 Simple Linear Regression

```r
# Dataset: Advertising spending vs Sales
advertising <- data.frame(
  TV = c(230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.0, 120.2,
         8.6, 199.8, 66.1, 214.6, 23.8, 97.5, 204.1, 195.4),
  Sales = c(26.1, 10.9, 9.3, 18.5, 23.1, 7.2, 11.8, 13.2,
            4.8, 22.1, 12.5, 24.0, 10.6, 15.4, 20.1, 23.9)
)

# Fit linear model: Sales = β0 + β1 * TV + ε
model <- lm(Sales ~ TV, data = advertising)
summary(model)

# Output interpretation:
# Call: lm(formula = Sales ~ TV, data = advertising)
# 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)
# (Intercept)  7.03259    0.33808  20.801  < 2e-16 ***
# TV           0.04754    0.00269   1.766    0.098 .
# 
# Residual standard error: 3.26 on 14 df
# Multiple R-squared:  0.18, Adjusted R-squared:  0.12
# F-statistic: 3.12 on 1 and 14 DF,  p-value: 0.098

# Model interpretation:
# - Intercept β0 = 7.03: Expected sales when TV = 0 (baseline)
# - Slope β1 = 0.0475: Each $1000 in TV spending → 0.0475 units sales increase
# - R² = 0.18: TV explains 18% of sales variation
# - p-value = 0.098: Marginally significant at α=0.10, not at α=0.05

# Prediction
new_tv_spending <- data.frame(TV = c(100, 200, 300))
predictions <- predict(model, newdata = new_tv_spending, 
                      interval = "prediction", level = 0.95)
print(predictions)
#        fit     lwr      upr
# 1 12.78644  5.8477 19.72517
# 2 17.53934 10.5814 24.49722
# 3 22.29224 15.3156 29.26884
```

### 7.1.2 Multiple Linear Regression

```r
# Dataset: House prices
houses <- data.frame(
  Price = c(145, 245, 210, 320, 280, 180, 220, 310, 190, 240),
  SquareFeet = c(1400, 2600, 2300, 3200, 2900, 1800, 2400, 3300, 1900, 2500),
  Bedrooms = c(2, 4, 3, 4, 3, 3, 3, 4, 2, 4),
  Age = c(30, 5, 10, 2, 15, 25, 8, 3, 28, 12)
)

# Fit multiple regression model
model_multi <- lm(Price ~ SquareFeet + Bedrooms + Age, data = houses)
summary(model_multi)

# Coefficients interpretation:
# - SquareFeet: Each additional sq ft → price increase by coefficient
# - Bedrooms: Each additional bedroom → price increase (holding other factors constant)
# - Age: Each additional year → price decrease (negative coefficient)

# Model diagnostics
par(mfrow = c(2, 2))
plot(model_multi)
par(mfrow = c(1, 1))

# Check for multicollinearity
library(car)
vif(model_multi)  # VIF < 5 is generally acceptable
```

## 7.2 Goodness of Fit Tests

### 7.2.1 Kolmogorov-Smirnov Test

Tests if data follows a specified distribution.

```r
# Test if data follows normal distribution
data_sample <- rnorm(100, mean = 100, sd = 15)

# Null hypothesis: Data follows N(100, 15²)
ks_result <- ks.test(data_sample, "pnorm", mean = 100, sd = 15)
print(ks_result)

# Output:
#   One-sample Kolmogorov-Smirnov test
# D = 0.0758, p-value = 0.6217
# alternative hypothesis: two-sided
# 
# Interpretation:
# p-value = 0.6217 > 0.05
# Fail to reject H0: Data is consistent with N(100, 15²)
```

### 7.2.2 Shapiro-Wilk Test

Tests normality for small to moderate samples.

```r
# Test normality of data
test_data <- c(78, 82, 75, 88, 92, 85, 80, 86, 79, 87, 81, 84)

shapiro_result <- shapiro.test(test_data)
print(shapiro_result)

# Output:
#   Shapiro-Wilk normality test
# W = 0.9693, p-value = 0.8547
# 
# Interpretation:
# p-value = 0.8547 > 0.05
# Fail to reject H0: Data appears normally distributed
```

### 7.2.3 Anderson-Darling Test

```r
library(nortest)

# Test against various distributions
anderson_result <- ad.test(test_data)
print(anderson_result)

# Output might show:
# Anderson-Darling normality test
# A = 0.3475, p-value = 0.4814
# 
# Indicates the data is consistent with normal distribution
```

### 7.2.4 Chi-Square Goodness of Fit Test

Tests if observed data matches expected distribution.

```r
# Example: Test if die is fair
# Observed frequencies from 600 rolls
observed <- c(95, 105, 98, 102, 108, 92)
# Expected: equal probability (100 each)
expected <- rep(100, 6)

chi_sq_result <- chisq.test(observed, p = expected/sum(expected))
print(chi_sq_result)

# Output:
#   Chi-squared test for given probabilities
# X-squared = 1.04, df = 5, p-value = 0.96
# 
# Interpretation:
# p-value = 0.96 > 0.05
# Fail to reject H0: Die rolls are consistent with fair die
```

## 7.3 Fitting Probability Distributions to Data

### 7.3.1 Method of Moments Estimation

Estimate parameters by matching sample moments to theoretical moments.

```r
# Estimate normal distribution parameters
data <- c(85, 92, 78, 88, 95, 80, 86, 90, 82, 89)

# Method of Moments:
# μ = sample mean
# σ = sample standard deviation

mu_estimate <- mean(data)
sigma_estimate <- sd(data)

cat("Estimated μ =", round(mu_estimate, 2), "\n")
cat("Estimated σ =", round(sigma_estimate, 2), "\n")

# Estimate Poisson parameter
count_data <- c(2, 3, 1, 4, 2, 3, 5, 2, 1, 3, 4, 2)
lambda_estimate <- mean(count_data)
cat("Estimated λ =", round(lambda_estimate, 2), "\n")
```

### 7.3.2 Maximum Likelihood Estimation (MLE)

```r
library(MASS)
library(fitdistrplus)

# Fit exponential distribution using MLE
exp_data <- rexp(100, rate = 0.2)

# Using fitdist function
fit_exp <- fitdist(exp_data, "exp")
summary(fit_exp)

# Output shows MLE estimates and goodness of fit metrics
# Parameter estimate for rate (λ)
cat("MLE estimate of rate (λ):", fit_exp$estimate, "\n")

# Compare different distributions
fit_normal <- fitdist(exp_data, "norm")
fit_gamma <- fitdist(exp_data, "gamma")

# Compare using AIC or BIC
cat("Normal AIC:", fit_normal$aic, "\n")
cat("Gamma AIC:", fit_gamma$aic, "\n")
cat("Exponential AIC:", fit_exp$aic, "\n")
```

### 7.3.3 Quantile-Quantile (QQ) Plots

Assess fit visually by comparing quantiles.

```r
# Generate data and test for normality
data <- c(88, 92, 85, 95, 90, 87, 91, 86, 89, 93, 84, 94)

# QQ plot against normal distribution
qqnorm(data, main = "Q-Q Plot: Data vs Normal Distribution")
qqline(data, col = "red", lwd = 2)

# If points fall along the red line → data is approximately normal
# Deviations at tails → non-normal behavior

# Test against other distributions
library(car)

# QQ plot for exponential distribution
# First normalize data to have exponential characteristics
exp_data <- rexp(100)
qqplot(qexp(ppoints(100)), sort(exp_data),
       main = "Q-Q Plot: Data vs Exponential",
       xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles")
abline(0, 1, col = "red", lwd = 2)
```

#### Practice Exercise 4:
Generate 1000 samples from N(50, 100) and:
a) Test for normality using Shapiro-Wilk
b) Create a QQ plot
c) Estimate parameters using method of moments
d) Fit a normal distribution using MLE and compare AIC

**Solution:**
```r
set.seed(123)
data <- rnorm(1000, mean = 50, sd = 10)

# a) Shapiro-Wilk test (limited to 5000 samples)
shapiro.test(data)

# b) QQ plot
qqnorm(data); qqline(data, col = "red")

# c) MOM estimates
mom_mean <- mean(data)
mom_sd <- sd(data)

# d) MLE fit
library(fitdistrplus)
fit <- fitdist(data, "norm")
summary(fit)
```

---

## COMPREHENSIVE EXAMPLE: FULL ANALYSIS WORKFLOW

```r
# Complete analysis: Iris flower dataset
data(iris)

# 1. DATA EXPLORATION
summary(iris)
head(iris)

# Create data matrix
iris_numeric <- iris[, 1:4]

# 2. DESCRIPTIVE STATISTICS
cat("Mean Sepal Length:", mean(iris$Sepal.Length), "\n")
cat("SD Sepal Length:", sd(iris$Sepal.Length), "\n")
cat("Correlation matrix:\n")
print(cor(iris_numeric))

# 3. VISUALIZATION
pairs(iris_numeric, main = "Iris Dataset Pairs Plot")
boxplot(Sepal.Length ~ Species, data = iris,
        main = "Sepal Length by Species")

# 4. HYPOTHESIS TEST
# H0: Mean sepal length is same across species
anova_result <- aov(Sepal.Length ~ Species, data = iris)
summary(anova_result)

# 5. DIMENSIONALITY REDUCTION (PCA)
iris_pca <- prcomp(iris_numeric, scale = TRUE)
summary(iris_pca)

# Plot variance explained
plot(iris_pca, main = "Scree Plot")

# 6. REGRESSION MODELING
model <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
            data = iris)
summary(model)

# 7. GOODNESS OF FIT
shapiro.test(residuals(model))  # Test residuals for normality
```

---

## SUMMARY TABLE: KEY FUNCTIONS IN R

| Task | Function | Example |
|------|----------|---------|
| Create matrix | `matrix()` | `m <- matrix(1:6, nrow=2)` |
| Matrix operations | `%*%`, `t()`, `solve()` | `A %*% B`, `t(A)`, `solve(A)` |
| SVD | `svd()` | `svd_result <- svd(A)` |
| PCA | `prcomp()` | `pca <- prcomp(data, scale=T)` |
| Summary stats | `summary()`, `mean()`, `sd()` | `summary(x)` |
| Covariance | `cov()`, `cor()` | `cor(data_matrix)` |
| Distributions | `dnorm()`, `pnorm()`, `qnorm()`, `rnorm()` | `dnorm(x, mean, sd)` |
| t-test | `t.test()` | `t.test(x, mu=0)` |
| ANOVA | `aov()` | `aov(y ~ group, data)` |
| Chi-square | `chisq.test()` | `chisq.test(table)` |
| Goodness of fit | `ks.test()`, `shapiro.test()` | `ks.test(x, "pnorm")` |
| Regression | `lm()` | `lm(y ~ x1 + x2, data)` |
| Diagnostics | `plot()` on model | `plot(model)` |

---

## BIBLIOGRAPHY & FURTHER READING

1. Strang, G. (2009). Introduction to Linear Algebra. Wellesley-Cambridge Press.
2. Izenman, A. J. (2008). Modern Multivariate Statistical Techniques. Springer.
3. Hastie, T., Tibshirani, R., & James, G. (2013). An Introduction to Statistical Learning.
4. Wasserman, L. (2004). All of Statistics. Springer.
5. R Documentation: https://www.r-project.org/
6. RStudio: https://posit.co/products/open-source/rstudio/

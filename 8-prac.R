# video : https://youtu.be/5vgP05YpKdE?si=SvMj-BxFT5s0jd_0
# Load the necessary libraries
library(FactoMineR)
library(factoextra)
library(ggplot2)
library(nnet)

# Use the iris dataset
data <- iris
df <- as.data.frame(data)

# Assume the last column is the target variable
target <- df[,ncol(df)]
features <- df[,-ncol(df)]

# Train a multinomial model on raw data
# multinomial is used to predict diffrent categorical var using multiple datas
model_raw <- multinom(target ~ ., data = features)

# Get predictions
predictions_raw <- predict(model_raw, newdata = features)

# Perform PCA on the features
pca <- PCA(features, graph = TRUE, scale.unit = TRUE)
fviz_eig(pca,addlabels = TRUE)

# Get the PCA scores
#pca scores are score given to each data pt 
scores <- get_pca_ind(pca)$coord

# Train a multinomial model on PCA data
model_pca <- multinom(target ~ ., data = as.data.frame(scores))

# Get predictions
predictions_pca <- predict(model_pca, newdata = as.data.frame(scores))

# Create a data frame for plotting
plot_data <- data.frame(
  Target = rep(target, 2),
  Predictions = c(predictions_raw, predictions_pca),
  Data = rep(c("Raw", "PCA"), each = length(target))
)


# Plot the predictions
ggplot(plot_data, aes(x = Target, y = Predictions, color = Data)) +
  geom_jitter(width = 0.2, height = 0.2) +  # Add some jitter to the points
  geom_smooth(method = "lm") +
  labs(title = "Predictions on Raw Data vs PCA Data", x = "Target", y = "Predictions") +
  theme_minimal()













# Load the Iris dataset (assuming it's available)
data(iris)

# Perform PCA on the Iris data (excluding species for simplicity)
pca_results <- prcomp(iris[, 1:4])  # 1:4 selects the first 4 features

# Access the principal components
principal_components <- pca_results$x

# View the explained variance ratio per component
summary(pca_results)










# Load required libraries
library(ggplot2)

# Load the iris dataset
data(iris)

# Perform PCA
pca_result <- prcomp(iris[, -5], scale. = TRUE)  # Exclude the species column (column 5)
summary(pca_result)

# Extract principal components
pc_df <- as.data.frame(pca_result$x)  # Extract the first two principal components

# Add species column from original data set for coloring
pc_df$Species <- iris$Species

# Plot principal components using ggplot2
ggplot(pc_df, aes(PC1, PC2, color = Species)) +
  geom_point(alpha=0.6) +
  labs(title = "PCA Components Plot",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_bw()



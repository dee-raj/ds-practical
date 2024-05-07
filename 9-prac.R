# Load the required libraries
library(ggplot2)

# Define the wssplot function
wssplot <- function(data, max_k = 10) {
  wss <- sapply(1:max_k, function(k){
    kmeans(data, centers = k)$tot.withinss
  })
  plot(1:max_k, wss, type = "b", pch = 19, frame = FALSE, 
       xlab = "Number of Clusters (k)",
       ylab = "Within-Cluster Sum of Squares (WSS)",
       main = "WSS Plot for K-means Clustering")
}


# Load the iris dataset
data(iris)

# Selecting only numeric variables for clustering
iris_numeric <- iris[, 1:4]


# Plot the WSS plot
wssplot(iris_numeric, max_k=15)

# Perform k-means clustering with the optimal number of clusters
optimal_k <- 3  # Assuming the optimal number of clusters from the WSS plot
kmeans_result <- kmeans(iris_numeric, centers = optimal_k)


# Add cluster labels to the original data set
iris_clustered <- cbind(iris_numeric, Cluster = kmeans_result$cluster)

# Extract cluster centers for plotting
centroids <- kmeans_result$centers

# Plot the clusters with centroids
ggplot(iris_clustered, aes(y=Sepal.Length, x=Sepal.Width, color=factor(Cluster))) +
  geom_point() +
  geom_point(data = as.data.frame(centroids), aes(y = Sepal.Length, x = Sepal.Width),
             color = "black", shape = 17, size = 5) +  # Add centroids
  labs(title = "K-means Clustering of Iris Dataset",
       x = "Sepal Length",
       y = "Sepal Width",
       color = "Cluster")



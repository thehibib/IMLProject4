# -*- coding: utf-8 -*-


from PIL import Image
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from google.colab import files

# Load the image
image_path = "/content/egg.png"  
image = Image.open(image_path)


image = image.convert("RGB")


size = 3
width, height = image.size
s_width = width // size
s_height = height // size

# Resize the image to a lower resolution to make clustering faster
s_image = image.resize((s_width, s_height), resample=Image.BOX)

# Optional: Rescale back up for visualizing larger blocks (comment this out if not needed)
pix_image = s_image.resize((width, height), resample=Image.NEAREST)

# Extract the pixel data from the resized image
pixels = list(s_image.getdata())

# Create a DataFrame with the pixel data
pixel_df = pd.DataFrame(
    [(x % s_width, x // s_width, *rgb) for x, rgb in enumerate(pixels)],
    columns=["x", "y", "R", "G", "B"]
)

# #DOWNLOADS AND RECONSTRUCTS IMAGE
# pixel_array = np.zeros((s_height, s_width, 3), dtype=np.uint8)

# for _, row in pixel_df.iterrows():
#     x, y, r, g, b = row['x'], row['y'], row['R'], row['G'], row['B']
#     pixel_array[y, x] = [r, g, b]

# # Create an image from the array
# reconstructed_image = Image.fromarray(pixel_array)

# # Visualize the image
# reconstructed_image.show()

# # Save the image
# output_path = "reconstructed_image.png"
# reconstructed_image.save(output_path)
# files.download(output_path)

features = pixel_df[["R", "G", "B","x","y"]]
dbmodel = DBSCAN(eps=8, min_samples=1, metric="euclidean")
pixel_df["cluster"] = dbmodel.fit_predict(features)

# # Determine the number of clusters, excluding noise (-1)
n_clusters = len(set(pixel_df["cluster"])) - (1 if -1 in pixel_df["cluster"].values else 0)
print(n_clusters)

# # Generate random colors for each cluster
# cluster_colors = np.random.randint(0, 256, size=(n_clusters + 1, 3))  # +1 for noise
# cluster_colors[-1] = [0, 0, 0]  # Set noise to black
cluster_colors = (
    pixel_df.groupby("cluster")[["R", "G", "B"]]
    .mean()  # Compute mean RGB values
    .astype(int)  # Convert to integers for valid color values
    .reset_index()
)

# Map the clusters to their average colors
color_mapping = {
    row["cluster"]: (row["R"], row["G"], row["B"]) for _, row in cluster_colors.iterrows()
}

# Assign the average color to each pixel based on its cluster
pixel_df["color"] = pixel_df["cluster"].map(color_mapping)

# Map the clusters to colors in the DataFrame
#pixel_df["color"] = pixel_df["cluster"].apply(lambda c: cluster_colors[c] if c != -1 else [0, 0, 0])

# Create an empty pixel array for the clustered image
pixel_array = np.zeros((height, width, 3), dtype=np.uint8)

# Assign the colors to the corresponding pixel positions
for _, row in pixel_df.iterrows():
    x, y = row["x"]*size, row["y"]*size
    pixel_array[y:y+size, x:x+size] = row["color"]

# Create and visualize the clustered image
clustered_image = Image.fromarray(pixel_array)
plt.imshow(clustered_image)
plt.axis('off')
plt.show()

# Save the clustered image
clustered_image.save("/content/clustered_image.png")
pixel_df

from sklearn.cluster import KMeans

# Step 1: Compute cluster-level features
cluster_features = (
    pixel_df.groupby("cluster")
    .agg(
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        R_mean=("R", "mean"),
        G_mean=("G", "mean"),
        B_mean=("B", "mean"),
        size=("cluster", "size")
    )
    .reset_index()
)

# Prepare the feature matrix for K-Means
kmeans_features = cluster_features[["x_mean", "y_mean", "R_mean", "G_mean", "B_mean", "size"]]

# Step 2: Apply K-Means

if n_clusters/80 < 15:
  n_clusters = 15
else:
  n_clusters = int(n_clusters/80)
print(n_clusters)
kmeans_model = KMeans(n_clusters=int(n_clusters),init='k-means++',n_init=10)  # Change n_clusters as needed
cluster_features["kmeans_cluster"] = kmeans_model.fit_predict(kmeans_features)

# Step 3: Map higher-level K-Means labels back to pixel_df
# Create a mapping of original clusters to K-Means clusters
kmeans_mapping = dict(
    zip(cluster_features["cluster"], cluster_features["kmeans_cluster"])
)

# Assign higher-level clusters to pixels
pixel_df["kmeans_cluster"] = pixel_df["cluster"].map(kmeans_mapping)

# Step 5: Compute average color for K-Means clusters
kmeans_color_mapping = (
    pixel_df.groupby("kmeans_cluster")[["R", "G", "B"]]
    .mean()  # Compute mean RGB values for each higher-level cluster
    .astype(int)  # Convert to integers
    .reset_index()
)

# Create a mapping of K-Means clusters to their average colors
kmeans_avg_color_mapping = {
    row["kmeans_cluster"]: (row["R"], row["G"], row["B"]) for _, row in kmeans_color_mapping.iterrows()
}

# Assign the average color to each pixel based on its K-Means cluster
pixel_df["kmeans_color"] = pixel_df["kmeans_cluster"].map(kmeans_avg_color_mapping)

# Step 6: Create the final image with K-Means average colors
kmeans_avg_pixel_array = np.zeros((height, width, 3), dtype=np.uint8)
for _, row in pixel_df.iterrows():
    x, y = row["x"]*size, row["y"]*size
    kmeans_avg_pixel_array[y:y+size, x:x+size] = row["kmeans_color"]

# Visualize the final image with K-Means average colors
plt.imshow(kmeans_avg_pixel_array)
plt.axis("off")
plt.show()

# Optional: Save the final clustered image
final_image = Image.fromarray(kmeans_avg_pixel_array)
final_image.save("kmeans_clustered_image_avg_colors.png")

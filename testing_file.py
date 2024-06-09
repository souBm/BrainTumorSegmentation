import pickle
import os
import lzma
import numpy as np
from skimage import io, color, filters
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.decomposition import PCA

# Function to load an image
def load_image(image_path):
    return io.imread(image_path, as_gray=True)

# Function to preprocess an image
def pre_process(image):
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    image_blur = filters.gaussian(image, sigma=1)
    return image_blur

# Fuzzy C-means clustering algorithm
def fuzzy_c_means(data, n_clusters, m):
    def objective_function(centers, data):
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        membership = 1 / distances ** (2 / (m - 1))
        membership_sum = np.sum(membership, axis=1)
        objective = np.sum((membership ** m) * distances)
        return objective

    initial_guess = np.random.rand(n_clusters * data.shape[-1])
    result = minimize(objective_function, initial_guess, args=(data,),
                      method='L-BFGS-B', options={'disp': False})
    return result.x.reshape(n_clusters, data.shape[-1])

# Function to segment tumor from an image
def segment_tumor(image, n_clusters=3, m=2):
    data = image.reshape(-1, 1)
    centers = fuzzy_c_means(data, n_clusters, m)
    kmeans = KMeans(n_clusters=n_clusters, init=centers.reshape(-1, 1), n_init=1)
    kmeans.fit(data)
    segmented_image = kmeans.labels_.reshape(image.shape)
    return segmented_image

# Function to load the trained model and PCA transformer
def load_model(model_path):
    with lzma.open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_pca(pca_path):
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    return pca

def generate_tumor_mask(image, clf, pca):
    image_flat = image.reshape(1, -1)  # Reshape image to 1D array for each sample
    if pca is not None:
        image_pca = pca.transform(image_flat)
    else:
        image_pca = image_flat
    print("Shape of image_pca:", image_pca.shape)
    print("Number of features expected by the classifier:", clf.n_features_in_)
    # Ensure that the input data has the same number of features as the trained model
    if image_pca.shape[1] != clf.n_features_in_:
        raise ValueError(f"Number of features in input ({image_pca.shape[1]}) does not match "
                         f"the number of features expected by the classifier ({clf.n_features_in_})")
    tumor_mask = clf.predict(image_pca)
    tumor_mask = tumor_mask.reshape(image.shape)
    return tumor_mask

# Main function
if __name__ == "__main__":
    input_folder = "./DATA/test-images"
    model_path = "./DATA/model/random_forest_model.pkl"
    pca_path = "./DATA/model/pca_transformer.pkl"
    output_folder = "./DATA/output_masks"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load trained model and PCA transformer
    clf = load_model(model_path)
    pca = load_pca(pca_path)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load input image
            input_image_path = os.path.join(input_folder, filename)
            print(f"Processing file: {input_image_path}")
            image = load_image(input_image_path)

            # Preprocess input image
            preprocessed_image = pre_process(image)

            # Generate tumor mask
            tumor_mask = generate_tumor_mask(preprocessed_image, clf, pca)

            # Save tumor mask
            output_path = os.path.join(output_folder, f"tumor_mask_{filename}")
            print(f"Saving tumor mask to: {output_path}")
            io.imsave(output_path, tumor_mask)

    print("Processing completed.")

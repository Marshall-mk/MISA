import itk
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as sitk
from pathlib import Path
from typing import List
from sklearn.cluster import KMeans
import warnings
import math
warnings.filterwarnings("ignore")

def calculate_tissue_models(image_files_list: str, label_files_list) -> np.ndarray:
    """
    Calculate tissue models from training images and their corresponding labels.
    
    Args:
        set_directory (str): Path to directory containing the images and labels
        
    Returns:
        np.ndarray: Posterior probability distributions for each tissue class
    """
    num_classes = 3
    label_histograms = np.zeros((3, 256))
    
    
    # Process each image and its corresponding label
    for image_file in tqdm(image_files_list, total=len(image_files_list)):
        image_filepath = image_file
        label_filepath = label_files_list[image_files_list.index(image_file)]
        
        # Load and normalize image
        images = sitk.GetArrayFromImage(sitk.ReadImage(image_filepath))
        labels = sitk.GetArrayFromImage(sitk.ReadImage(label_filepath))
        images = (images - images.min()) / (images.max() - images.min()) * 255
        
        # Calculate histogram for each class
        for c in range(0, num_classes):
            label_histograms[c, :] += np.histogram(images[labels == c+1], 
                                                 bins=256, 
                                                 range=[0, 256])[0]
    
    # Calculate posterior probabilities
    label_histograms_density = label_histograms / np.sum(label_histograms, axis=0)[None]
    
    return label_histograms_density

def segment_image_with_tissue_model(image: np.ndarray, tissue_models: np.ndarray, gt_label) -> np.ndarray:
    """
    Segment an input image using pre-calculated tissue models and ground truth label mask.

    Args:
        image (np.ndarray): Input image to be segmented
        tissue_models (np.ndarray): Pre-calculated tissue probability models of shape (num_classes, 256)
            representing the probability distribution for each tissue class
        gt_label (np.ndarray): Ground truth label mask of same shape as input image,
            used to mask out background (where gt_label == 0)

    Returns:
        np.ndarray: Segmented image of same shape as input, where each pixel value
            represents the predicted tissue class (0-3)
    """
    # Normalize image to [0, 255] range
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    
    # Mask out background
    # image[gt_label == 0] = 0
    
    flat_image = image.ravel()
    posteriors = tissue_models[:, flat_image]  # Shape: (num_classes, num_pixels)
    segmentation = np.argmax(posteriors, axis=0) +1
    
    # Reshape back to original image dimensions
    return segmentation.reshape(image.shape)*(gt_label != 0)

def calculate_dice_coefficient(fixed_label: itk.Image, registered_label: itk.Image) -> float:  
    # Ensure binary labels
    dice=0
    for cls in [1,2,3]:
        fixed_array = (fixed_label == cls).astype(int)
        registered_array = (registered_label == cls).astype(int)
        intersection = np.sum(fixed_array * registered_array)
        sum_fixed = np.sum(fixed_array)
        sum_registered = np.sum(registered_array)
        dice += (2.0 * intersection) / (sum_fixed + sum_registered + 1e-6) 
    return dice/3

def dice_coefficient(segmented_image: np.ndarray, gt_labels: np.ndarray, label: int) -> float:
    """Calculate Dice coefficient for a specific label.
    
    Args:
        segmented_image (np.ndarray): Predicted segmentation
        gt_labels (np.ndarray): Ground truth labels
        label (int): Label to calculate Dice coefficient for
        
    Returns:
        float: Dice coefficient value between 0 and 1
    """
    seg_mask = segmented_image == label
    gt_mask = gt_labels == label
    
    intersection = np.sum(seg_mask & gt_mask)
    denominator = np.sum(seg_mask) + np.sum(gt_mask)
    
    return (2.0 * intersection) / denominator if denominator > 0 else 0.0

def show_results(image, gt, prediction, modality='T1', i=0, slice_idx=150, plot=True, show_hist=True, cmap='tab20'):
    csf, gm, wm = dice_coefficient(prediction, gt, 1), dice_coefficient(prediction, gt, 2), dice_coefficient(prediction, gt, 3)
    print(f'Dice Score: CSF: {csf}, GM: {gm}, WM: {wm}, Avg: {(csf+gm+wm)/3}')
    if plot:
        slice_idx = slice_idx
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 5, 1)
        plt.imshow(image[slice_idx], cmap='gray')
        plt.title(f'Original Image Slice for {modality} image {i+1}')
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.imshow(gt[slice_idx], cmap='gray')
        plt.title('Ground Truth Slice')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.imshow(prediction[slice_idx], cmap='gray')
        plt.title('Segmented Image Slice')
        plt.axis('off')
        plt.tight_layout()
        
        # plot overlay
        overlay = np.zeros_like(image[slice_idx])
        overlay[prediction[slice_idx] == 1] = 1
        overlay[prediction[slice_idx] == 2] = 2
        overlay[prediction[slice_idx] == 3] = 3
        plt.subplot(1, 5, 4)
        plt.imshow(image[slice_idx], cmap='gray')
        plt.imshow(overlay, cmap=cmap, alpha=0.5)
        plt.title('Prediction Overlay')
        plt.axis('off')
        
        # ground truth overlay
        overlay_gt = np.zeros_like(image[slice_idx])
        overlay_gt[gt[slice_idx] == 1] = 1
        overlay_gt[gt[slice_idx] == 2] = 2
        overlay_gt[gt[slice_idx] == 3] = 3
        plt.subplot(1, 5, 5)
        plt.imshow(image[slice_idx], cmap='gray')
        plt.imshow(overlay_gt, cmap=cmap, alpha=0.5)
        plt.title('Ground Truth Overlay')
        plt.axis('off')
        
        if show_hist:
            # plot histogram
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 3, 1)
            plt.hist(image[slice_idx].ravel(), bins=256, histtype='step', color='black')
            plt.title(f'Original Image Histogram for {modality} image {i+1}')
            plt.xlabel('Intensity')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 3, 2)
            plt.hist(gt[slice_idx].ravel(), bins=256, histtype='step', color='black')
            plt.title('Ground Truth Histogram') 
            plt.xlabel('Intensity')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 3, 3)
            plt.hist(prediction[slice_idx].ravel(), bins=256, histtype='step', color='black')
            plt.title('Segmented Image Histogram')
            plt.xlabel('Intensity')
            plt.ylabel('Frequency')
            plt.tight_layout()
        
        
        plt.show()
    return csf, gm, wm

def load_and_process_images(image_files_list: str, label_files_list):
    """Load and process multiple images efficiently."""
    tissue_data = {i: [] for i in range(1, 4)}  # Initialize dict for 3 tissue types

    # Process each image and its corresponding label
    for image_file in tqdm(image_files_list, total=len(image_files_list)):
        image_filepath = image_file
        label_filepath = label_files_list[image_files_list.index(image_file)]
        
        # Load and normalize image
        img = sitk.GetArrayFromImage(sitk.ReadImage(image_filepath))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(label_filepath))
        
        # Process each tissue type
        for tissue_type in range(1, 4):
            tissue_data[tissue_type].extend(img[lab == tissue_type].flatten())
    
    return tissue_data

def calculate_probabilities(tissue_data):
    """Calculate probability distributions for each tissue type."""
    # Find unique intensity values across all tissues
    all_values = sorted(set().union(*[set(data) for data in tissue_data.values()]))
    
    # Calculate probabilities using numpy for better performance
    probabilities = {}
    total_counts = np.zeros(len(all_values))
    tissue_counts = {}
    
    for tissue_type, data in tissue_data.items():
        # Use numpy histogram for efficient counting
        counts, _ = np.histogram(data, bins=len(all_values), range=(min(all_values), max(all_values)))
        tissue_counts[tissue_type] = counts
        total_counts += counts
    
    # Calculate probabilities for each tissue type
    for tissue_type in tissue_data.keys():
        probabilities[tissue_type] = np.divide(
            tissue_counts[tissue_type], 
            total_counts,
            out=np.zeros_like(total_counts, dtype=float),
            where=total_counts > 0
        )
    
    return all_values, probabilities

def tissue_prob(img, lab, probabilities, all_values):
    """
    Calculate tissue model probabilities for an image using pre-computed probability maps.
    
    Args:
        img (np.ndarray): Input image
        lab (np.ndarray): Label mask
        probabilities (dict): Pre-computed probability distributions for each tissue class
        all_values (np.ndarray): Array of intensity values used for probability lookup
        
    Returns:
        np.ndarray: Tissue model probabilities for each valid voxel
    """
    # Create mask for valid voxels (non-zero and labeled)
    mask = (lab > 0) & (img != 0)
    
    # Convert probabilities dict to array once
    prob_array = np.stack(list(probabilities.values()))
    
    # Get indices for probability lookup, clipped to valid range
    value_indices = np.clip(
        np.searchsorted(all_values, img[mask], side='left'),
        0, 
        prob_array.shape[1] - 1
    )
    
    # Directly index probability array for masked voxels
    return prob_array[:, value_indices]

def relabel(gt, predictions, image):
    """
    Find the best label configuration and reassign labels accordingly.

    Args:
        gt: Ground truth labels
        predictions: Predicted labels
        image: Original image for relabeling

    Returns:
        relabeled_image: Image with reassigned labels
        relabeled_predictions: Predictions with reassigned labels
        best_score: Best Dice score achieved
    """
    def generate_label_combinations():
        return [
            {1: 1, 2: 2, 3: 3},
            {1: 1, 2: 3, 3: 2},
            {1: 2, 2: 3, 3: 1},
            {1: 2, 2: 1, 3: 3},
            {1: 3, 2: 1, 3: 2},
            {1: 3, 2: 2, 3: 1}
        ]

    def apply_label_mapping(data, config):
        relabeled_data = np.zeros(data.shape)
        for old_label, new_label in config.items():
            relabeled_data[data == old_label] = new_label
        return relabeled_data

    def calculate_dice_for_label(gt, worker, label):
        intersection = np.sum(np.bitwise_and(worker == label, gt == label))
        union = np.sum(worker == label) + np.sum(gt == label)
        return (2 * intersection) / union if union > 0 else 0

    def calculate_dice_scores(gt, predictions, config):
        worker = apply_label_mapping(predictions, config)
        labels = np.unique(worker)
        dice_scores = [calculate_dice_for_label(gt, worker, label) for label in labels if label != 0]
        return dice_scores

    # Main logic to find the best configuration
    configurations = generate_label_combinations()
    best_mean_score = 0
    best_config = configurations[0]
    best_score = []

    for config in configurations:
        dice_scores = calculate_dice_scores(gt, predictions, config)
        mean_score = np.mean(dice_scores)
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_config = config
            best_score = dice_scores

    print('Best combination has average dice score of:', best_mean_score)

    relabeled_image = apply_label_mapping(image, best_config)
    relabeled_predictions = apply_label_mapping(predictions, best_config)
    
    return relabeled_image, relabeled_predictions, best_score

# Global Functions    
def gaussian(inp, mean, cov, k, d):
  num = []

  inv = np.linalg.inv(cov[k] + 1e-6 * np.eye(d))
  den = (2*math.pi)**(d/2) * math.sqrt(np.linalg.det(cov[k]) + 1e-6)
  diff = inp - mean[k].reshape((-1, 1))

  num = -0.5 * np.sum(diff * (inv @ diff), axis=0)

  gaussian = np.exp(num)/(den + 1e-6)
  return np.array(gaussian)

def EM(data, data_label=None, init='kmeans', mask=None, init_weights=None, 
       comb_mode=None, atlas_weights=None, freq=2, max_iter=100):
    """
    Expectation-Maximization algorithm for image segmentation with improved stability and atlas integration.
    
    Args:
        data: Input image data
        data_label: Label data (optional)
        init: Initialization method ('kmeans' or 'other')
        mask: Binary mask for valid pixels
        init_weights: Initial weights if not using kmeans
        comb_mode: Atlas combination mode ('into', 'end', or None)
        atlas_weights: Atlas probabilities for combination
        freq: Frequency of atlas combination for 'into' mode
        max_iter: Maximum number of iterations
        
    Returns:
        pred_em: Predicted segmentation labels
    """
    # Constants for numerical stability
    epsilon = 1e-10
    min_covar = 1e-7
    tol = 1e-6
    
    # Initialize mask if not provided
    if mask is None and init == 'kmeans':
        mask = data_label > 0

    # Prepare data
    y = np.array([data[mask]])
    d, k = 1, 3  # dimensions and number of clusters

    # Initialize parameters
    if init == 'kmeans':
        # Initialize with KMeans
        kmeans = KMeans(n_clusters=k, random_state=42).fit(y.T)
        x = kmeans.predict(y.T)
        mean = kmeans.cluster_centers_
        alpha = np.array([np.mean(x == i) for i in range(k)])
        cov = np.array([np.cov(y.T[x == i], rowvar=False) for i in range(k)])
        cov = cov[:, np.newaxis, np.newaxis] if d == 1 else cov
        weights = np.array([gaussian(y, mean, cov, i, d) * alpha[i] for i in range(k)])
    else:
        # Initialize with provided weights
        weights = init_weights
        N_k = np.sum(weights, axis=1, keepdims=True)
        mean = (weights @ y.T) / (N_k + epsilon)
        cov = np.array([np.eye(d)] * k)

    # Normalize initial weights
    weights /= (np.sum(weights, axis=0) + epsilon)
    prev_weights = np.zeros_like(weights)
    prev_ll = -np.inf

    # EM algorithm
    for v in range(max_iter):
        # M-Step
        N_k = np.sum(weights, axis=1, keepdims=True)
        alpha = N_k / len(y[0])
        mean = (weights @ y.T) / (N_k + epsilon)
        
        # Calculate and regularize covariance
        cov = np.array([
            np.dot((weights[i] * (y - mean[i].reshape(-1, 1))), 
                  (y - mean[i].reshape(-1, 1)).T) / (N_k[i] + epsilon) 
            for i in range(k)
        ])
        for i in range(k):
            cov[i] += min_covar * np.eye(d)

        # Save previous weights and calculate previous log-likelihood
        prev_weights = weights.copy()
        prev_ll = np.sum(np.log(np.sum(prev_weights, axis=0) + epsilon))

        # E-Step
        weights = np.array([gaussian(y, mean, cov, i, d) * alpha[i] for i in range(k)])
        weights /= (np.sum(weights, axis=0) + epsilon)

        # Combine with atlas weights if specified
        if comb_mode == 'into' and v % freq == 0 and atlas_weights is not None:
            normalized_atlas = atlas_weights / (np.sum(atlas_weights, axis=0) + epsilon)
            weights = 0.5 * weights + 0.5 * normalized_atlas
            weights /= (np.sum(weights, axis=0) + epsilon)

        # Check convergence
        new_ll = np.sum(np.log(np.sum(weights, axis=0) + epsilon))
        if v > 0:
            relative_change = np.abs((new_ll - prev_ll) / (prev_ll + epsilon))
            if relative_change < tol:
                print(f'Converged at iteration {v} with log-likelihood {new_ll:.4f}')
                break

    # Final atlas combination if specified
    if comb_mode == 'end' and atlas_weights is not None:
        normalized_atlas = atlas_weights / (np.sum(atlas_weights, axis=0) + epsilon)
        weights = 0.5 * weights + 0.5 * normalized_atlas
        weights /= (np.sum(weights, axis=0) + epsilon)

    # Final prediction
    pred_em = np.argmax(weights, axis=0)
    return pred_em
    
def save_probability_atlases(
    val_image_paths: List,
    mean_volume,
    probability_atlases: List,
    topological_atlas,
    class_labels: dict,
    output_dir: str = "val_probability_atlases"
) -> None:
    """
    Save probability atlases for test images after registration.
    
    Args:
        test_image_directory (str): Directory containing test images
        test_label_directory (str): Directory containing test labels
        mean_volume: Reference mean volume image
        probability_atlases (List): List of probability atlas images
        topological_atlas: Reference topological atlas
        class_labels (dict): Dictionary mapping class indices to names
        output_dir (str): Directory to save probability atlases
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    
    for image_path in tqdm(val_image_paths, desc="Processing images"):
        try:
            # Create image-specific output directory
            image_output_dir = Path(f"{output_dir}/{os.path.splitext(image_path)[0]}")
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load fixed image
            fx_image = itk.imread(image_path, itk.F)
            
            # Setup registration parameters
            parameter_object = itk.ParameterObject.New()
            parameter_object.AddParameterFile("parameters/9/Parameters.Par0009.affine.txt")
            parameter_object.AddParameterFile("parameters/9/Parameters.Par0009.elastic.txt")
            
            # Perform registration
            _, result_transform_parameters = itk.elastix_registration_method(
                fx_image, mean_volume,
                parameter_object=parameter_object,
                log_to_console=False
            )
            
            # Set interpolator for transformations
            result_transform_parameters.SetParameter("ResampleInterpolator", 
                                                ["FinalNearestNeighborInterpolator"])
            
            # Transform and save probability atlases
            for c, atlas in enumerate(probability_atlases):
                prob_atlas = itk.transformix_filter(atlas, result_transform_parameters)
                prob_atlas = sitk.GetImageFromArray(prob_atlas)
                prob_atlas.CopyInformation(sitk.ReadImage(image_path))
                sitk.WriteImage(
                    prob_atlas, 
                    str(image_output_dir / f"probability_atlas_{class_labels[c]}.nii.gz")
                )
            
            # Transform and save topological atlas
            topological_atlas_warped = itk.transformix_filter(topological_atlas, result_transform_parameters)
            topological_atlas_warped = sitk.GetImageFromArray(topological_atlas_warped)
            topological_atlas_warped.CopyInformation(sitk.ReadImage( image_path))
            sitk.WriteImage(
                topological_atlas_warped, 
                str(image_output_dir / "topological_atlas.nii.gz")
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

def plot_tissue_model(all_values, probabilities):
    """Plot the tissue model distributions."""
    plt.figure(figsize=(10, 6))
    
    colors = {1: 'blue', 2: 'orange', 3: 'green'}
    labels = {1: 'CSF', 2: 'WM', 3: 'GM'}
    
    x = np.arange(len(all_values))
    for tissue_type, probs in probabilities.items():
        plt.plot(x, probs, label=labels[tissue_type], color=colors[tissue_type])
    
    plt.xlabel('Intensity values')
    plt.ylabel('Probability of Belonging to Each Class')
    plt.title('Tissue Model')
    plt.xticks(np.arange(0, len(all_values), 200))
    plt.legend()
    plt.show()


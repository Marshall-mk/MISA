import os
import matplotlib.pyplot as plt
from monai.transforms import (
    Spacingd,
)
import SimpleITK as sitk
import numpy as np
from monai.utils import set_determinism
from intensity_normalization.plot.histogram import HistogramPlotter
import voxelmorph as vxm
import tensorflow as tf
set_determinism(seed=42)

def bias_correct(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    return sitk.N4BiasFieldCorrection(image)

def intensity_normalize(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Normalize(image)

def visualize_histograms(images, title):
    """Visualize histograms of the given images."""
    hp = HistogramPlotter(title=title)
    _ = hp(images, masks=None)
    plt.show()
    
def create_data_files(root_dir, is_train=True, is_val=False):
    subdirs = sorted(os.listdir(root_dir))
    images = []
    labels = []

    for subdir in subdirs:
        img_path = os.path.join(root_dir, subdir, f"{subdir}.nii.gz")
        lbl_path = os.path.join(root_dir, subdir, f"{subdir}_seg.nii.gz")
        if is_train or is_val:
            images.append(img_path)
            labels.append(lbl_path)
        else:
            images.append(img_path)
            
    if is_train or is_val:
        data_files = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    else:
        data_files = [{"image": img} for img in images]
    return data_files

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
        plt.title(f'Original Image Slice for {modality} image {i+1}', fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.imshow(gt[slice_idx], cmap='gray')
        plt.title('Ground Truth Slice', fontsize=12)
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.imshow(prediction[slice_idx], cmap='gray')
        plt.title('Segmented Image Slice', fontsize=12)
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
        plt.title('Prediction Overlay', fontsize=12)
        plt.axis('off')
        
        # ground truth overlay
        overlay_gt = np.zeros_like(image[slice_idx])
        overlay_gt[gt[slice_idx] == 1] = 1
        overlay_gt[gt[slice_idx] == 2] = 2
        overlay_gt[gt[slice_idx] == 3] = 3
        plt.subplot(1, 5, 5)
        plt.imshow(image[slice_idx], cmap='gray')
        plt.imshow(overlay_gt, cmap=cmap, alpha=0.5)
        plt.title('Ground Truth Overlay', fontsize=12)
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

def load_and_preprocess(data_files, prefix):
    for data in data_files:
        image_path = data["image"]
        label_path = data["label"] if "label" in data else None
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        label = sitk.ReadImage(label_path, sitk.sitkUInt8) if label_path is not None else None
        
        # Apply bias field correction and save the corrected image
        corrected_image = bias_correct(image)
        corrected_image_name = os.path.basename(image_path).replace(".nii.gz", "_bias_corrected.nii.gz")
        corrected_image_output_path = os.path.join(os.path.dirname(image_path), corrected_image_name)
        sitk.WriteImage(corrected_image, corrected_image_output_path)

        # Apply intensity normalization and save the normalized image
        normalized_image = intensity_normalize(corrected_image)
        normalized_image_name = os.path.basename(image_path).replace(".nii.gz", "_normalized.nii.gz")
        normalized_image_output_path = os.path.join(os.path.dirname(image_path), normalized_image_name)
        sitk.WriteImage(normalized_image, normalized_image_output_path)

        # Convert to NumPy array for resampling
        normalized_image_np = sitk.GetArrayFromImage(normalized_image)

        # Resample the image and save the resampled image
        respaced_image = Spacingd(keys=["image"], pixdim=(1.0, 1.5, 1.0), mode=("bilinear"))({"image": normalized_image_np})["image"]
        respaced_image_name = os.path.basename(image_path).replace(".nii.gz", "_respaced.nii.gz")
        respaced_image_output_path = os.path.join(os.path.dirname(image_path), respaced_image_name)
        sitk.WriteImage(sitk.GetImageFromArray(respaced_image), respaced_image_output_path)
        
        # Resample the label and save the resampled label
        if label is not None:
            label_np = sitk.GetArrayFromImage(label)
            respaced_label = Spacingd(keys=["label"], pixdim=(1.0, 1.5, 1.0), mode=("nearest"))({"label": label_np})["label"]
            respaced_label_name = os.path.basename(label_path).replace(".nii.gz", "_respaced.nii.gz")
            respaced_label_output_path = os.path.join(os.path.dirname(label_path), respaced_label_name)
            sitk.WriteImage(sitk.GetImageFromArray(respaced_label), respaced_label_output_path)

    return f"Done preprocessing {prefix} images"

def compute_mattes_mi(moved_path: str, fixed_path: str) -> float:
    """Compute the Mattes mutual information between two images.

    Args:
        moved_path (str): Path to the moved image.
        fixed_path (str): Path to the fixed image.

    Returns:
        float: The computed Mattes mutual information metric.
    """
    # Read the moved and fixed images using SimpleITK
    moved_image = sitk.ReadImage(moved_path)
    fixed_image = sitk.ReadImage(fixed_path)

    # Convert the moved image to float64 for improved accuracy in calculations
    moved_image = sitk.Cast(moved_image, sitk.sitkFloat64)

    # Initialize the registration method and set the metric to Mattes mutual information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation()

    # Evaluate the metric between the moved and fixed images
    metric = registration_method.MetricEvaluate(moved_image, fixed_image)

    return metric

def voxmorph_register(fixed_path: str, moving_path: str, label_path: str):
    """Register a moving image to a fixed image using VoxelMorph.

    Args:
        fixed_path (str): Path to the fixed image.
        moving_path (str): Path to the moving image.
        label_path (str): Path to the label image.

    Returns:
        tuple: A tuple containing the moved image, moved label, warp, and fixed affine.
    """
    # Define the model path for VoxelMorph
    model_path = 'voxelmorph/vxm_dense_brain_T1_3D_mse.h5'

    # Set up the TensorFlow device for computation
    device, nb_devices = vxm.tf.utils.setup_device()
    print('Using device:', device)

    # Load the moving and label images with batch and feature axes
    moving = vxm.py.utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=True)
    label = vxm.py.utils.load_volfile(label_path, add_batch_axis=True, add_feat_axis=True)

    # Change label values: set background (0) to a different value (2)
    label[label == 0] = 2

    # Load the fixed image and its affine transformation
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_path, add_batch_axis=True, add_feat_axis=True, ret_affine=True
    )

    # Extract the input shape and number of features from the moving image
    inshape = moving.shape[1:-1]  # Exclude batch and feature dimensions
    nb_feats = moving.shape[-1]     # Number of features in the moving image

    with tf.device(device):
        # Load the VoxelMorph model configuration
        config = {'inshape': inshape, 'input_model': None}
        
        # Register the moving image to the fixed image and obtain the warp
        warp = vxm.networks.VxmDense.load(model_path, **config).register(moving, fixed)
        
        # Apply the warp to the moving image and label
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])
        moved_label = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([label, warp])

    # Mask the moved label using the moved image: set background to 0
    moved_label[moved == 0] = 0

    return moved, moved_label, warp, fixed_affine
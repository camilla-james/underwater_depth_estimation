
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as fn
from torch import nn
import numpy as np
import torch
import scipy
import wandb

class EvaluationMetric:

    def __init__(self, wandb_logging:bool) -> None:
        self.logging = wandb_logging


    def scale_offset(self, y_pred, y_true):
        scale_factor = np.mean(y_pred) / np.mean(y_true)

        # Adjust the second depth map by the scale factor
        true_scaled = y_pred * scale_factor

         # Calculate the offset
        offset = np.mean(y_pred) - (scale_factor * np.mean(y_true))

        # Adjust the second depth map by the offset
        true_adjusted = true_scaled + offset

        val_min = y_pred.min()
        val_range = y_pred.max() - val_min + 1e-7

        pred_normed = (y_pred - val_min) / val_range

        # apply identical normalization to the denoised image (important!)
        true_adjusted_normed = (true_adjusted - val_min) / val_range

        return pred_normed, true_adjusted_normed

    def absolute_relative_error(self, y_pred, y_true):
        """
        Calculate the Absolute Relative Error (MARE).

        Parameters:
        y_pred : torch.Tensor
            Predicted depth values.
        y_true : torch.Tensor
            Ground truth depth values.

        Returns:
        float
        Absolute Relative Error (MARE).
        """
        y_pred, y_true = self.scale_offset(y_pred, y_true)
        # mask = y_true == 0
        # y_true[mask] = 1 
        absolute_relative_error = np.abs(y_pred - y_true) / y_true

        return np.mean(absolute_relative_error)
    
    def root_mean_squared_error(self, y_pred, y_true, log = False): 
        
        y_pred, y_true = self.scale_offset(y_pred, y_true)
        if log:
            y_pred = np.log(y_pred)
            y_true = np.log(y_true)
        mse = np.mean((y_pred - y_true)**2)
        rmse = np.sqrt(mse)
        return rmse

    def delta1_metric(self, y_pred, y_true, threshold=1.25):
        """
        Calculate the δ1 metric for monocular depth estimation.

        Parameters:
        y_pred : torch.Tensor
            Predicted depth values.
        y_true : torch.Tensor
            Ground truth depth values.
        threshold : float, optional
            Threshold for considering a pixel as correctly estimated (default is 1.25).

        Returns:
        float
            Percentage of pixels for which max(d*/d, d/d*) < threshold.
            
        """
        y_pred, y_true = self.scale_offset(y_pred, y_true)
        # Compute element-wise ratios
        ratio_1 = y_true / (y_pred + 1e-7)  # Adding epsilon to avoid division by zero
        ratio_2 = (y_pred + 1e-7) / y_true  # Adding epsilon to avoid division by zero
        
        # Calculate element-wise maximum ratio
        max_ratio = torch.max(ratio_1, ratio_2)
        
        # Count the number of pixels where max_ratio < threshold
        num_correct_pixels = torch.sum(max_ratio < threshold).item()
        
        # Calculate the percentage of pixels satisfying the condition
        total_pixels = y_true.numel()
        percentage_correct = (num_correct_pixels / total_pixels) * 100.0
        
        return percentage_correct
    
    def si_log(y_pred, y_true):
        """
        Calculate the Scale Invarient error that takes into account the global scale of a scene. 
        This metric is sensitive to the relationships between points in the scene, 
        irrespective of the absolute global scale.

        Parameters:
        y_pred : torch.Tensor
            Predicted depth values.
        y_true : torch.Tensor
            Ground truth depth values.
    
        Returns:
        float
            SI Error
            
        """
        bs = y_pred.shape[0]

        y_pred = torch.reshape(y_pred, (bs, -1))
        y_true = torch.reshape(y_true, (bs, -1))

        mask = y_true > 0  # 0=missing y_true
        num_vals = mask.sum(dim=1)

        log_diff = torch.zeros_like(y_pred)
        log_diff[mask] = torch.log(y_pred[mask]) - torch.log(y_true[mask])
        
        si_log_unscaled = torch.sum(log_diff**2, dim=1) / num_vals - (torch.sum(log_diff, dim=1)**2) / (num_vals**2)
        si_log_score = torch.sqrt(si_log_unscaled) * 100
        
        si_log_score = torch.mean(si_log_score)
        return si_log_score

    def compute_metrics(self, input_image, outputs, labels):
        metrics = []

        with torch.no_grad():
            predicted_depth = outputs
            img_size = fn.get_image_size(input_image)
            img_size.reverse()
            prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size = img_size,
            mode = "bicubic",
            align_corners=False)
            
            # Convert depth prediction to numpy array and resize to match ground truth depth map size
            depth_output = prediction.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            # Handle invalid or unexpected depth values
            depth_output[depth_output <= 0] = 1e-7  # Replace negative or zero values with a small epsilon
            labels[labels <= 0] = 1e-7 

            # Calculate metrics
            absRel = self.absolute_relative_error(depth_output, labels)
            
            rmse = self.root_mean_squared_error(depth_output, labels)
            rmseLog = self.root_mean_squared_error(depth_output, labels, log = True)
            
            out_t = torch.from_numpy(depth_output)
            labels_t = torch.from_numpy(labels)
            
            delta1 = np.mean(self.delta1_metric(out_t, labels_t))
            si_error = self.si_log(out_t, labels_t)
            

            if self.logging == True:
                wandb.log({"Absolute Relative error (AbsRel)": absRel})
                wandb.log({"Root Mean Squared Error (RMSE)": rmse})
                wandb.log({"Log Root Mean Squared Error (Log-RMSE)": rmseLog})
                wandb.log({"Scale Invarient Error (SI Error)": si_error})
                wandb.log({"Delta1 with thresold=1.25": delta1})

            metrics.append([absRel, rmse, rmseLog, si_error, delta1])
            
            return metrics
        
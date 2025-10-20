"""CLIP model wrapper for feature extraction and similarity comparison"""
import os
import random
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm


class CLIPModel:
    """Wrapper class for CLIP model operations"""

    def __init__(self, model_name="ViT-B/32", device=None, gpu_id="0"):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model variant (default: "ViT-B/32")
            device: device to run model on (default: auto-detect)
            gpu_id: GPU ID to use (default: "0")
        """
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"CLIP model loaded on device: {self.device}")

    def extract_features(self, image):
        """
        Extract CLIP features from an image.

        Args:
            image: PIL Image, numpy array, or path to image file

        Returns:
            normalized feature tensor
        """
        # Convert to PIL Image if necessary
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Preprocess and extract features
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)

        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)

        return features

    def compute_similarity(self, features1, features2):
        """
        Compute cosine similarity between two feature vectors.

        Args:
            features1: first feature tensor
            features2: second feature tensor

        Returns:
            similarity score (float)
        """
        similarity = (features1 @ features2.T).item()
        return similarity

    def compare_images(self, image1, image2):
        """
        Compare two images and return their similarity score.

        Args:
            image1: first image (PIL Image, numpy array, or path)
            image2: second image (PIL Image, numpy array, or path)

        Returns:
            similarity score (float)
        """
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        return self.compute_similarity(features1, features2)

    def batch_similarity(self, image_dir, K=10, patch_size=64, file_extension='.png'):
        """
        Compute similarity between full images and random patches.

        For each image in the directory:
        1. Extract K random patches
        2. Compute similarity between each patch and the full image
        3. Return average similarity

        Args:
            image_dir: directory containing images
            K: number of random patches per image (default: 10)
            patch_size: size of each square patch (default: 64)
            file_extension: file extension to filter (default: '.png')

        Returns:
            dict with overall statistics and per-image results
        """
        # Get all image files
        image_files = [f for f in os.listdir(image_dir)
                      if f.endswith(file_extension)]

        print(f"Found {len(image_files)} images in {image_dir}")

        all_similarities = []
        per_image_results = {}

        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_dir, img_file)

            # Load the full image
            full_img = Image.open(img_path).convert('RGB')
            width, height = full_img.size

            # Skip if image is too small for patches
            if width < patch_size or height < patch_size:
                print(f"Skipping {img_file}: image too small")
                continue

            # Extract features from full image
            full_features = self.extract_features(full_img)

            image_similarities = []

            # Extract K random patches
            for k in range(K):
                # Random crop coordinates
                left = random.randint(0, width - patch_size)
                top = random.randint(0, height - patch_size)
                right = left + patch_size
                bottom = top + patch_size

                # Extract patch
                patch = full_img.crop((left, top, right, bottom))

                # Extract features from patch
                patch_features = self.extract_features(patch)

                # Compute cosine similarity
                similarity = self.compute_similarity(patch_features, full_features)
                image_similarities.append(similarity)

            # Compute average similarity for this image
            avg_similarity = np.mean(image_similarities)
            all_similarities.append(avg_similarity)
            per_image_results[img_file] = avg_similarity

            print(f"{img_file}: Average similarity = {avg_similarity:.4f}")

        # Compute overall statistics
        overall_avg = np.mean(all_similarities)
        overall_std = np.std(all_similarities)

        print(f"\nOverall average cosine similarity: {overall_avg:.4f}")
        print(f"Standard deviation: {overall_std:.4f}")

        return {
            'overall_mean': overall_avg,
            'overall_std': overall_std,
            'all_similarities': all_similarities,
            'per_image_results': per_image_results
        }

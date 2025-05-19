import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def sharpness_score(gray_img):
    # Laplacian variance as sharpness score
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def contrast_score(gray_img):
    # Enhanced contrast measure using histogram metrics
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Calculate non-zero percentiles for better contrast assessment
    non_zero = gray_img[gray_img > 15]  # Ignore near-black pixels
    if len(non_zero) > 0:
        p_low = np.percentile(non_zero, 10) if len(non_zero) > 0 else 0
        p_high = np.percentile(non_zero, 90) if len(non_zero) > 0 else 255
        dynamic_range = p_high - p_low
    else:
        dynamic_range = 0
    
    # Standard deviation gives us texture information
    std_dev = gray_img.std()
    
    return std_dev * 0.5 + dynamic_range * 0.5

def completeness_score(gray_img):
    # Check if the eye is complete and well-centered
    h, w = gray_img.shape
    center_region = gray_img[h//4:3*h//4, w//4:3*w//4]
    
    # Center should be different from edges (pupil vs sclera)
    center_mean = center_region.mean()
    edge_mean = gray_img.mean() - (center_region.sum() / (gray_img.size - center_region.size))
    center_edge_diff = abs(center_mean - edge_mean)
    
    # Check for circular patterns (iris detection)
    edge_img = cv2.Canny(gray_img, 30, 70)
    circles = cv2.HoughCircles(
        gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w//2,
        param1=50, param2=30, minRadius=w//6, maxRadius=w//2
    )
    
    circle_score = 50 if circles is not None else 0
    return center_edge_diff + circle_score

def combined_quality_score(gray_img):
    # Get basic image stats
    h, w = gray_img.shape
    area = h * w
    
    # Check if image is too small, too dark or too bright
    if area < 400 or gray_img.mean() < 30 or gray_img.mean() > 220:
        return 0
    
    # Calculate individual scores
    sharp = sharpness_score(gray_img)
    contr = contrast_score(gray_img)
    compl = completeness_score(gray_img)
    
    # Normalize sharpness (can get very high for some images)
    sharp_norm = min(sharp, 300) / 300 * 100
    
    # Apply weights to each component
    return 0.4 * sharp_norm + 0.3 * contr + 0.3 * compl

def detect_best_eye(image_path, save_best=False, output_dir=None, visualize=False):
    """
    Detects the best quality eye in the image.
    
    Args:
        image_path (str): Path to the input image
        save_best (bool): Whether to save the best eye image to disk
        output_dir (str): Directory to save the output images (if None, uses the same directory as input)
        visualize (bool): Whether to display the visualization
        
    Returns:
        tuple: (best_eye_image, best_eye_score, all_eye_images, all_scores)
            - best_eye_image: BGR image of the best eye
            - best_eye_score: Quality score of the best eye
            - all_eye_images: List of all detected eye images
            - all_scores: List of all eye quality scores
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect eyes with more strict parameters
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(25, 25))

    # Filter out small or very narrow detections
    filtered_eyes = [eye for eye in eyes if eye[2] > 25 and eye[3] > 20 and 0.5 < eye[2]/eye[3] < 2]

    if not filtered_eyes:
        print("No valid eyes detected")
        return None, 0, [], []

    # Calculate quality scores and find best eye
    scores = []
    eye_crops_gray = []
    eye_crops_color = []
    eye_rects = []  # Store the coordinates for visualization

    for (x, y, w, h) in filtered_eyes:
        # Add a margin around the eye for better context
        margin_x, margin_y = w//8, h//8
        x1, y1 = max(0, x-margin_x), max(0, y-margin_y)
        x2, y2 = min(gray.shape[1], x+w+margin_x), min(gray.shape[0], y+h+margin_y)
        
        eye_crop_gray = gray[y1:y2, x1:x2]
        eye_crop_color = img[y1:y2, x1:x2]
        
        score = combined_quality_score(eye_crop_gray)
        scores.append(score)
        eye_crops_gray.append(eye_crop_gray)
        eye_crops_color.append(eye_crop_color)
        eye_rects.append((x1, y1, x2-x1, y2-y1))  # Store expanded rectangle

    if not scores:
        print("No valid eyes detected after scoring")
        return None, 0, [], []
        
    best_index = np.argmax(scores)
    best_eye_image = eye_crops_color[best_index]
    best_eye_score = scores[best_index]
    
    # Save the best eye image if requested
    if save_best:
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}_best_eye{ext}")
        
        cv2.imwrite(output_path, best_eye_image)
        print(f"Best eye saved to: {output_path}")
    
    if visualize:
        # Plot results
        fig, axes = plt.subplots(1, len(filtered_eyes) + 1, figsize=(15, 5))

        # Original image with detected eyes
        img_with_boxes = img.copy()
        for i, rect in enumerate(eye_rects):
            x, y, w, h = rect
            color = (0, 255, 0) if i == best_index else (0, 0, 255)
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)

        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Detected Eyes (Green = Best)")
        axes[0].axis("off")

        # Show each eye crop with scores
        for i in range(len(filtered_eyes)):
            eye_crop_rgb = cv2.cvtColor(eye_crops_color[i], cv2.COLOR_BGR2RGB)
            axes[i + 1].imshow(eye_crop_rgb)
            title = f"Eye {i + 1}\nScore: {scores[i]:.1f}"
            if i == best_index:
                title += " (Best)"
            axes[i + 1].set_title(title)
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()
    
    return best_eye_image
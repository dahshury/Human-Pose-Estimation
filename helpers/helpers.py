import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import shutil
import os
import cv2
import ffmpeg
from PIL import Image
plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")

# For images
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
    
def show_img_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_img_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
def show_img_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    
# For videos
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def extract_frames(video_path, output_dir, quality=2, start_number=0):
    """
    Extracts frames from a video file and saves them as JPEG images.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_dir (str): Directory to save the extracted frames.
    - quality (int): Quality scale for the output images (lower is better quality, default is 2).
    - start_number (int): Starting number for the output images (default is 0).
    
    Example usage:
    extract_frames('video.mp4', 'output_dir/')
    """
    output_pattern = f"{output_dir}/%05d.jpg"
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    if len(os.listdir(output_dir))==0:
        
        # Build and run the ffmpeg command with -q:v
        (
            ffmpeg
            .input(video_path)
            .output(output_pattern, q=quality, start_number=start_number)
            .run()
        )
        frame_names = [
            p for p in os.listdir(output_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        return frame_names
    else:
        return "The output directory isn't empty."

def visualize_frames(frames_dir, frame_idx=0):
    # Scan all JPEG frame names in the directory
    frame_names = [
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    
    # Sort the frame names by the numerical part of the file names
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Visualize the specified frame
    if 0 <= frame_idx < len(frame_names):
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(os.path.join(frames_dir, frame_names[frame_idx])))
        plt.show()
        
    else:
        raise IndexError(f"Frame index {frame_idx} is out of range. There are only {len(frame_names)} frames available.")

def show_annotation(points, labels, out_mask_logits, out_obj_ids, vid_frames_dir, ann_frame_idx, prompts=None, show_pts=True):
    # show the results on the current (interacted) frame
    frame_names = [
        p for p in os.listdir(vid_frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx])))
    
    if show_pts:
        show_points(points, labels, plt.gca())
    
    if prompts:
        for i, out_obj_id in enumerate(out_obj_ids):
            if show_pts:
                show_points(*prompts[out_obj_id], plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    else:
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        
def merge_yolo_datasets(dataset_dir_1, dataset_dir_2, output_dir, map_1=None, map_2=None):
    
    # Create output directories
    output_dir = os.path.join(output_dir, 'combined_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    for dataset_index, (dataset_dir, map_dict) in enumerate([(dataset_dir_1, map_1), (dataset_dir_2, map_2)], 1):
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_dir, split, 'images')
            labels_dir = os.path.join(dataset_dir, split, 'labels')
            
            if os.path.exists(img_dir) and os.path.exists(labels_dir):
                # Copy images
                for img in os.listdir(img_dir):
                    shutil.copy2(os.path.join(img_dir, img), 
                                 os.path.join(output_dir, split, 'images', img))
                
                # Copy and potentially modify labels
                for label in os.listdir(labels_dir):
                    src_path = os.path.join(labels_dir, label)
                    dst_path = os.path.join(output_dir, split, 'labels', label)
                    
                    # Read label content
                    with open(src_path, 'r') as f:
                        lines = f.readlines()
                    
                    modified_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if map_dict and class_id in map_dict:
                                parts[0] = str(map_dict[class_id])
                            modified_lines.append(' '.join(parts) + '\n')
                    
                    # Write modified content
                    with open(dst_path, 'w') as f:
                        f.writelines(modified_lines)
            
            else:
                print(f"Dataset {dataset_index} not found")
        
def plot_yolo_keypoints(dataset_dir, split='train', num_images=5, plot_visible_only=True, visibility_threshold=0.5):
    """
    Plots a specified number of images with their YOLOv8 keypoint annotations.
    
    Args:
        dataset_dir (str): Path to the directory containing the dataset.
        split (str): Dataset split to use ('train', 'val', 'test').
        num_images (int): Number of images to plot.
        plot_visible_only (bool): If True, plots only visible keypoints; otherwise, plots all keypoints.
        visibility_threshold (float): Threshold for considering a keypoint visible (0.0 to 1.0).
    """
    
    # Construct paths for images and labels based on the specified split
    image_dir = os.path.join(dataset_dir, split, 'images')
    label_dir = os.path.join(dataset_dir, split, 'labels')

    # Get the list of image and label files
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # Randomly select a subset of images
    selected_indices = random.sample(range(len(image_files)), min(num_images, len(image_files)))

    # Plot the selected images with their keypoints
    for idx in selected_indices:
        image_path = os.path.join(image_dir, image_files[idx])
        label_path = os.path.join(label_dir, label_files[idx])

        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Load the labels
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Create a new figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # plt.axis('off')

        # Plot keypoints for each object in the image
        for label in labels:
            data = list(map(float, label.strip().split()))
            class_id, x_center, y_center, width, height = data[:5]
            keypoints = data[5:]

            # Calculate bounding box coordinates
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)

            # Draw bounding box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))

            # Plot keypoints
            for j, i in enumerate(range(0, len(keypoints), 3)):
                x, y, visibility = keypoints[i:i+3]
                x = int(x * img_width)
                y = int(y * img_height)

                if plot_visible_only and visibility < visibility_threshold:
                    continue

                color = 'g' if visibility >= visibility_threshold else 'y'
                plt.text(x, y, j)
                plt.plot(x, y, 'o', color=color, markersize=5)

        plt.title(f'Image: {image_files[idx]}')
        plt.show()

def analyze_yolo_dataset(dataset_dir):
    """
    Analyzes the number of images in each split of a YOLOv8 dataset,
    creates a DataFrame with counts and percentages, and plots a pie chart.

    Args:
        dataset_dir (str): Path to the directory containing the dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the analysis results.
    """
    if not os.path.exists(dataset_dir):
        return "Invalid dataset path."
    
    # Define the splits to analyze
    splits = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    
    if 'train' not in splits:
        return "Invalid dataset structure."

    # Initialize a dictionary to store the image counts
    image_counts = {}

    # Count images in each split
    for split in splits:
        split_dir = os.path.join(dataset_dir, split, 'images')
        if os.path.exists(split_dir):
            image_files = [f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            image_counts[split] = len(image_files)
        else:
            image_counts[split] = 0

    # Calculate total images and percentages
    total_images = sum(image_counts.values())
    percentages = {split: count / total_images * 100 for split, count in image_counts.items()}

    # Create a DataFrame
    df = pd.DataFrame({
        'Split': splits,
        'Image Count': [image_counts[split] for split in splits],
        'Percentage': [percentages[split] for split in splits]
    })

    # Sort the DataFrame by Image Count in descending order
    df = df.sort_values('Image Count', ascending=False).reset_index(drop=True)

    # Format the Percentage column
    df['Percentage'] = df['Percentage'].apply(lambda x: f"{x:.2f}%")

    # Create a pie chart
    plt.figure(figsize=(3, 2))
    plt.pie(df['Image Count'], labels=df['Split'], autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Images Across Splits')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Display the plot
    plt.show()

    return df

# Plots images within a path to a 4xn grid using Matplotlib
def plot_4x(directory):
    files = os.listdir(directory)
    png_files = [file for file in files if file.lower().endswith('.png') or file.lower().endswith('.jpg')]

    num_images = len(png_files)
    cols = 4
    rows = (num_images + cols - 1) // cols

    dpi = 400

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), dpi=dpi)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image_path = os.path.join(directory, png_files[i])
            ax.imshow(plt.imread(image_path))
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def clean_img_dir_and_labels(image_dir):
    # Define acceptable image and label file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    label_extensions = {'.txt'}
    
    # Lists to keep track of corrupted files
    corrupted_imgs = []
    corrupted_labels = []
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if the file is an image
            if file_ext in image_extensions:
                ann_path = file_path.replace('images', 'labels').replace(file_ext, '.txt')
                try:
                    with Image.open(file_path) as img:
                        img.load()  # Ensure image can be loaded
                except (IOError, OSError) as e:
                    corrupted_imgs.append(file_path)
                    if os.path.exists(ann_path):
                        os.remove(ann_path)
                    os.remove(file_path)
                    continue
                try:
                    if os.path.exists(ann_path):
                        with open(ann_path, 'r') as lbl:
                            annotation = np.loadtxt(lbl)
                            if len(annotation) <= 0:
                                corrupted_labels.append(ann_path)
                                os.remove(file_path)
                                os.remove(ann_path)
                except (IOError, OSError, ValueError) as e:
                    corrupted_labels.append(ann_path)
                    if os.path.exists(ann_path):
                        os.remove(ann_path)
                    os.remove(file_path)
            elif file_ext in label_extensions:
                # Check if the label file corresponds to an image
                img_path = file_path.replace('labels', 'images').replace(file_ext, '.jpg')
                if not os.path.exists(img_path):
                    corrupted_labels.append(file_path)
                    os.remove(file_path)
                
    return (
        "The following corrupted image entries have been removed:\n" + "\n".join(corrupted_imgs),
        "The following corrupted label entries have been removed:\n" + "\n".join(corrupted_labels)
    )
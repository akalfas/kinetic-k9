import cv2
import glob
import os

def process_grids():
    input_dir = "datasets/generated"
    output_base = "datasets/mouth_cls/train"
    
    # Ensure targets exist
    lick_dir = os.path.join(output_base, "licking")
    closed_dir = os.path.join(output_base, "closed")
    os.makedirs(lick_dir, exist_ok=True)
    os.makedirs(closed_dir, exist_ok=True)
    
    grid_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not grid_files:
        print("No .jpg files found in datasets/generated")
        return
        
    count = 0
    for file_path in grid_files:
        img = cv2.imread(file_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Calculate midpoints (accounting for potential grid lines)
        mid_y = h // 2
        mid_x = w // 2
        
        # Quarter Slices
        top_left = img[0:mid_y, 0:mid_x]
        top_right = img[0:mid_y, mid_x:w]
        bottom_left = img[mid_y:h, 0:mid_x]
        bottom_right = img[mid_y:h, mid_x:w]
        
        # Resize aggressively to our Native 160x160 resolution
        tl_rsz = cv2.resize(top_left, (160, 160))
        tr_rsz = cv2.resize(top_right, (160, 160))
        bl_rsz = cv2.resize(bottom_left, (160, 160))
        br_rsz = cv2.resize(bottom_right, (160, 160))
        
        # Top Row is Positive Class (Licking)
        cv2.imwrite(os.path.join(lick_dir, f"syn_grid_{count}_TL.jpg"), tl_rsz)
        cv2.imwrite(os.path.join(lick_dir, f"syn_grid_{count}_TR.jpg"), tr_rsz)
        
        # Bottom Row is Negative Class (Closed)
        cv2.imwrite(os.path.join(closed_dir, f"syn_grid_{count}_BL.jpg"), bl_rsz)
        cv2.imwrite(os.path.join(closed_dir, f"syn_grid_{count}_BR.jpg"), br_rsz)
        
        print(f"Processed grid: {os.path.basename(file_path)} -> Extracted 4 normalized images.")
        count += 1
        
    print(f"Successfully unpacked {count * 4} synthetic training samples into the pipeline!")

if __name__ == "__main__":
    process_grids()

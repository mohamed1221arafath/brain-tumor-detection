import os
import cv2
import matplotlib.pyplot as plt

# Step 1: List folders
for split in ["Training", "Testing"]:
    print(split, "->", os.listdir(split))

# Step 2: Count images per class
for split in ["Training", "Testing"]:
    print(f"\n{split}:")
    for tumor_type in os.listdir(split):
        path = os.path.join(split, tumor_type)
        count = len(os.listdir(path))
        print(f"  {tumor_type}: {count} images")

# Step 3: Show a sample image
img_path = os.path.join("Training", "glioma", os.listdir("Training/glioma")[0])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Sample Glioma MRI")
plt.axis("off")
plt.show()
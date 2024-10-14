import os
import shutil

root = './data/flower_photos'

save_path = './data/flower_data'

# read all imgs path
classes = os.listdir(root)
print(classes)

all_images = []
class_to_idx = {flower_class: i for i, flower_class in enumerate(classes)}

for flower_class in classes:
    img_paths = os.listdir(os.path.join(root, flower_class))  # [img_name, img_name, ...]
    print(flower_class, len(img_paths))
    for img in img_paths:
        # tmp = os.path.join(root, flower_class, img)
        all_images.extend( [os.path.join(root, flower_class, img)])
    
print("total imgs:", len(all_images))

# split to train, val and test
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(all_images, test_size=0.2, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(len(train_data), len(val_data), len(test_data))

os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'val'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)

# Function to copy images to the respective directories
def save_images(image_list, directory):
    for image_path in image_list:
        shutil.copy(image_path, directory)


# save label
def save_class_info(image_list, save_path):
    # image_list: full path
    with open(save_path, 'w') as f:
        for img_path in image_list:
            tmp1 = os.path.dirname(img_path) #  gets the directory of the file
            tmp2 = os.path.basename(tmp1) # gets the last directory name
            class_label = os.path.basename(os.path.dirname(img_path))
            f.write(f"{os.path.basename(img_path)}\t{class_label}\n")

        

''' 
# Save the datasets
save_images(train_data, os.path.join(save_path, 'train'))
save_images(val_data, os.path.join(save_path, 'val'))
save_images(test_data, os.path.join(save_path, 'test'))
'''
save_class_info(train_data, os.path.join(save_path, 'train.txt'))
save_class_info(val_data, os.path.join(save_path, 'val.txt'))
save_class_info(test_data, os.path.join(save_path, 'test.txt'))

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def main(data_dir, output_dir, val_size):
    # Create Dataset_Classifier directory
    dataset_classifier_dir = os.path.join(output_dir, 'Dataset_Classifier')
    os.makedirs(dataset_classifier_dir, exist_ok=True)

    # Create train and val directories
    train_dir = os.path.join(dataset_classifier_dir, 'train')
    val_dir = os.path.join(dataset_classifier_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get the list of classes (subdirectories)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # For each class
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        # Get the list of images
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        # Check if the list of images is not empty
        if images:
            # Split the images into train and val sets
            train_images, val_images = train_test_split(images, test_size=val_size)

            # Create the class subdirectories in the train and val directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # Move the train images to the train directory
            for image in train_images:
                src_path = os.path.join(class_dir, image)
                dst_path = os.path.join(train_class_dir, image)
                shutil.move(src_path, dst_path)

            # Move the val images to the val directory
            for image in val_images:
                src_path = os.path.join(class_dir, image)
                dst_path = os.path.join(val_class_dir, image)
                shutil.move(src_path, dst_path)
        else:
            print(f'Empty subdirectory: {class_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Path to the source data directory')
    parser.add_argument('output_dir', type=str,
                        help='Path to the output directory')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Proportion of the data to include in the validation set')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.val_size)
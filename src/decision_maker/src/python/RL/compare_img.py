from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import shutil


def calculate_image_similarity(img_path1, img_path2):
    # Load the InceptionV3 model, 3 x 3 filter
    model = InceptionV3(weights='imagenet', include_top=False)

    # Load the two images
    # Note that the target size for InceptionV3 is 299x299
    img1 = image.load_img(img_path1, target_size=(299, 299))
    img2 = image.load_img(img_path2, target_size=(299, 299))

    # Convert the images to numpy arrays and preprocess them
    img1 = image.img_to_array(img1)
    img2 = image.img_to_array(img2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)

    # Extract the features from the images
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Calculate the similarity between the two sets of features
    similarity = cosine_similarity(features1.reshape(
        1, -1), features2.reshape(1, -1))[0][0]

    return similarity


if __name__ == "__main__":
    # parameters
    gazebo_env = 'aws_house'
    # train_data, dqn, ddqn, dueling dqn, dueling ddqn
    algo = 'train_data'
    repeat_count = 5

    base_folder_path = f'/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/{gazebo_env}/{algo}/{repeat_count}'
    original_img = "/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/original/aws_house.png"

    # Create a new directory to store similar images
    similar_images_folder = f'/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/similar_images/{algo}'
    os.makedirs(similar_images_folder, exist_ok=True)

    # List of sub-folders to check
    sub_folders = ['completed']

    # Loop over sub-folders
    for sub_folder in sub_folders:
        folder_path = os.path.join(base_folder_path, sub_folder)

        # Loop over all png files in the folder_path
        for img2_path in glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True):
            # Calculate similarity
            similarity = calculate_image_similarity(original_img, img2_path)
            similarity = similarity * 100
            similarity = round(similarity)

            # If similarity is more than 75%
            if similarity >= 75:
                # Construct new path for the image
                new_img_path = os.path.join(
                    similar_images_folder, os.path.basename(img2_path))

                # Copy the image to the new directory
                shutil.copy2(img2_path, new_img_path)

                print(f'Image {img2_path} copied to {new_img_path}')

            # Print the similarity score and the image path
            print(
                f'Similarity between {original_img} and {img2_path}: {similarity}%')

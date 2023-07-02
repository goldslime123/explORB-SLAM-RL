# libraries
# libraries
import numpy as np
import os
import glob
import shutil
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity


def calculate_image_similarity_inception(img_path1, img_path2):
    # Load the InceptionV3 model
    model = InceptionV3(weights='imagenet', include_top=False)

    # Load and preprocess the images
    img1 = image.load_img(img_path1, target_size=(299, 299))
    img2 = image.load_img(img_path2, target_size=(299, 299))
    img1 = image.img_to_array(img1)
    img2 = image.img_to_array(img2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)

    # Extract features using InceptionV3
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1.reshape(
        1, -1), features2.reshape(1, -1))[0][0]
    return similarity


def calculate_image_similarity_vgg(img_path1, img_path2):
    # Load the VGG16 model
    model = VGG16(weights='imagenet', include_top=False)

    # Load and preprocess the images using VGG preprocessing
    img1 = image.load_img(img_path1, target_size=(224, 224))
    img2 = image.load_img(img_path2, target_size=(224, 224))
    img1 = image.img_to_array(img1)
    img2 = image.img_to_array(img2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    img1 = vgg_preprocess_input(img1)
    img2 = vgg_preprocess_input(img2)

    # Extract features using VGG16
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1.reshape(
        1, -1), features2.reshape(1, -1))[0][0]
    return similarity


def calculate_image_similarity_resnet(img_path1, img_path2):
    # Load the ResNet50 model
    model = ResNet50(weights='imagenet', include_top=False)

    # Load and preprocess the images using ResNet preprocessing
    img1 = image.load_img(img_path1, target_size=(224, 224))
    img2 = image.load_img(img_path2, target_size=(224, 224))
    img1 = image.img_to_array(img1)
    img2 = image.img_to_array(img2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    img1 = resnet_preprocess_input(img1)
    img2 = resnet_preprocess_input(img2)

    # Extract features using VGG16
    features1 = model.predict(img1)
    features2 = model.predict(img2)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1.reshape(
        1, -1), features2.reshape(1, -1))[0][0]
    return similarity



if __name__ == "__main__":
    # Parameters
    gazebo_env = 'aws_house'
    algos = ['dqn', 'ddqn', 'dueling_dqn', 'dueling_ddqn']
    repeat_counts = [5, 10, 15, 20]

    # List of comparison models - vgg, resnet, inception
    cnn_models = ['vgg16', 'resnet50', 'inceptionV3']

    best_picture = '11-501a_completed'

    for algo in algos:
        for repeat_count in repeat_counts:
            base_folder_path = f'/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/rviz_results/{gazebo_env}/{algo}/{repeat_count}'
            original_img = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/similar_images/base/{best_picture}.png"

            for cnn_model in cnn_models:
                # Create a new directory to store similar images based on the CNN model chosen
                similar_images_folder = f'/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/similar_images/{gazebo_env}/{algo}/{cnn_model}'
                os.makedirs(similar_images_folder, exist_ok=True)

                sub_folders = ['completed']

                for sub_folder in sub_folders:
                    folder_path = os.path.join(base_folder_path, sub_folder)

                    for img2_path in glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True):
                        if cnn_model == 'inception':
                            similarity = calculate_image_similarity_inception(original_img, img2_path)
                        elif cnn_model == 'vgg':
                            similarity = calculate_image_similarity_vgg(original_img, img2_path)
                        else:
                            similarity = calculate_image_similarity_resnet(original_img, img2_path)

                        if similarity is not None:
                            similarity = similarity * 100
                            similarity = round(similarity)

                        if similarity >= 80:
                            img_filename = os.path.basename(img2_path)
                            img_name, img_ext = os.path.splitext(img_filename)
                            img_name = img_name.replace('_completed', '')
                            new_img_filename = f"{img_name}_{similarity}%{img_ext}"
                            new_img_path = os.path.join(similar_images_folder, new_img_filename)

                            shutil.copy2(img2_path, new_img_path)
                            print(f'Image {img2_path} copied to {new_img_path}')

                        print(f'Similarity between {original_img} and {img2_path}: {similarity}%')

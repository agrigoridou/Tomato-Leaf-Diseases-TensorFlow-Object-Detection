# Tomato-Leaf-Diseases-TensorFlow-Object-Detection

![image](https://github.com/athanasiagrigoridou/Tomato-Leaf-Diseases-TensorFlow-Object-Detection/assets/167294620/90b82183-2f6c-40c4-88bb-85281501124e)


# Tomato-Leaf-Diseases-TensorFlow-Object-Detection

This repository contains code for detecting tomato leaf diseases using TensorFlow Object Detection API. The project includes scripts to set up the environment, train a custom object detection model using TensorFlow 2.x, convert the trained model to TensorFlow Lite format, and perform inference using the TFLite model.

## Installation

1. Clone the TensorFlow models repository:

    ```bash
    !pip uninstall Cython -y # Temporary fix for "No module named 'object_detection'" error
    !git clone --depth 1 https://github.com/tensorflow/models
    ```

2. Set up dependencies and environment:

    ```bash
    !pip install pyyaml==5.3
    !pip install /content/models/research/
    !pip install tensorflow==2.8.0
    !pip install tensorflow_io==0.23.1
    # Additional setup steps as per your environment requirements
    ```

3. Upload your image dataset and prepare training data:

    ```python
    from google.colab import drive
    drive.mount('/content/gdrive')

    !cp /content/gdrive/MyDrive/ColabNotebooks/data.zip /content
    !cp /content/gdrive/MyDrive/ColabNotebooks/images.zip /content
    !unzip -q data.zip -d /content
    !unzip -q images.zip -d /content
    ```

4. Customize the training configuration:

    ```python
    # Modify model configurations, batch size, and other parameters
    ```

5. Train the custom TensorFlow Lite detection model:

    ```python
    # Run training and save the model in TensorFlow Lite format
    ```

6. Test the TensorFlow Lite model:

    ```python
    # Perform inference with the TFLite model on test images
    ```
    
![leafDeseases](https://github.com/athanasiagrigoridou/Tomato-Leaf-Diseases-TensorFlow-Object-Detection/assets/167294620/b7cdaafb-83ff-4ca0-8f09-a131aeed9408)



7. Calculate mAP (mean Average Precision):

    ```bash
    # Run mAP calculation on detection results
    ```

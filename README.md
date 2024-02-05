# Using-Action-Recognition-to-Crack-reCAPTCHA

## Summary
<!-- Summary -->
The goal of this project is to distinguish between machine and human beings when using reCAPTCHA with action. We implement the most powerful attacker and defense strategies to find out what degree of a cropped image will successfully distinguish machine and human beings. Also, to produce the cropped image for defense, we introduce the Grad-CAM technique to know where the reference of the attacker is and crop these places. Ultimately, we find the most effective method to defend the attacker, resulting in the attacker having 98%, 60%, and 65% accuracy for 3 categories.


## Technique usage
<!-- Technique usage -->
### Transfer-Learning
This project adopts Transfer Learning [1] and utilizes ResNet152 as the pretrained model for training. The underlying concept involves initially fixing the convolutional layers in the neural network architecture to enable the model to extract features from images, such as edges, corners, and contours. Once the model acquires the ability to obtain fundamental information about the underlying structure of images, the classifier is then retrained. Subsequently, with a lower learning rate, the entire model undergoes fine-tuning to adapt to new data, ultimately enhancing the overall learning performance of the model.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/fb6e6a00dec15c5fb1f5077675bba0d0cc2e5bb2/image%20reference/Transfer-Learning.png)


The advantage of using Transfer Learning is that it eliminates the need to gather a large amount of data for model training. This is particularly beneficial in the early stages of the project when there might not be an extensive dataset available, yet it allows the model to achieve a decent recognition rate. Additionally, there is no requirement to invest a significant amount of time in training the model from scratch. This aspect proves to be time-saving, especially in the later stages of the project when continuous optimization of the model is undertaken.


### Grad-CAM
We utilized the Grad-CAM [2] technique, based on the CAM [3] technology, to aid in understanding the regions of focus for the model. The working principle involves a concept similar to back-propagation, where the weights of pixels in various regions of the image are computed at each layer. These weights are then stacked on their corresponding pixels, ultimately generating a class activataion map (as shown in the diagram below). This technique provides insights into the specific areas the model emphasizes during its operations.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/fb6e6a00dec15c5fb1f5077675bba0d0cc2e5bb2/image%20reference/Grad-CAM.png)

By examining the class activataion map, it becomes straightforward to assess the significance of a particular region in influencing the prediction results. In this project, we rely on these maps to evaluate whether the model training outcomes align with the expected attention areas. Additionally, the maps serve as a basis for determining the criteria for image cropping, which is crucial for subsequent training phases.


## DataSet example
<!-- Dataset -->

## Models
<!-- Models 
    process
    basic one
    gray
    level 1 
    level 2
    level 3 -->
### Training process

### Basic model

### Gray

### Level 1

### level 2

### level 3
<!-- conclusion -->

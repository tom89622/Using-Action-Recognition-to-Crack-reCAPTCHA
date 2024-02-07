# Using-Action-Recognition-to-Crack-reCAPTCHA

[Complete report](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/ab04b7a8758076a8a95ff77e174bc60c7b7a5709/%E5%88%A9%E7%94%A8%E5%8B%95%E4%BD%9C%E8%BE%A8%E8%AD%98%E7%A0%B4%E8%A7%A3_%E5%A0%B1%E5%91%8A.pdf)
## Summary
<!-- Summary -->
The goal of this project is to distinguish between machine and human beings when using reCAPTCHA with action. We implement the most powerful attacker and defense strategies to find out what degree of a cropped image will successfully distinguish machine and human beings. Also, to produce the cropped image for defense, we introduce the Grad-CAM technique to know where the reference of the attacker is and crop these places. Ultimately, we find the most effective method to defend the attacker, resulting in the attacker having 98%, 60%, and 65% accuracy for 3 categories.


## Technique usage
<!-- Technique usage -->
### Transfer-Learning
This project adopts Transfer Learning [1] and utilizes ResNet152 as the pretrained model for training. The underlying concept involves initially fixing the convolutional layers in the neural network architecture to enable the model to extract features from images, such as edges, corners, and contours. Once the model acquires the ability to obtain fundamental information about the underlying structure of images, the classifier is then retrained. Subsequently, with a lower learning rate, the entire model undergoes fine-tuning to adapt to new data, ultimately enhancing the overall learning performance of the model.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/0f26738022b270d495537fc10fbfd94c2d622227/image%20reference/Transfer%20Learning.png)


The advantage of using Transfer Learning is that it eliminates the need to gather a large amount of data for model training. This is particularly beneficial in the early stages of the project when there might not be an extensive dataset available, yet it allows the model to achieve a decent recognition rate. Additionally, there is no requirement to invest a significant amount of time in training the model from scratch. This aspect proves to be time-saving, especially in the later stages of the project when continuous optimization of the model is undertaken.


### Grad-CAM
We utilized the Grad-CAM [2] technique, based on the CAM [3] technology, to aid in understanding the regions of focus for the model. The working principle involves a concept similar to back-propagation, where the weights of pixels in various regions of the image are computed at each layer. These weights are then stacked on their corresponding pixels, ultimately generating a class activataion map (as shown in the diagram below). This technique provides insights into the specific areas the model emphasizes during its operations.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/0f26738022b270d495537fc10fbfd94c2d622227/image%20reference/Grad-CAM.png)

By examining the class activataion map, it becomes straightforward to assess the significance of a particular region in influencing the prediction results. In this project, we rely on these maps to evaluate whether the model training outcomes align with the expected attention areas. Additionally, the maps serve as a basis for determining the criteria for image cropping, which is crucial for subsequent training phases.


## DataSet example
<!-- Dataset -->
[Level 1](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/tree/ab04b7a8758076a8a95ff77e174bc60c7b7a5709/level%201%20picture%20example)

[Level 2](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/tree/ab04b7a8758076a8a95ff77e174bc60c7b7a5709/level%202%20picture%20example)

[Level 3](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/tree/ab04b7a8758076a8a95ff77e174bc60c7b7a5709/level%203%20picture%20example)



## Models
<!-- Models 
    process
    basic one
    gray
    level 1 
    level 2
    level 3 -->
### Training process
The objective of model training is to standardize the subject and emphasize the model's focus on human body movements. Therefore, experimental image cropping is employed following the workflow depicted in the following diagram. The aim is to train the model continuously in an attempt to push it closer to the recognition limit perceivable by humans. Furthermore, the goal is to identify the extent of image cropping that is sufficient to distinguish between machine and human recognition. This degree of cropping will be referred to as the 'sweet spot' of recognition in this project.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/0f26738022b270d495537fc10fbfd94c2d622227/image%20reference/Training%20diagram.png)

Therefore, in this phase, multiple versions will be developed following the workflow diagram until the 'sweet spot' is identified. For each version, the test images will reference the Grad-CAM heatmap generated by that version. This will help in summarizing the locations where the machine focuses its attention. These summaries will serve as the basis for cropping, with the aim of testing whether the model can still successfully recognize images after removing parts of the areas it concentrates on.


### Basic model
This version involves removing the basketball hoop factor from the training and validation sets. The objective is to achieve a generalization effect by including basketball occurrences across all categories, thereby eliminating specific features associated with the hoop. The expected outcome is for the model to rely on identifying the body parts of the subject.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/b136581dea22f73a1aa592c846f50b553d732d70/image%20reference/original%20image%20(Basic%20model).png)

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/b136581dea22f73a1aa592c846f50b553d732d70/image%20reference/statistic%20(Basic%20model).png)
Table 1

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/f53b65cc55373909dbf7022fc0a22e9e8226b953/image%20reference/Grad-CAM%20image%20(Basic%20model).png)

Test results after training:

> **Dribble: 100%,  Dunk: 85.3%, Shoot: 95.7%**

The following are the features summarized based on the Grad-CAM Picture:

* Dribble: both sides of the head to the shoulders, ball
* Dunk: wrist, lower edge of the ball, court + lighting
* Shoot: arm, elbow, ball

From the results, it is known that the goal of focusing on the main character has been achieved, and the model no longer pays attention to the basket area,
Therefore, this version is chosen as the beginning of the flow chart. The next versions will continue to test the pictures after cropping,
If the accuracy is poor, then add them to the training set and retrain.


### Level 1
This version was initially tested using a dataset, as shown in the example image below. The dataset had already cropped some of the model's attention areas. The results are shown in Table 2, indicating a noticeable need for improvement in accuracy. Consequently, the dataset was added to the training set and retrained, as shown in Table 3.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/test%20table%20(Level%201%20model).png)

Table 2

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/original%20image%20(Level%201%20model).png)

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/statistic%20(Level%201%20model).png)

Table 3

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/Grad-CAM%20image%20(Level%201%20model).png)

<!-- level one images -->

Test results after training:

> **Dribble: 100%,  Dunk: 97%, Shoot: 87%**

The following are the features summarized based on the Grad-CAM Picture:

* Dribble: Head to shoulders on both sides, and the ball.
* Dunk: Arm, lower edge of the ball, and head.
* Shoot: Arm, underarm.

From the results, it is evident that at this stage, the model has learned features similar to those described above and remains comparable to the previous version, resulting in very high accuracy. The next version will continue testing by cropping around the model's attention areas.

### level 2
Following the workflow diagram, images were cropped based on their attention areas and tested for accuracy. As shown in Table 4, the results were unsatisfactory, indicating a need for further improvement in accuracy. Consequently, the dataset was added to the training set and retrained, as shown in Table 5.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/test%20table%20(Level%202%20model).png)

Table 4

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/original%20image%20(Level%202%20model).png)

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/statistic%20(Level%202%20model).png)

Table 5

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/Grad-CAM%20image%20(Level%202%20model).png)
<!-- level two images -->

Test results after training:

> **Dribble: 99%,  Dunk: 97%, Shoot: 95%**

The following are the features summarized based on the Grad-CAM Picture:

* Dribble: Team name and number on the chest, ball, arm.
* Dunk: Arm, wrist, relatively scattered (such as court, lighting, etc.).
* Shoot: Arm, armpit.

This version achieved extremely high testing accuracy, indicating that images at the same level can be successfully recognized. There is still room for further cropping in the regions, and the model's performance remains within the range recognizable by humans. Therefore, the experiment proceeds to the next version.


### level 3

Following the workflow diagram, testing was conducted using images cropped based on their attention areas. The performance was still unsatisfactory, as shown in Table 6, prompting another round of retraining.

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/test%20table%20(Level%203%20model).png)

Table 6

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/original%20image%20(Level%203%20model).png)

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/statistic%20(Level%203%20model).png)

Table 7

![image](https://github.com/tom89622/Using-Action-Recognition-to-Crack-reCAPTCHA/blob/042e2120b8ce9ce5e9d1672d21d529f67bde6ef9/image%20reference/Grad-CAM%20image%20(Level%203%20model).png)

<!-- level three images -->

Test results after training:

> **Dribble: 98%,  Dunk: 60%, Shoot: 65%**

The following are the features summarized based on the Grad-CAM Picture:

* Dribble: Team name and number on the chest, arm.
* Dunk: Face, court background, wrist.
* Shoot: Arm, armpit, face.

In this version, the testing accuracy is exceptionally high only for the dribble category, while the other two have significantly decreased compared to the previous versions. The images used in this version have gradually moved towards areas that are challenging for human recognition. Continuous cropping may render them difficult for humans to discern, and the machine may not be able to enhance accuracy through further training. Therefore, this version is considered the final iteration. The discussion on human and machine recognition rates will be included in the final report.


## Conclusion
<!-- conclusion -->
In the results of the final version, we believe that the 'sweet spot' for 'Dunk' and 'Shoot' is to crop the images to retain 'partial arms' and 'chest areas.' The experiments conducted in this phase aimed to enhance the model's accuracy as much as possible under conditions where humans can recognize. Throughout the experiments, we also influenced the areas of focus. However, it can be observed that the testing accuracy for dribbling remains high, indicating that the sweet spot for dribbling has not been reached. We believe that the actions in this category show significant differences after cropping compared to the other two, making them easier to recognize. This phenomenon is due to the initial design oversight in anticipating such a substantial similarity between 'Dunk' and 'Shoot' after cropping.
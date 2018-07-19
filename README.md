# Car-Plate-Image-Recognition
Recognize the plate numbers with computer vision and image processing techniques in MATLAB

Please run the function main.m to detect and process input images. By changing the path to different input photos from folder "Plate Number", you can recognize different plates for testing. 

As for some computer, chinese characters cannot be properly displayed, so this function for now recognize the plate characters from the second digit. however, you can recognize the first digit by changing the loop conditions near line 449. The logic of recognizing the characters is the same as doing it for letters & numbers.

Summary of steps:

1. Read in the original image
2. Use multiple image processing techniques to calculate the area of plate
3. Elimiate and clean unnecessary small parts and remember the x y values
    for plate only
4. Go back to original photo and seperate each number or character 
5. Compare the character to images in our database
6. Apply algorithm and calculation to find the image with highest
    similarities of features (KNN or other machine learning techniques can be
    used here)
7. Control the accuracy_score to be more than 97%

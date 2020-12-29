# MotionBlurDetection

AGH Computer Vision course project. 

There are 2 detectors:
  * CNN model trained on 7k+ images from Intel Image Classification dataset;
  * Laplacian based algorithm for blur detection;

Blurred images were created by applying motion blur to all images from VOC2010 dataset. Then, in order to achieve partially blurred images, the final image is created by choosing pieces from original (not blurred) image and its blurred version with likelihood of 0.5. The pieces are of size 30x30 taken from image of size 150x150, thus each image was cut into 5 rows and 5 columns. CNN model is taking each piece as an input in batch of size 5x5 (one whole image) as shown in Figure 1 - original image on the left and partially blurred on the right. The labels and predictions for single image is a list of predictions/labels for each piece.

![Original image](../master/results/original.jpg)
![Blurred image](../master/results/blurred.jpg)

## Acknowledgements

Puneet Bansal for sharing image dataset on https://www.kaggle.com/puneet6060/intel-image-classification?fbclid=IwAR3sGlvbM9Tfn1a01bNctiKtNW9GhkPAs0WvhPkmKVsYdALf94cuxRehSzk.

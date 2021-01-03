import cv2
from data_preparation import ImageRegion, get_regions_with_labels


class MotionBlurDetectionLaplacian:

    def __init__(self, threshold=1000.):
        self.threshold = threshold
        self.images_regions = get_regions_with_labels()

    @staticmethod
    def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def predict(self):
        predictions = []
        for img_region in self.images_regions:
            image = img_region.get_crop()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[1] > 3 else image
            fm = self.variance_of_laplacian(image)
            predictions.append((1, fm)) if fm < self.threshold else predictions.append((0, fm))
        return predictions

    def evaluate(self, image_regions: [ImageRegion]):
        evaluations = []
        predictions = self.predict()
        for i, predicted_label, _ in enumerate(predictions):
            evaluations.append(predicted_label == image_regions[i].label)
        return evaluations

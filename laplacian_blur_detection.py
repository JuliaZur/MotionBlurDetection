import cv2
from data_preparation import ImageRegion, get_regions_with_labels, get_img_regions


class MotionBlurDetectionLaplacian:

    def __init__(self, threshold=1700.):
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

    def predict_single_crop(self, image_region: ImageRegion):
        image = image_region.get_crop()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[1] > 3 else image
        fm = self.variance_of_laplacian(image)
        prediction = 1 if fm < self.threshold else 0
        return prediction

    def predict_single_image(self, image_path):
        predictions = []
        image_regions = get_img_regions(image_path)
        for image in image_regions:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[1] > 3 else image
            fm = self.variance_of_laplacian(image)
            predictions.append((1, fm)) if fm < self.threshold else predictions.append((0, fm))
        return predictions

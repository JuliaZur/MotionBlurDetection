import cv2


class MotionBlurDetectionLaplacian:

    def __init__(self, threshold=1000.):
        self.threshold = threshold

    @staticmethod
    def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def predict(self, images_paths):
        predictions = []
        blur_level = []
        for path in images_paths:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[1] > 3 else image
            fm = self.variance_of_laplacian(image)
            predictions.append(1) if fm < self.threshold else predictions.append(0)
            blur_level.append(fm)
        return predictions, blur_level

    def evaluate(self, images_paths, images_labels):
        evaluations = []
        predictions, _ = self.predict(images_paths)
        for i in range(len(images_paths)):
            if 1 in images_labels[i]:
                evaluations.append(1) if predictions[i] == 1 else evaluations.append(0)
            else:
                evaluations.append(1) if predictions[i] == 0 else evaluations.append(0)
        return evaluations

from cnn_blur_detection import MotionBlurDetectionCNN
from data_preparation import get_img_regions, TARGET_SHAPE, BLUR_COLS, BLUR_ROWS, TARGET_W, TARGET_H
import cv2


def mask_on_image(img_path, mask, show_img=False):
    result = cv2.imread(img_path)
    if result.size != TARGET_SHAPE:
        result = cv2.resize(result, TARGET_SHAPE)
    regions = get_img_regions(img_path)
    for region in regions:
        left, top, right, bottom = region.get_boundaries()
        left += 1
        top += 1
        bottom -= 1
        right -= 1
        label = mask[region.row][region.col]
        if label == 1:
            color = (0, 255, 0)
            result[left:right, top] = color
            result[left:right, bottom] = color
            result[left, top:bottom] = color
            result[right, top:bottom] = color

    if show_img:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 500, 500)
        cv2.imshow('result', result)
        cv2.waitKey(0)

        # closing all open windows
        cv2.destroyAllWindows()
    return result

# USAGE
# img_path = YOUR_PATH
# cnn = MotionBlurDetectionCNN()
# cnn.load_model(YOUR_MODEL)
# p = cnn.predict(img_path)
# mask_on_image(img_path, p)
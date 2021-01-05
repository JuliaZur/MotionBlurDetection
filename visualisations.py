from cnn_blur_detection import MotionBlurDetectionCNN
from data_preparation import get_img_regions, TARGET_SHAPE, BLUR_COLS, BLUR_ROWS, TARGET_W, TARGET_H
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, roc_curve, accuracy_score, \
    precision_score, recall_score, f1_score, classification_report
import pandas as pd
import seaborn as sns


# USAGE
# img_path = YOUR_PATH
# cnn = MotionBlurDetectionCNN()
# cnn.load_model(YOUR_MODEL)
# p = cnn.predict(img_path)
# mask_on_image(img_path, p)
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
# from cnn_blur_detection import MotionBlurDetectionCNN
# cnn = MotionBlurDetectionCNN()
# cnn.load_model('motionblur_v1.h5')
# X_test, y_test = cnn.validation_generator.get_all()
# y_probs, y_pred = cnn.predict_crops(X_test)
# roc_pr_rec_plots(y_test, y_pred, y_probs)
def roc_pr_rec_plots(y_true, y_predicted, y_probs):
    roc_pr_fig = plt.figure(figsize=(15, 10))
    probs = y_probs[:, 1]

    # --------- ROC ------------
    ax_roc_fig = roc_pr_fig.add_subplot(221)
    ax_roc_fig.set_title('ROC')
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = auc(fpr, tpr)
    ax_roc_fig.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

    ax_roc_fig.plot([0, 1], [0, 1], 'r--')

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')

    # --------- PrecisionRecall ------------

    ax_pr_fig = roc_pr_fig.add_subplot(222)
    ax_pr_fig.set_title('PrecisionRecall')
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, y_predicted)
    ax_pr_fig.plot(recall, precision, label='AP = {:.2f}'.format(ap))
    ax_pr_fig.plot([0, 1], [0.5, 0.5], 'r--')

    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([-0.1, 1.1])
    plt.ylim([0.4, 1.1])

    plt.show()


def confusion_matrix(y_true, y_pred):
    data = {'y_true': y_true,
            'y_pred': y_pred
            }

    df = pd.DataFrame(data, columns=['y_true', 'y_pred'])
    confusion_matrix = pd.crosstab(df['y_true'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'])

    sns.heatmap(confusion_matrix, annot=True)
    plt.show()


def show_scores(y_true, y_pred, name):
    print(f'''
    {name} metrics:
          Accuracy: \t{accuracy_score(y_true, y_pred)}
          Precision: \t{precision_score(y_true, y_pred)}
          Recall: \t{recall_score(y_true, y_pred)}
          F1 score: \t{f1_score(y_true, y_pred)}
          ''')

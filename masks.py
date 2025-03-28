from sklearn.cluster import KMeans
from config import *


def skin_mask(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_flattened = img.reshape((-1, 3))

    k = 2
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flattened)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    skin_cluster_index = np.argmax(np.mean(cluster_centers, axis=1))
    mask = (labels == skin_cluster_index).astype(np.uint8)

    mask = mask.reshape(frame.shape[:2])

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def red_square_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    return red_mask

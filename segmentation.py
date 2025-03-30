import cv2
import numpy as np
from sklearn.cluster import KMeans


def segment_person_kmeans(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    lab_2d = lab.reshape(-1, 3)

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(lab_2d)
    centers = kmeans.cluster_centers_

    bg_label = 0 if np.sum(centers[0]) < np.sum(centers[1]) else 1
    person_mask = (labels != bg_label).astype(np.uint8).reshape(h, w)

    kernel = np.ones((5, 5), np.uint8)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel, iterations=2)


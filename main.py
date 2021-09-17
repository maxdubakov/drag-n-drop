import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import config as cf
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, cf.width)
cap.set(4, cf.height)

detector = HandDetector(detectionCon=cf.detectionCon)   # detection confidence


def get_lm_list(_hands):
    if _hands:
        return _hands[0]["lmList"]
    return None


class DragRect(object):

    IS_MOVING = False
    MOVING_RECT = None

    def __init__(self, pos_center, size=(200, 200)):
        self._pos_center = pos_center
        self._size = size
        self._is_moving = False

    def update(self, cursor):
        cx, cy = self._pos_center
        w, h = self._size

        if cx - w//2 < cursor[0] < cx + w//2 and cy - h//2 < cursor[1] < cy + h//2:
            self._pos_center = cursor
            DragRect.IS_MOVING = True
            self._is_moving = True
            DragRect.MOVING_RECT = self
            return True
        else:
            self._is_moving = False
        return False

    def pos_center(self):
        return self._pos_center

    def size(self):
        return self._size


rects = []
for x in range(5):
    rect = DragRect([x * 250+150, 150], [200, 200])
    rects.append(rect)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    lm_list = get_lm_list(hands)

    if lm_list:
        index = lm_list[8]
        middle = lm_list[12]

        if detector.findDistance(index, middle)[0] < 50:
            if DragRect.IS_MOVING and DragRect.MOVING_RECT is not None:
                DragRect.MOVING_RECT.update(index)
            else:
                for rect in rects:
                    moving = rect.update(index)
                    if moving:
                        break
        else:
            DragRect.IS_MOVING = False
            DragRect.MOVING_RECT = None

    # Draw
    newImg = np.zeros_like(img, np.uint8)
    for rect in rects:
        cx, cy = rect.pos_center()
        w, h = rect.size()
        cv2.rectangle(newImg, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), cf.rec_color, cv2.FILLED)
        cvzone.cornerRect(newImg, (cx - w//2, cy - h//2, w, h), 10, colorC=cf.rec_border_color, rt=0)
    out = img.copy()
    alpha = 0.5
    mask = newImg.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, newImg, 1 - alpha, 0)[mask]

    cv2.imshow('Video', out)
    cv2.waitKey(1)

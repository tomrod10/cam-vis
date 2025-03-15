import cv2


class MotionDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            varThreshold=10.5, detectShadows=False
        )

    def alert(self):
        print("Motion detected!")

    def background_subtraction(self, cap, frame):
        fg_mask = self.backSub.apply(frame)
        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(
            frame,
            str(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            (15, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )
        return fg_mask

    def contour_detection(self, frame, fg_mask):
        cont, hier = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ft_cont = self.filter_contour(cont)
        frame_ct = cv2.drawContours(frame, ft_cont, -1, (0, 255, 0), 2)
        return frame_ct

    def filter_contour(self, contours):
        min_contour_area = 200000
        large_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
        ]
        return large_contours

    def threshold(self, fg_mask):
        ret, mask_th = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)
        return (ret, mask_th)

    def erosion_dilation(self, mask_th):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_ed = cv2.morphologyEx(mask_th, cv2.MORPH_OPEN, kernel)
        return mask_ed

    def gaussian_blur(self): ...

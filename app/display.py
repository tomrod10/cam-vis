import cv2
import processing


class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.md = processing.MotionDetector()

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def start(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem receiving frames...")
                break

            fg_mask = self.md.background_subtraction(self.cap, frame)
            _, mask_th = self.md.threshold(fg_mask)
            mask_ed = self.md.erosion_dilation(mask_th)
            frame_ct = self.md.contour_detection(frame, mask_ed)
            cv2.imshow("CT Frame", frame_ct)

            if cv2.waitKey(1) == ord("q"):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

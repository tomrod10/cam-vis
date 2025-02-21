import cv2

class Display:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def start(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem receiving frames...")
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord("q"):
                break
    
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

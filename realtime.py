import cv2
from motion import MotionDetector
from style_transfer import StyleTransfer
from threading import Thread


class RealtimeTransfer():
    def __init__(self, src) -> None:
        self.detector = MotionDetector(default=True, tolerance=5)
        self.cap = cv2.VideoCapture(0)
        W, H = 300, 300
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        cv2.namedWindow('BGR')
        cv2.createTrackbar("Blending", "BGR", 0, 100, self.null)
        cv2.createTrackbar("Detail", "BGR", 0, 100, self.null)
        self.st = StyleTransfer()
        self.st.initialize_reference(src)
        self.grabbed, self.frame = self.cap.read()
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

    @staticmethod
    def null(x):
        pass

    def start(self):
        self.stopped = False
        self.t.start()
    # method passed to thread to read next available frame

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.cap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def transfer(self):
        while True:
            _, image = self.cap.read()
            isMoving = self.detector.detect(image)
            b = cv2.getTrackbarPos('Blending', 'BGR')
            d = cv2.getTrackbarPos('Detail', 'BGR')

            b = b / 100
            d = d / 100

            if isMoving:
                try:
                    self.st.update_face(image, "raw")
                    self.st.warp_img('affine')
                    self.st.process_reference_layers()

                    image = self.st.transform()
                except:
                    print('bla')

            cv2.imshow('BGR', image)
            if cv2.waitKey(1) == ord('q'):
                break
        self.cap.release()


if __name__ == "__main__":
    rt = RealtimeTransfer("poses/makeup_tryon.jpeg")
    rt.transfer()

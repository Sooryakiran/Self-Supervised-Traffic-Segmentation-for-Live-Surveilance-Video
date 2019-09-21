import numpy as np
import cv2
import memory_buffer
import utils

def main():
    """
    The main body of the program

    """
    memory = memory_buffer.Memory(max_size = 4096, image_size = [500, 500, 3], batch_size = 16 )
    motion_estimator = utils.FourrierMotionEstimator()
    video_path = "../Tests/test1.asf"
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened()== False):
        raise ValueError("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame = utils.preprocess(frame)

            """
            Detect motion using FourrierMotionEstimator

            Replace with your own motion estimator.

            """
            motion_estimator.add(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255)
            motion = motion_estimator.get_motion()


            """
            Add the normalized frame and motion detection to Memory.

            """

            memory.add(frame/127.5 - 1, motion/255)
            batchx, batchy = memory.get_batch()

            """
            Use the batch to train the network

            """
            # TODO

            """
            Use the trained network to predict for current frame

            """
            # TODO

            cv2.imshow('Frame', frame)
            cv2.imshow('Motion', motion)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("Video stream ended")
    return 1


if __name__ == "__main__" :
    main()

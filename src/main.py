import numpy as np
import cv2
import memory_buffer
import utils
import neural_network

def main(train = False):
    """
    The main body of the program

    """

    print("Initializing Memory...")
    memory = memory_buffer.Memory(max_size = 256, image_size = [500, 500, 3], batch_size = 2 )
    print("Done\nInitializing Motion Detector...")
    motion_estimator = utils.FourrierMotionEstimator()
    print("Done\nInitializing Neural Network...")
    image_segmenter = neural_network.ImageSegmenter(batch_size = 2)
    image_segmenter.initialize()

    if train == False:
        image_segmenter.load('model/model.ckpt')
    print("Done")
    video_path = "../Tests/test1.asf"
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened()== False):
        raise ValueError("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame = utils.preprocess(frame)
            x = frame
            """
            Detect motion using FourrierMotionEstimator

            Replace with your own motion estimator.

            """
            if train:
                motion_estimator.add(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255)
                motion = motion_estimator.get_motion()
                cv2.imshow('Motion', motion)


                """
                Add the normalized frame and motion detection to Memory.

                """

                memory.add(x/127.5 - 1, motion/255)
                batchx, batchy = memory.get_batch()

                """
                Use the batch to train the network

                """
                if memory.length() > 100:
                    training_loss = image_segmenter.fit(batchx, batchy)
                    image_segmenter.save()
                    print("Current Loss : %f" %training_loss)


            """
            Use the trained network to predict for current frame

            """

            prediction = image_segmenter.predict((x-127.5)/127.5)
            prediction = prediction[0,:,:,0]
            prediction = cv2.resize(prediction, (500,500))
            cv2.imshow('Frame', frame)
            visualise = frame.copy()
            visualise[:,:,0] = prediction*255.0
            # print(visualise)
            cv2.imshow('Learned', visualise)
            cv2.imshow('prediction', prediction)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print('Choose option:\n1. Train\n2. No Train')
                input_choice = int(input())
                if input_choice == 1:
                    train = True
                else:
                    train = False



    print("Video stream ended")
    return 1


if __name__ == "__main__" :
    utils.title("Online Traffic Segmentation")
    print("Choose:\n 1. Run(Default)\n 2. Train from Scratch")
    inp = int(input())

    if inp == 2:
        main(train = True)

    main()

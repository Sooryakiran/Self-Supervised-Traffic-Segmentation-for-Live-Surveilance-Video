import cv2
import numpy as np

class FourrierMotionEstimator:
    def __init__(self, receptive_field = 10):
        self.receptive_field = receptive_field
        self.__images = []

    def add(self, image):
        self.__images.append(image)
        if len(self.__images) > self.receptive_field:
            self.__images.pop(0)
        return 1

    def get_motion(self):
        pixels_in_motion = self.__image_fft(np.asarray(self.__images))[0, :, :]
        ret, thresh = cv2.threshold(pixels_in_motion, 0.8, 255, cv2.THRESH_BINARY_INV)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # thresh = cv2.dilate(thresh, kernel, iterations = 2)
        return thresh


    def __image_fft(self, images):
        """

        3D implimentation of plot_point function

        """
        t = images
        sp = np.fft.fft(np.sin(t), axis = 0)
        freq = np.fft.fftfreq(t.shape[0])
        full = (sp.real**2 + sp.imag**2)**0.5
        arg_null = np.where(freq==0)

        """
        TODO: Calculate probablilty from amplitudes in an interval
        around zero frequency.

        Current Implimentation direclty takes the amplitude corresponding
        to zero frequency. This is effective only for smaller stacks of
        images. For bigger stacks with more images, pixels that are not
        moving lies in an interval around zero.

        """
        p = full[arg_null[0], :, :]
        sum = np.sum(full, axis = 0)
        p = np.nan_to_num(p/sum)
        return p

def preprocess(image, image_size = 500, blur_kernel_size = 7):
    image = cv2.resize(image, (image_size, image_size))
    return cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

def hist_correction(input):
    
    """
    Histogram correction for the input image

    """
    output = np.copy(input)
    for i in range(3):
        output[:, :, i] = cv2.equalizeHist(output[:, :, i])

    return output

if __name__ == "__main__":
    print("This is not an executable file")
    print("Please run main.py")

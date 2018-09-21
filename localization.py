import config
import cv2
import numpy as np
import sys


class FiletypeErrorException(Exception):
    '''An exception raises when the filetype is wrong.'''
    def __init__(self, filename):
        Exception.__init__(self)
        self.filename = filename


def get_led_matrix_location(img, ):
    '''get_led_matrix_location(img, [led_matrix_row, [led_matrix_col]]) -> label,
    location .\n
    @brief Get the location of the led matrix..\n
    . @param img input image..'''

    # check that img exists
    if isinstance(img, type(None)):
        print("[ERROR] type(img) is None")
        exit(1)

    # check that led_matrix_row is not smaller than led_matrix_col
    if led_matrix_row < led_matrix_col:
        print("[ERROR] led_matrix_row is smaller than led_matrix_col")
        exit(1)

    # denoise
    img_blur = cv2.bilateralFilter(img, 5, 150, 150)
    # cv2.imshow('img_blur', img_blur)

    # convert to YUV colorspace
    img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YUV)

    # mask of matrix
    mask_1 = cv2.inRange(img_yuv,  np.array([0, 0, 0]),
                         np.array([80, 255, 255]))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8), iterations=2)
    mask_1 = cv2.dilate(mask_1, np.ones((5, 5), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8), iterations=2)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN,
                              np.ones((9, 9), np.uint8), iterations=2)
    mask_1 = cv2.erode(mask_1, np.ones((3, 3), np.uint8), iterations=3)
    # cv2.imshow('img_mask_1', cv2.bitwise_and(img, img, mask=mask_1))

    # mask of LED
    mask_2 = cv2.inRange(img_yuv,  np.array([190, 0, 0]),
                         np.array([255, 255, 255]))
    mask_2 = cv2.dilate(mask_2, np.ones((3, 3), np.uint8), iterations=3)
    # cv2.imshow('img_mask_2', cv2.bitwise_and(img, img, mask=mask_2))

    # total mask
    mask = cv2.bitwise_or(mask_1, mask_2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
                            iterations=2)
    img_mask = cv2.bitwise_and(img_yuv, img_yuv, mask=mask)
    # cv2.imshow('img_mask', cv2.cvtColor(img_mask, cv2.COLOR_YUV2BGR))

    # find the contours of the img_mask
    gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    binaryIMG = cv2.Canny(blurred, 20, 160)
    (_, cnts, _) = cv2.findContours(binaryIMG.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # print("[INFO] Found {} contours".format(len(cnts)))

    # find the pattern of the led matrix if found contours
    if len(cnts) != 0:
        # return value
        led_matrix_location = []

        # # draw the contours that were founded in blue
        # img_cnts = img.copy()
        # cv2.drawContours(img_cnts, cnts, -1, 255, 3)

        # sort the contours by the area from the biggest to the smallest
        cnts.sort(key=cv2.contourArea, reverse=True)

        for c in cnts:
            # find the led matrix and circle it in green
            if cv2.contourArea(c) < cv2.contourArea(cnts[0]) * 0.6:
                break
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            # cv2.drawContours(img_cnts, [box], 0, (0, 255, 0), 2)

            # find the mass center of the led matrix
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # get the part of the surrounding of led matrix and transform it
            w = np.int0((((box[1] - box[0])**2).sum())**0.5)
            h = np.int0((((box[2] - box[1])**2).sum())**0.5)
            (H, mask) = cv2.findHomography(
                    box.astype('single'),
                    np.array(
                        [[[0., 0.]], [[0., w]], [[h, w]], [[h, 0.]]],
                        dtype=np.single))
            img_led_matrix = cv2.warpPerspective(img_blur, H, (h, w))

            # mask of LED
            mask_led = cv2.inRange(img_led_matrix, np.array([225, 225, 0]),
                                   np.array([255, 255, 255]))

            # find the contours of the mask_led
            (_, cnts_led, _) = cv2.findContours(mask_led.copy(),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

            # find the location of the center of each led \
            # and the radius of each led
            # img_cnts_led = img_led_matrix.copy()
            led_prop = []
            for c in cnts_led:
                (x, y), radius = cv2.minEnclosingCircle(c)
                led_prop.append([int(x), int(y), radius])
                # cv2.circle(img_cnts_led, (int(x), int(y)), 1, (0, 255, 0), 1)
            # cv2.imshow('img_cnts_led', img_cnts_led)
            # print(led_prop)

            # find the uppest, lowest, leftest and rightest point
            extreme_point = []
            led_prop.sort(key=lambda led_prop: led_prop[1]/(led_prop[0] + 0.1))
            extreme_point.append([led_prop[0][0] + led_prop[0][2],
                                  led_prop[0][1] - led_prop[0][2]])
            index = len(led_prop) - 1
            extreme_point.append([led_prop[index][0] - led_prop[index][2],
                                  led_prop[index][1] + led_prop[index][2]])
            led_prop.sort(key=lambda led_prop: (h - led_prop[1]) /
                                               (led_prop[0] + 0.1))
            extreme_point.append([led_prop[0][0] + led_prop[0][2],
                                  led_prop[0][1] + led_prop[0][2]])
            index = len(led_prop) - 1
            extreme_point.append([led_prop[index][0] - led_prop[index][2],
                                  led_prop[index][1] - led_prop[index][2]])
            extreme_point = np.array(extreme_point, dtype=np.single)
            # print(extreme_point)

            # get the part of led matrix and transform it to be square
            (H, mask) = cv2.findHomography(
                            extreme_point,
                            np.array(
                                [[[200., 0.]], [[0., 200]],
                                 [[200, 200]], [[0, 0.]]],
                                dtype=np.single))
            mask = cv2.warpPerspective(mask_led, H, (200, 200))

            # detect the pattern
            pattern = []
            led_matrix_row = 8
            led_matrix_col = 8
            for i in range(0, led_matrix_col):
                tmp = []
                for j in range(0, led_matrix_row):
                    ROI = mask[200//led_matrix_col*i:200//led_matrix_col*(i+1),
                               200//led_matrix_row*j:200//led_matrix_row*(j+1)]
                    # print("({0:3}, {1:3}) - ({2:3}, {3:3}): {4}"
                    #       .format(200//led_matrix_col*i,
                    #               200//led_matrix_row*j,
                    #               200//led_matrix_col*(i+1),
                    #               200//led_matrix_row*(j+1),
                    #               ROI.mean()))
                    if ROI.mean() > 0:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                pattern.append(tmp)
            pattern = np.array(pattern, int)
            # print("[INFO]\nLED Pattern:\n {}".format(pattern))

            # find the most similar pattern in pattern_dict
            diff = 64
            label = None
            for item in config.pattern_dict:
                for i in range(0, 4):
                    tmp = ((np.rot90(pattern, i)-item)**2).sum()
                    if tmp <= diff:
                        diff = tmp
                        label = config.pattern_dict[item]

            led_matrix_location.append((label, (cx, cy)))

        # show the images
        # while True:
        #     cv2.namedWindow(
        #         'img_cnts',
        #         flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO |
        #         cv2.WINDOW_GUI_EXPANDED)
        #     cv2.imshow('img_cnts', img_cnts)
        #     cv2.imshow('img_led_matrix', img_led_matrix)
        #     # press 'q' to quit
        #     if cv2.waitKey(100) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()

        if led_matrix_location != []:
            print("[INFO] Found {} LED Matrix"
                  .format(len(led_matrix_location)))
            return led_matrix_location

    return None


if __name__ == "__main__":

    # get the file path
    if len(sys.argv) == 1:
        filepath = './img/4_1.jpg'
        # print("[ERROR] Please input the path to the file.")
        # exit(1)
    else:
        filepath = sys.argv[1]

    # try to open the file
    try:
        img = cv2.imread(filepath)
        if isinstance(img, type(None)):
            raise FiletypeErrorException(filepath)
    except FiletypeErrorException as ex:
        print("[ERROR] Cannot open '{0}'. "
              "It might not be an image file.".format(ex.filename))
        exit(1)

    led_matrix_location = get_led_matrix_location(img)
    # print(led_matrix_location)

    if not isinstance(led_matrix_location, type(None)):
        for label, location in led_matrix_location:
            print("[INFO] {}: {}".format(label, location))

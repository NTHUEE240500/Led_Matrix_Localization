import cv2
import numpy as np
import localization

cap = cv2.VideoCapture(1)
i = 0
while(True):
    ret, frame = cap.read()
    print(i)
    i+=1
    led_matrix_location = localization.get_led_matrix_location(frame)
    if not isinstance(led_matrix_location, type(None)):
        for label, location in led_matrix_location:
            print("[INFO] {}: {}".format(label, location))

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

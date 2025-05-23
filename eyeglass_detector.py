import dlib
import cv2
import numpy as np  

# =============================================================================
# 1. landmarks conversion function
#    Input : landmarks in dlib format
#    Output: landmarks in numpy format
# =============================================================================
def landmarks_to_np(landmarks, dtype="int"):
    # number of landmark points
    num = landmarks.num_parts

    # initialize the (x, y)-coordinate array
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over all facial landmarks and convert to (x, y) tuples
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y) #[[337 202]
                                                                #[311 206]
                                                                #[247 214]
                                                                #[272 211]
                                                                #[295 265]]

    return coords

# =============================================================================
# 2. Draw regression line & locate pupils
#    Input : image & landmarks in numpy format
#    Output: coordinates of left and right pupils
# =============================================================================
def get_centers(img, landmarks):
    # linear regression on the four eye-corner points
    EYE_RIGHT_OUTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]
    EYE_LEFT_OUTER  = landmarks[2]
    EYE_LEFT_INNER  = landmarks[3]
    

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left  = (EYE_LEFT_OUTER[0]  + EYE_LEFT_INNER[0])  / 2
    x_right = (EYE_RIGHT_OUTER[0] + EYE_RIGHT_INNER[0]) / 2
    LEFT_EYE_CENTER  = np.array([np.int32(x_left),  np.int32(x_left  * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    # draw the regression line and mark pupils
    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    cv2.circle(img, tuple(LEFT_EYE_CENTER),  3, (0, 0, 255), -1)
    cv2.circle(img, tuple(RIGHT_EYE_CENTER), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

# =============================================================================
# 3. Face-alignment function
#    Input : image & left / right pupil coordinates
#    Output: aligned face image
# =============================================================================
def get_aligned_face(img, left, right):
    desired_w, desired_h = 256, 256
    desired_dist = desired_w * 0.5   # target inter-pupil distance

    eyes_center = ((left[0] + right[0]) * 0.5,
                   (left[1] + right[1]) * 0.5)
    dx, dy = right[0] - left[0], right[1] - left[1]
    dist   = np.sqrt(dx * dx + dy * dy)
    scale  = desired_dist / dist
    angle  = np.degrees(np.arctan2(dy, dx))

    # rotation / scaling matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # shift so that eyes land at the centre of the output image
    tX, tY = desired_w * 0.5, desired_h * 0.5
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))
    return aligned_face

# =============================================================================
# 4. Glasses-detection function
#    Input : aligned face image
#    Output: True if wearing glasses, else False
# =============================================================================
def judge_eyeglass(img):
    cv2.imshow('orijinal', img)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    cv2.imshow('gaussian', img)
    # Sobel edge detection (vertical edges)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    cv2.imshow('sobel_y', sobel_y)

    # Otsu threshold to get binary edge map
    _, thresh = cv2.threshold(sobel_y, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # compute ROI positions relative to face size
    d = len(thresh) * 0.5
    x, y  = int(d * 6/7), int(d * 3/4)
    w, h  = int(d * 2/7), int(d * 2/4)

    x2_1, x2_2 = int(d * 1/4), int(d * 5/4)
    w2, y2, h2 = int(d * 1/2), int(d * 8/7), int(d * 1/2)

    roi_1   = thresh[y:y+h, x:x+w]
    roi_2_1 = thresh[y2:y2+h2, x2_1:x2_1+w2]
    roi_2_2 = thresh[y2:y2+h2, x2_2:x2_2+w2]
    roi_2   = np.hstack([roi_2_1, roi_2_2])

    measure_1 = roi_1.sum() / (roi_1.size * 255)
    measure_2 = roi_2.sum() / (roi_2.size * 255)
    measure   = measure_1 * 0.3 + measure_2 * 0.7 #0-1

    cv2.imshow('thresh', thresh)
    cv2.imshow('roi_1', roi_1)
    cv2.imshow('roi_2', roi_2)
    print(measure)

    # threshold  (~0.15)
    wearing_glasses = measure > 0.15
    print(wearing_glasses)
    return wearing_glasses

# =============================================================================
# ************************** main function *********************************
# =============================================================================

predictor_path = "./data/shape_predictor_5_face_landmarks.dat" 
detector  = dlib.get_frontal_face_detector()          # face detector
predictor = dlib.shape_predictor(predictor_path)      # landmark predictor

cap = cv2.VideoCapture(0)   # open webcam

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)   # face detection

    # process every detected face
    for i, rect in enumerate(rects):
        x_face, y_face = rect.left(), rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face

        cv2.rectangle(img, (x_face, y_face),
                      (x_face + w_face, y_face + h_face),
                      (0, 255, 0), 2)
        cv2.putText(img, f"Face #{i+1}",
                    (x_face - 10, y_face - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # landmarks
        landmarks = predictor(gray, rect) 
        landmarks = landmarks_to_np(landmarks) # landmark shape: (5, 2)
        print(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # regression line & pupils
        L_center, R_center = get_centers(img, landmarks)

        # alignment
        aligned = get_aligned_face(gray, L_center, R_center)
        cv2.imshow(f"aligned_face #{i+1}", aligned)

        

        # glasses detection
        if judge_eyeglass(aligned):
            text, color = "With Glasses", (0, 255, 0)
        else:
            text, color = "No Glasses", (0, 0, 255)
        cv2.putText(img, text,
                    (x_face + 100, y_face - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA)

    cv2.imshow("Result", img)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
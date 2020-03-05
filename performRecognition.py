# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
im = cv2.imread("photo_2.jpg")
im1 = im.copy()

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * im.shape[1] )

# Get rectangles contains each contour
rects_old = [cv2.boundingRect(ctr) for ctr in ctrs]

def findLine(rects_sort, line):
    find_line = []
    for i in range(len(rects_old)):
        if (((rects_sort[i][1] <= line[1]) and (rects_sort[i][1] + rects_sort[i][3] >= line[1])) or
                ((rects_sort[i][1] + rects_sort[i][3] >= line[1] + line[3]) and (rects_sort[i][1] <= line[1] + line[3])) or
                (rects_sort[i][1] > line[1]) and (rects_sort[i][1] < line[1] + line[3]) or
                (rects_sort[i][1] + rects_sort[i][3] > line[1]) and (rects_sort[i][1] < line[1] + line[3])):
            find_line.append(rects_old[i])
            # cv2.rectangle(im, (rects_sort[i][0], rects_sort[i][1]),
            #               (rects_sort[i][0] + rects_sort[i][2], rects_sort[i][1] + rects_sort[i][3]), (0, 255, 0), 3)

    for i in range(len(find_line)):
        for j in range(i):
            if find_line[i][0] < find_line[j][0]:
                rectsold = find_line[i]
                find_line[i] = find_line[j]
                find_line[j] = rectsold

    arr_y = [y[1] + y[3] for y in find_line]

    return find_line, max(arr_y), len(find_line)

def sortRects(rect_sort):
    rects_sorted_count = 0
    rects_sorted = []
    endline_y = 0

    while(rects_sorted_count < len(rects_old)):
        minn = rects_old[0]
        for i in range(len(rects_old) - 1):
            if (rects_old[i + 1][1] < rects_old[i][1]) and endline_y < rects_old[i+1][1]:
                minn = rects_old[i + 1]
        # cv2.rectangle(im, (minn[0], minn[1]), (minn[0] + minn[2], minn[1] + minn[3]), (255, 255, 255), 3)

        line, endline_y, len_line = findLine(rects_old, minn)
        rects_sorted_count += len_line
        rects_sorted += line
        # print(line, endline_y)
    return rects_sorted
rects_sorted = sortRects(rects_old)

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
a = 0
for rect in rects_sorted:
    a = a+1
    # Draw the rectangles
    cv2.putText(im1, str(a), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # Calculate the HOG features
    # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    # cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    # print(nbr[0])

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.imshow("im1", im1)
cv2.waitKey()


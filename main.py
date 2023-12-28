import cv2 as cv

background = cv.imread("./files/background.png")
background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
background = cv.GaussianBlur(background, (21,21),0)

cap = cv.VideoCapture("./files/test.avi")

while(cap.isOpened()):
    ret, frame = cap.read()
    cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)

    cv.resizeWindow("Resized_Window", 600, 420)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    gray = cv.GaussianBlur(gray, (21,21), 0)
    diff = cv.absdiff(background,gray)
    thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations = 2)

    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour)<20000:
            continue
        (x,y,w,h) = cv.boundingRect(contour)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv.imshow("Resized_Window",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

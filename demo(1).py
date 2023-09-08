import numpy as np
import cv2
import time
import pandas as pd 


FONT = cv2.FONT_HERSHEY_SIMPLEX

# Display speed , if 1 means no speed displayed 
THRES = 0

# POINTS to watch 
WATCH_POINTS = [(0, 2)]
WATCH_FLOW  = [[]]


RATIO  = 1.414 / 2 * 2.45 * 30 / 4 / 100 # change the ration based on camera intrics parameters and extrics parameters

RESIZE_RATIO= 1


def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    
    # add water speed
    v = np.sqrt(fx ** 2 + fy ** 2)
    v_max = v.max()
    for i in range(x.shape[0]):
        if v[i] / v_max > THRES:
            cv2.putText(img_bgr, str(round(v[i] * RATIO)), (int(x[i] + 0.5), int(y[i])), FONT, 
                        0.2, (255, 0, 0), 1, cv2.LINE_AA)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

# def watch_points(points, flow):
#     flow_y = flow[:, :, 1]
#     for point in points:


if __name__ == "__main__":


    cap = cv2.VideoCapture("./videos/Videos1.avi")

    suc, prev = cap.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    h, w = list(prevgray.shape[:2])[::-1]
    prevgray = cv2.resize(prevgray,( h // RESIZE_RATIO, w // RESIZE_RATIO))

    # print(prevgray.shape)
    count = 0
    # result = cv2.VideoWriter('filename.mp4', 
    #                          cv2.VideoWriter_fourcc(*'MP4V'),
    #                          20.0, ( h // 4, w // 4))
    while True:
        count += 1
        suc, img = cap.read()
        if suc:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (h // RESIZE_RATIO, w // RESIZE_RATIO))
            # start time to calculate FPS
            start = time.time()


            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            for (i, point) in enumerate(WATCH_POINTS):
                flow_point = flow[point[0], point[1]]
                WATCH_FLOW[i].append(np.sqrt(flow_point[0] ** 2 + flow_point[1] ** 2) * RATIO)
            prevgray = gray


            # End time
            end = time.time()
            # calculate the FPS for current frame detection
            fps = 1 / (end-start)

            print(f"{fps:.2f} FPS")

            flow_img = draw_flow(gray, flow)
            hsv_img = draw_hsv(flow)

            # result.write(np.concatenate((flow_img, hsv_img), axis=1) )

            cv2.imshow('flow', flow_img)
            cv2.imshow('flow HSV', hsv_img)
            if count == 1:
                cv2.imwrite('./out/image/flow.jpg', flow_img)
                cv2.imwrite('./out/image/hsv.jpg', hsv_img)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
    print(WATCH_FLOW)

    # result.release()
    cap.release()
    cv2.destroyAllWindows()
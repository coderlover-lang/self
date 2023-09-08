import numpy as np 
import cv2 

def load_frames(source):
    arr = None
    with open(source, 'rb') as f:
        arr = np.load(f) 
    return arr   


def frames_to_bgrs(arr):
    total_observe_line = arr.shape[1]
    total_flow_direction_line = arr.shape[2]
    fx, fy = arr[:, :,:,0], arr[:, :,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    
    # create hsv figure for flow vecoity
    
    hsv = np.zeros((ang.shape[0], total_observe_line, total_flow_direction_line, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    
    # hsv to bgr
    bgr = np.array([cv2.cvtColor(hsv[i], cv2.COLOR_HSV2RGB).tolist() for i in range(arr.shape[0])])
    return bgr

# input 
#   frame np.ndarray (bgr figure)
#   y_ind number
# return
#   (start, end)
def find_flow_observe_line_sd(frame, y_ind):
    different_map = np.where(frame.sum(axis = 2) == 0, 0, 255 )
    start_index = different_map[y_ind, :].tolist().index(255)
    end_index = frame.shape[1] - different_map[y_ind, :].tolist()[::-1].index(255) - 1
    return start_index, end_index

# input:
#   bgr shape [frame, w, h, 3] 
#   frame_index : number
#   y_ind : number 
#   step : number (measure intetrval)
#   uninterval_index : None | iterable (make efferts only when step is not a const)
# return:
#   

def calc_frame_flow_index(frame, y_ind = 120, step = 60, uninterval_indicies = None, surface_width = 420):
    start_index, end_index = find_flow_observe_line_sd(frame, y_ind)
    if not uninterval_indicies:
        indices = np.array([*range(0, surface_width + 1, step)])
    else:
        indices = np.array(uninterval_indicies)
    map_indices = (indices * (end_index - start_index) / surface_width).astype("int")
    return map_indices


def calc_flow_index(bgr, frame_index = 0, y_ind = 120, step = 60, uninterval_indicies = None, surface_width = 420):
    map_indices = calc_frame_flow_index(bgr[frame_index], y_ind, step, uninterval_indicies, surface_width)
    return map_indices





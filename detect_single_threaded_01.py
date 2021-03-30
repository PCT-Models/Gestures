from utils import detector_utils as detector_utils
import cv2
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime
import argparse


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture('participant14_edit.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    
    im_width =  600
    im_height = 600
    # max number of hands we want to detect/track
    num_hands_detect = 2

#    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Single-Threaded Detection', 600,600)
#    size = (im_width,im_height)
#    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    f = open("pat_33_001.txt","a+")
    k = 0
# pat_01 5500-6500, pat_02 6500-7500
    
#    while k < frame_count-1:

    for k in range(9300,10500):
#        k = k+1
#        print(k)
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        image_np = image_np[1:600,201:800]
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        except:
            print("Error converting to RGB")

        boxes, scores = detector_utils.detect_objects(image_np,detection_graph, sess)
        if len(boxes) > 1: 
            (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                              boxes[0][0] * im_height, boxes[0][2] * im_height)
        else:
             (left, right, top, bottom) = (0, 0, 0, 0)
        
        
        f.write(str(left) + '\t' + str(right) + '\t' + str(top) + '\t' + str(bottom) + '\t' + '\n')    
        # draw bounding boxes on frame
        cv2.circle(image_np,( int((int(left)+int(right))/2), int((int(top)+int(bottom))/2) ),10,(0,255,0))    
        
        detector_utils.draw_box_1_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)
        detector_utils.draw_box_2_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = k

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
#            out.write(image_np)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
#                out.release()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
                    
             
            out.release()
    f.close()
    cv2.destroyAllWindows()
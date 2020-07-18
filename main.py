"""People Counter."""
"""
person-detection-retail-0013
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.42,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    last_count = 0
    total_count = 0
    start_time = 0
    current_request_id =0
    video_frames = 0
    bbx_frames = 0
    dropped_frames =0

    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,current_request_id, args.cpu_extension)[1]
    
    # A flag for single image
    image_flag = False
    # Check if the input is a webcam
    if args.input =='CAM':
        args.i =0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
    else:
        assert os.path.isfile(args.input), "File not found!"

    #Read from the video capture
    cap = cv2.VideoCapture(args.input)
    
    if args.input:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("ERROR with video source!")

    # Grab the shape of the input 
    net_input_shape = infer_network.get_input_shape()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #Loop until stream is over
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        video_frames = video_frames +1
            
        # Pre-process the frame
        p_frame = cv2.resize(frame, (w,h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape((n, c, h, w))

        #Start asynchronous inference for specified request
        inf_start = time.time()
        infer_network.exec_net(current_request_id, p_frame)

        #Wait for the result 
        if infer_network.wait(current_request_id) == 0:
            det_time = time.time() - inf_start
            
            #Get the results of the inference request
            result = infer_network.get_output(current_request_id)
            
            #Update the frame and count to include detected bounding boxes
            frame, current_count = draw_boxes(frame, args, result, width, height)
            
            #Message for dropped frames
            dropped_frames_message = "Dropped frames : {}"\
                               .format(dropped_frames)
            cv2.putText(frame, dropped_frames_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            

            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            total_video_frames_message = "Total video frames: {}"\
                               .format(video_frames)
            cv2.putText(frame, total_video_frames_message, (15, 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            bbx_frames_message = "Total video bbx frames:{}"\
                               .format(bbx_frames)
            cv2.putText(frame, bbx_frames_message, (15, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            
            # Frame Counting for accuracy estimation
            if current_count !=0:
                bbx_frames = bbx_frames +1
            else :
                dropped_frames = dropped_frames + 1

            # Check if a person enters in the frame
            if current_count > last_count:
                start_time = time.time()
                bbx_frames = bbx_frames +1
                total_count = total_count + current_count - last_count
                # Send total count to server
                client.publish("person", json.dumps({"total": total_count}))
            
            # Check if a person exits based on time thresholding and calculate the duration
            if current_count < last_count and int(time.time() - start_time) > 2  : 
                duration = int(time.time() - start_time)
                # Publish the duration to the server
                client.publish("person/duration", json.dumps({"duration": duration}))
                
            # Send the current count to the server
            client.publish("person", json.dumps({"count": current_count}))
            
            last_count = current_count
                    
            # Break if k key pressed
            if cv2.waitKey(1) & 0xFF == ord("k"):
                break

        #Send the frame to the FFMPEG server 
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        #Write an output image if `single_image_mode`
        if image_flag:
            cv2.imwrite('images/output_image.jpg', frame)    
        
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    #Disconnect from MQTT
    client.disconnect()

def draw_boxes(frame, args, result, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    
    for box in result[0][0]: # Output shape is 1x1xNx7
        conf = box[2]
        if float(conf) > float(args.prob_threshold) :
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count +1
    
    return frame, current_count



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()


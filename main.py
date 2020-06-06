"""People Counter."""
"""
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

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 120


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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")
    return parser


def connect_mqtt():
    # TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT)
    return client

def analyze_frame(obj_det_out,video_stats,decision_param):
    current_max_prob_threeshold = decision_param['THRESHOLD_PROB']
    person_detected = False
    for i in range(obj_det_out.shape[2]):#Read confidence  and label iteratively for each detected objects
            confidence =  obj_det_out[0,0,i,2]  
            label=obj_det_out[0,0,i,1]
            #print(obj_det_out[0,0,i,:])
            
            if ((label==1) and (confidence> current_max_prob_threeshold)): #Check if person(label=1) is detected and and confidence is greater than prob. threshold
                current_max_prob_threeshold =confidence
                person_detected = True
                #print("label & Confidence %d ,%d",(label,confidence))
                video_stats['Box_coordinate'][0]=obj_det_out[0,0,i,3]
                video_stats['Box_coordinate'][1]=obj_det_out[0,0,i,4]
                video_stats['Box_coordinate'][2]=obj_det_out[0,0,i,5]
                video_stats['Box_coordinate'][3]=obj_det_out[0,0,i,6]
    #Below state machine update video state based on last video state and condition whether person detected in current frame or not.
    
    # Different Video states are : First frame  , No person in Frame , Person in frame , Missing Person in Frame
    
    # Also it update stats for current frame and video i.e. person in frame ,total count, video_state['person_time_spent_in_frame']
    
    #Copy video_stats dictionary to local variables for better code readablity 
    
    video_state =video_stats['video_state']
    person_time_spent_in_frame=video_stats['person_time_spent_in_frame']
    no_person_in_consecutive_frames=video_stats['no_person_in_consecutive_frames']
    total_count =video_stats['total_count']
    THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE=decision_param['THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE']
    person_exited_frame = False 
    
    
    #State Machine starts here, see write-up for state machine diagram.
    if(video_state=='first_frame'):  # first frame
            
            if(person_detected):  # if person detected then set video state as 'person_in_frame' and update person_in_frame flag , person_time_spent_in_frame and total_count
                video_state='person_in_frame'
                person_in_frame =1
                person_time_spent_in_frame=1
                total_count =1
                
            else:# if no person detected then set video state as 'no_person_in_frame' and update person_in_frame flag 
                video_state='no_person_in_frame'
                person_in_frame =0
                
    elif(video_state=='no_person_in_frame'): # last frame state is  'no_person_in_frame'
    
        if(person_detected):# if person detected then set video state as 'person_in_frame' and update person_in_frame flag , increment person_time_spent_in_frame and total_count(new person has come)
            video_state='person_in_frame'
            person_in_frame =1
            person_time_spent_in_frame+=1
            total_count +=1
            
        else: #if no person detected then keep video state as 'no_person_in_frame' and update person_in_frame flag
            video_state='no_person_in_frame'
            person_in_frame =0
            person_time_spent_in_frame=0
            
    elif(video_state=='person_in_frame'):# last frame state is  'person_in_frame'
    
        if(person_detected): # if person detected then keep video state as 'person_in_frame' and update person_in_frame flag , increment person_time_spent_in_frame 
            video_state='person_in_frame'
            person_in_frame =1
            person_time_spent_in_frame+=1
           
                     
        else:  #if no person detected then update video state as 'missing_person_in_frame' and update person_in_frame flag , set no_person_in_consecutive_frames=1
            video_state='missing_person_in_frame' 
            person_in_frame =1
            no_person_in_consecutive_frames=1
            
    elif(video_state=='missing_person_in_frame'): # last frame state is  'missing_person_in_frame'
    
        if(person_detected):# if person detected before no_person_in_consecutive_frames exceed THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE it means model as failed to detect person for last few frames hence update video state as 'person_in_frame'  and treat him/her as last person. Update person_in_frame flag. Update person_time_spent_in_frame by taking account that model has missed few frames to detect(no_person_in_consecutive_frames)   
            video_state='person_in_frame'
            person_in_frame =1
            person_time_spent_in_frame=person_time_spent_in_frame + no_person_in_consecutive_frames + 1
            
        elif(no_person_in_consecutive_frames<THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE): # if person is not detected and no_person_in_consecutive_frames not exceed THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE then keep video_state as 'missing_person_in_frame'  and increment no_person_in_consecutive_frames. Update person_in_frame flag. 
            video_state='missing_person_in_frame' 
            no_person_in_consecutive_frames+=1
            person_in_frame =0
        else:# if person is not detected and no_person_in_consecutive_frames  exceed THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE then person has exited scene. Update video_state as 'no_person_in_frame'.  Update person_in_frame and person_exited_frame flag. 
            video_state='no_person_in_frame'
            person_exited_frame = True
            person_in_frame =0
            
    else:  #this condition should never come
        print("Invalid State")
        exit(0)
        
    #Update local variables back to video_stats dictionary 
    video_stats['video_state'] = video_state
    video_stats['person_time_spent_in_frame']=person_time_spent_in_frame
    video_stats['no_person_in_consecutive_frames']=no_person_in_consecutive_frames
    video_stats['total_count'] = total_count
    video_stats['person_exited_frame'] = person_exited_frame
    video_stats['person_in_frame'] = person_in_frame
    return person_detected
    

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

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model,args.cpu_extension,args.device)
    input_shape = infer_network.get_input_shape()
    input_image_width = input_shape[3]
    input_image_height=input_shape[2]
    
    ### TODO: Handle the input stream ###
    try: # try opening input file as image if file is not image, if it throw exception then try opening as video.   
        frame=cv2.imread(args.input)
        IS_IMAGE = True
        hasFrame =True
        out_image_file = os.path.splitext(args.input)[0] + "_inferred" + ".jpg"
        #print("Successfully Opened Image")
        fps=0
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
    except :
        try: # Trying opening as video if it throw exception it means input is neither valid  video nor image file.
            if(args.input =='0'): # check if input is webcam
               #print('input is webcam')
               args.input =int(args.input) 
            video=cv2.VideoCapture(args.input) #Open video stream
            if (video.isOpened()): # check video stream is successfully opened
                hasFrame,frame=video.read()
                IS_IMAGE = False
                fps=int(video.get(cv2.CAP_PROP_FPS))
               
                #print ("FPS is {}".format(fps))
                frame_height = frame.shape[0]
                frame_width = frame.shape[1]
                
                if(args.input):  
                    out_video_file = os.path.splitext(args.input)[0] + "_inferred" + ".avi"
                else: # if webcam input fixed output filename
                    out_video_file = 'webcam_inferred.avi'
                          
                out_video=cv2.VideoWriter(out_video_file,cv2.CAP_OPENCV_MJPEG,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
                
            else: # Video stream is failed to open
                print('Video capture is not opened properly, Exiting')
                video.release()
                exit()
        except: # Both try to open input as video or image failed , exiting  
            
             print(" Error Opening input!!! ,Input is neither valid image nor video file, please provide right input. Exiting !!!")
             exit()
   
    # initialize video stats variables    
    last_stat_person_in_frame =-1
    last_stat_total_count =-1
    THREESHOLD_NO_OF_SECONDS_FOR_PERSON_LEFT_SCENE = 1.5
    THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE = int(THREESHOLD_NO_OF_SECONDS_FOR_PERSON_LEFT_SCENE*fps)
    frame_no =1
    video_stats ={'video_state' : 'first_frame' , 'person_in_frame' : 0, 'person_time_spent_in_frame' :0 ,'no_person_in_consecutive_frames' :0 ,'total_count':0, 'person_exited_frame' : False,'Box_coordinate' :[None,None,None,None]} # Video statistics  dictionary which will be updated as frames by get processed by analyze_frame() function
    
    decision_param = {'THRESHOLD_PROB' : prob_threshold , 'THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE' :THREESHOLD_NO_OF_FRAMES_FOR_PERSON_LEFT_SCENE} # Decision threshold parameters
    
     ### TODO: Read from the video capture ###
    while(hasFrame and cv2.waitKey(1)<0): #Read video frame by frame
       
        ### TODO: Pre-process the image as needed ###
        input_image = cv2.resize(frame,(input_image_width, input_image_height))
        input_image = input_image.transpose((2,0,1))
        input_image = input_image.reshape(1, 3, input_image_height, input_image_width)
        
        ### TODO: Start asynchronous inference for specified request ###
        t0=time.time()
        async_infer_req_handle=infer_network.exec_net(input_image,0)
        
        ### TODO: Wait for the result ###
        infer_network.wait(async_infer_req_handle)
        t1=time.time()
        infer_time =round((t1-t0)*1000)
        #print("For frame no. {} , infer taken {} miliseconds".format(frame_no, infer_time)) 
        
        ### TODO: Get the results of the inference request ###
        obj_det_out=infer_network.get_output(async_infer_req_handle)['DetectionOutput']
        
        ### TODO: Extract any desired stats from the results ###
        #Function to analyze frame and update video statistics  
        person_detected = analyze_frame(obj_det_out,video_stats,decision_param)
               
   
        
       # if person detected draw box on image frame
        if(person_detected):
            x1 =int(video_stats['Box_coordinate'][0] *frame_width)
            y1 = int(video_stats['Box_coordinate'][1]*frame_height)
            x2 =int(video_stats['Box_coordinate'][2]*frame_width)
            y2 = int(video_stats['Box_coordinate'][3]*frame_height)
            frame=cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), int(round(frame_height/150)), 8)
            cv2.putText(frame,'Person :' + str(video_stats['total_count']),(x2,y2+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
          
        # put frame_no , frame inference time, person in frame and total person stats in frame   
        cv2.putText(frame,'Frame No. ' + str(frame_no) +' Infer Time in ms: ' +str(infer_time),(10,20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 1, cv2.LINE_AA) 
        cv2.putText(frame,'Current Count:' + str(video_stats['person_in_frame']),(10,40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 1, cv2.LINE_AA) 
        cv2.putText(frame,'Total No. of Person:' + str(video_stats['total_count']),(10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 1, cv2.LINE_AA) 
        
        if(not IS_IMAGE): # if input is video put current person duration stat in frame
            cv2.putText(frame,'Current person duration' + str(video_stats['person_time_spent_in_frame']/fps),(10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
        
       ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
       #Here statistics send over MQTT    
       
        #sending personduration to MQTT server whenever person exit frame
        if(video_stats['person_exited_frame'] and (not IS_IMAGE)): # if person exited frame and input is video then send  last exited person duration to MQTT server.
            json_last_person_time_spent =json.dumps({'duration': video_stats['person_time_spent_in_frame']/fps})
            client.publish('person/duration',json_last_person_time_spent)
            video_stats['person_exited_frame'] =False
            #print('Person duration :{}'.format(json_last_person_time_spent))
            
        #sending current count to MQTT server
        if((last_stat_person_in_frame !=video_stats['person_in_frame']) or (last_stat_counter >9)): # Instead of sending current count every frame , send current count when it is updated or after every 10 frames.  Network data saving!!! 
            count_data = {'count' :video_stats['person_in_frame']}
            json_count_data = json.dumps(count_data)
            client.publish('person',json_count_data)
            last_stat_person_in_frame = video_stats['person_in_frame']
            #print('Current Count {}'.format(json_count_data))
            last_stat_counter = -1
        last_stat_counter+=1 
        
        #sending total count to MQTT server
        if(last_stat_total_count !=video_stats['total_count']): # Instead of sending total count every frame , send total count when it is updated. Network data saving!!!  
            total_count_data = {'total':video_stats['total_count']}
            json_total_count_data = json.dumps(total_count_data)
            client.publish('person',json_total_count_data)
            last_stat_total_count =video_stats['total_count']
           # print('Total Count {}'.format(json_total_count_data))
         
        
        ### TODO: Send the frame to the FFMPEG server ###
        if ( not IS_IMAGE):
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
            
        
        #show frame (only for local pc)    
        #frame1 = cv2.resize(frame,(frame_width,frame_height))
        #cv2.imshow('Inferred Image' ,frame1)
        
         ### TODO: Write an output image if `single_image_mode` ###
        if (IS_IMAGE):
            cv2.imwrite(out_image_file,frame)
            cv2.waitKey(0)
            break
        else:
            out_video.write(frame)
            hasFrame,frame=video.read()
            frame_no+=1
    
    # Sending person duration if last frame ended in 'missing_person_in_frame' or 'person_in_frame' state
    if((video_stats['video_state']=='missing_person_in_frame' or video_stats['video_state']=='person_in_frame' )and (not IS_IMAGE)):
            json_person_time_spent =json.dumps({'duration': video_stats['person_time_spent_in_frame']/fps})
            client.publish('person/duration',json_person_time_spent)
    client.disconnect()
    if (not IS_IMAGE):    
        video.release()
        out_video.release()
    cv2.destroyAllWindows() 

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    #print(args)
    #Connect to the MQTT server
    client = connect_mqtt()
    #Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

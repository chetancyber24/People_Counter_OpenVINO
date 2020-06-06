**Project Write-up**

**1.  Explaining Custom Layers in OpenVINO™**
	
Below are steps involved to convert custom layers

 - Use model extension generator script(extgen.py) to generates
   templates for custom layer. Extensions need to be generated for model
   optimizer(frame work and Ops) and inference engine (device specific
   i.e. CPU , GPU etc).
   
 - Edit model optimizer extractor extension template file for layer
   specific parameters and operation extension template file for output
   shape.
      
 - Generate IR files with above extension templates (--extensions   
   option)
 - Edit hardware specific template file (C++ file in case of CPU) to   
   implement custom layer operation.
   
 - Compile above C++ file to generate extension library file(.dll for   
   Windows & .so for Linux).
    
 - Use above library extension file with inference engine plugin during 
   inference.



Reasons for handling custom layers. Model optimizer have list of supported layers for each ML framework(i.e. TF, Caffe, Mxnet etc). Layers which are not in supported layers need to be implemented as custom layers. As these ML frameworks are rapidly involving there is good chance newly added layers in latest ML framework release will not be supported by Model optimizer for some time( it may be supported in future release of OpenVINO). So custom layers give user option to implement non-supported layers. Also, there is possibility that user can have particular operation (for  e.g. new non-linear function) in layer which is not standard operation supported by Model optimizer then it need to be implemented as custom layers.

**2. OpenVINO Model performance**
  To compare the model performance with & without OpenVINO , test video(provided by Udacity) is inferred in Tensorflow environment using frozen(.pb) SSD_Mobilenet_v2_COCO model vs IR model in OpenVINO environment. Below are observations on different parameters.
  
 - **Speed** - To calculate average inference time for one frame, all frames (of video) inference time is averaged. In OpenVINO environment average inference time is ~24.5 ms while in Tensorflow environment is ~92ms. So OpenVINO model is ~3.8X time faster than native Tensor flow model. Local machine configuration is CPU: i5-8350 & RAM: 16 GB
 
- **Model Size**-  SSD_Mobilenet_v2_COCO model size for both OpenVINO IR format and Tensor flow pb format is  almost same(around 64MB) .

- **Accuracy** – Between OpenVINO &  Tensor flow inferred test video frames, there is mismatch for person detection in some frames . Mismatch is less than 10% for total test video frames (1394 frames).

- **Network Cost** – If inference is deployed in cloud below, will be approx. data need to be sent for test video.  
      Data size = Input Image resolution * No of Channels *8(8 bit color) *fps *duration of video
              =300*300*3*8*10*139 bits
	           = 3002400000 bits
               = ~ 357MB
For above calculation, assumption is raw video is sent without any video compression. Assuming edge device don’t have enough compute to do video compression. As above calculation show, for small 139 sec video 357 MB video data need to be sent which is very huge. For 1 hour video it will be 9.2GB
If inference is done on edge then only MQTT data need to be sent which has negligible size compared to video frames in previous cloud case. Also latency will be better for inference on edge compare to cloud case. In summary there can be huge advantage for network cost in case of inference on edge.

**3. Assess Model Use Cases**
This model can be used in retail space (like showroom) to analysis how much particular product showcase(or demo) is engaging to customer by counting no of people stop by on particular product demo booth and how much time they are spending.  For example in big electronics appliance and gadget store, it can give stats, which particular gadget showcase counter attract lot of peoples and deploy more sales person to help customers in particular gadget section. It also help  to identify which product(gadget) showcase is not getting highlighted(low footfall) and need optimization for product placement in retail space.

**4. Assess Effects on End User Needs**
- Low  lighting for scene can affects inference model accuracy and it may fail to detect person. Possible fix can be use model which are trained with low lighting images (or re train exiting model with low light images). Use pre-processing of image frame to improve lighting before feeding it to model. Use camera hardware which performs better in low light.

-	Model accuracy depends on end user requirement. Higher accuracy needs higher compute power which may not be available on edge device. Generally, on edge devices inference is done on lower precision (FP16, INT8). Instead of training model in high precision and converting it to low precision for inference which may cause loss of accuracy compared to training phase.  It is advisable to train model on low precision in high end machine so that there is no loss of accuracy when it is ported to edge device for inference in low precision. Further model can be tailored to end user requirement. For e.g. in people counter application model need to just identify whether person is in seen or not. But publicly available objection detection model identifies 100 classes of which increase model complexity (more parameters require more compute and memory). If model can be retrained to just identify one class (person) in scene, it can be greatly simplified and serve the purpose for this application (people counter).

- Camera focal length and Image size- If subject(here person) is not in camera focus, image will be blurred and model may miss to detect person in frame. Also if image resolution is lower than image resolution which is used for training then model may miss to detect person in frame.

**5. Model Research**

I explored below models in Tensorflow Object Detection Model Zoo([https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)).

- **ssd_mobilenet_v2_coco(IR format) –** Inference time is good , accuracy is ok. But it missed lot of consecutive frames for ~4sec when 2nd person(in test video) was in standstill position with probability threshold of 0.5. To solve that issue, I put logic(state machine , see below state diagram) in program , once person detected in frame and missed in subsequent frame ,it starts counter and wait for threshold  no of  consecutive frames where person is not detected. If person is detected again before counter value crosses threshold value, it decides person detected was same person and model missed to detect person in few frames. If counter value cross threshold before detecting person, it decides person has left scene. Even though with this logic when threshold time was set to 4 sec, it created another issue , time gap between 1st person exit and 2nd person entering video is less than 4 sec which causes  2nd person to detect as 1st person. To solve this issue I needed to improve  person detection of model so I reduce probability threshold to 0.3 and then detection was improved and don’t have issue.

![State Machine Diagram](https://github.com/chetancyber24/People_Counter_OpenVINO/blob/master/state_machine_dig.png)

Other than playing with probability threshold, I tried other object detection model(discussed below).

I explored **ssd_resnet_50_fpn_coco** which has better mAP score than Mobilenet_V2 . But resnet model also failed to detect  2nd person in standstill position for ~4sec consecutive frames with probability threshold of 0.5. Also its inference time (1sec /frame) is very high compare to Mobilenet.  I also tried **ssd_inception_v2_coco** but that also has same issue.

In summary I stick to  **ssd_mobilenet_v2_coco(IR format)** because it was meeting requirement after reducing probability threshold to 0.3 and logic to handle few frames where person is not detected. Also it is considerably faster than other models, its inference time is ~24ms per frame. Mobilenet V2  model size is also small ~64MB. Small model size and low compute resource make it suitable for deployment on edge devices.

**Reference:**

- Intel OpenVINO Documentation

- Udacity Intel® Edge AI for IoT Developers Nanodegree Program Course Material.

- TensorFlow Objection Detection Model Zoo





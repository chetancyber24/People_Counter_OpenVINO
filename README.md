**Command to run application on Uadcity workspace(Linux Environment)**

For better detection and accuracy use probability threshold of 0.3
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ./frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```
**Command to convert Tensorflow model to IR format**

 Below is command(inside model directory) to convert ssd mobilenet v2 objection detection model to IR format. Link for original model : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
 
 ```
 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ./frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ./pipeline.config  --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  --reverse_input_channel 
 ```
 
 **Project write up is** [here](https://github.com/chetancyber24/People_Counter_OpenVINO/blob/master/WRITEUP.md)

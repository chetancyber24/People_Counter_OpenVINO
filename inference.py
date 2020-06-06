#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        model_net = None
        infer_engine_plugin = None
        input_blob = None
        ### TODO: Initialize any class variables desired ###

    def load_model(self,model_xml_file,cpu_ext,device):
        ### TODO: Load the model ###
        model_bin_file = os.path.splitext(model_xml_file)[0] + ".bin"
        net= IENetwork(model=model_xml_file, weights=model_bin_file)
        self.model_net = net
        
        plugin =IECore()
               
        ### TODO: Add any necessary extensions ###
        if (cpu_ext is not None): # load cpu extension only if unsupported layers are present
            plugin.add_extension(cpu_ext,"CPU")
            
            ### TODO: Check for supported layers ###
            # check unsupported layers after CPU plugin loaded
        layers_supported = plugin.query_network(network=net, device_name=device)
        layers_unsupported =[] 
        for layer in net.layers.keys():
            if(layer not in layers_supported):
               layers_unsupported.append(layer)
        if(len(layers_unsupported) !=0):
            #print(" Unsupported layers present even after CPU extension loaded. Will Exit Now.")
            exit(2)
                    
                    
        ### TODO: Return the loaded inference plugin ###
        self.infer_engine_plugin = plugin.load_network(net, device)
        ### Note: You may need to update the function parameters. ###
        return  

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        self.input_blob= next(iter(self.model_net.inputs))
        # print(self.model_net.inputs[self.input_blob].shape)
        # print(self.model_net.outputs[next(iter(self.model_net.outputs))].shape)
        # exit(0)
        return self.model_net.inputs[self.input_blob].shape
        
        

    def exec_net(self,image,request_id=0):
        ### TODO: Start an asynchronous request ###
        async_infer_request= self.infer_engine_plugin.start_async(request_id,{self.input_blob :image})
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return async_infer_request

    def wait(self,async_infer_req):
        ### TODO: Wait for the request to be complete. ###
        async_infer_req.wait()
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_output(self,async_infer_req):
        ### TODO: Extract and return the output results
        return async_infer_req.outputs
        ### Note: You may need to update the function parameters. ###
        

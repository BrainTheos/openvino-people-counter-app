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
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.net_plugin= None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request_handle = None

    def load_model(self, model, device, input_size, output_size, num_requests,cpu_extension=None, plugin=None):
        """Load the model
        Check for supported layers
        Add any necessary extensions
        Return the loaded inference plugin  
        """
        
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device=device)
        else:
            self.plugin = plugin
        
        
        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Get the supported layers of the network
        supported_layers = self.plugin.get_supported_layers(self.network)
        
        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            #exit(1)
        
        # Add cpu extensions to the plugin 
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)
            
        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.net_plugin = self.plugin.load(network=self.network)
        else:
            self.net_plugin = self.plugin.load(network=self.network, num_requests=num_requests)
        
        # Set the network inputs and outputs
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        
        return self.network.inputs[self.input_blob].shape
    
    def performance_counter(self, request_id):
        """
            Returns perfomance measure per layer
        """
        perf_count = self.net_plugin.requests[request_id].get_perf_counts()
        return perf_count

    def exec_net(self, request_id,image):
        ### Start an asynchronous request based on the request id of a frame ###
        ### Return net_plugin ###
        
        self.infer_request_handle =self.net_plugin.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return self.net_plugin

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return the status of a request ###
        status = self.net_plugin.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id, output=None):
        ### TODO: Extract and return the output results
        ### based on request_id and output params###
        if output:
            result = self.infer_request_handle.outputs[output]
        else:
            result = self.net_plugin.requests[request_id].outputs[self.output_blob]
        
        return result

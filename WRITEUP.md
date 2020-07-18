# Project Write-Up

### Suitable prob. threshold for the model used : 0.43

## Explaining Custom Layers

The process behind converting custom layers involves adding layers that are not among the built in supported layers depending on the supported frameworks. In case of Tensorflow and Caffe, some options are :
-Register the custom layers as extensions to the Model Optimizer
-register the layers as custom, then use Caffe to calculate the output shape of the layer
-Replace the unsupported subgraph with different subgraph
-offload computation of the subgraph to TF during Inference.

Some of the potential reasons for handling custom layers are taking benefit from unsupported layers, getting useful information, adding new features.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were as following :
- Before conversion :
I run the pre-trained model using openCV on my CPU computer and evaluate the model speed through inference time calculation. [see test_opencv.py]
For accuracy I relied on frames counting
- After conversion
I used the OpenVINO™ Toolkit to evaluate the same parameters

The difference between model accuracy pre- and post-conversion was about:
0.64 - 0.62 = 0.02

The size of the model pre- and post-conversion was : 69.7M(pre-conversion) and 65M post conversion

The inference time of the model pre- and post-conversion was about 187ms pre-conversion and about 65ms post conversion

## Assess Model Use Cases

Some of the potential use cases of the people counter app are :
1.Counting people in stadium : the people counter app is pretty much adapted for counting huge crowd of people in places like stadium. 
2.Absence and presence detection in a classroom :
With some additional custom feature recognition the people counter app can help in identifying and counting absent student as well as present one in a given class.

Each of these use cases would be useful because counting a huge number of people or recognizing someone in that situation can be an overwelming task.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. Some of the potential effects on the End user needs are :
For user with more resources available, we can go for higher accuracy but more resource-intensive app.
For remote purpose and in case of lower-power devices, some accuracy will likely be sacrificed for lighter and faster app.
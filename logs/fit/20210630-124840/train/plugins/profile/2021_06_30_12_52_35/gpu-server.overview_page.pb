?	t??P?#y@t??P?#y@!t??P?#y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCt??P?#y@sJ@L¡c@1J???fn@A?ip[[??I????.???rEagerKernelExecute 0*	"??~???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??Gߤ)3@!?@de??X@)??Gߤ)3@1?@de??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?{eު???!0]B?m	??)?{eު???10]B?m	??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??_?????!_?Y3????)???????1?? UG??:Preprocessing2F
Iterator::ModelT??b???!???????)am???l?1?H??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap]?&?*3@!???9??X@)RH2?w?]?1????Z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIF??a9?C@Q??i??;N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	sJ@L¡c@sJ@L¡c@!sJ@L¡c@      ??!       "	J???fn@J???fn@!J???fn@*      ??!       2	?ip[[???ip[[??!?ip[[??:	????.???????.???!????.???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qF??a9?C@y??i??;N@?"k
=gradient_tape/model_35/conv2d_950/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ???Ȳ?! ???Ȳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam????fD??!??P2???"k
=gradient_tape/model_35/conv2d_949/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??r<??!??z'????08"]
<gradient_tape/model_35/max_pooling2d_140/MaxPool/MaxPoolGradMaxPoolGradz??c? ??!F'?ӡ???"8
model_35/conv2d_968/Conv2DConv2D?g?̼??!@?U;???0"8
model_35/conv2d_971/Conv2DConv2D)???+u??!E??????0"8
model_35/conv2d_962/Conv2DConv2D*e2l????!???????0"i
=gradient_tape/model_35/conv2d_956/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\Uz????!?>???_??0"i
=gradient_tape/model_35/conv2d_962/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??86??!D?%.???0"k
=gradient_tape/model_35/conv2d_968/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterTr??????!ib??[??08I2??.x0A@Qg???gP@Y?c5?25??a8?????X@q????h6$@yADC??;P?"?	
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?10.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
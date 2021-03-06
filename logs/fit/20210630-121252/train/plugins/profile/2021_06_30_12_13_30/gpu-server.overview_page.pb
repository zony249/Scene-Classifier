?	?w??x@?w??x@!?w??x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?w??x@?2??Ec@1??.\:n@A?e??E??I???s? @rEagerKernelExecute 0*	/?$??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator? ?bG3=@!???m*?X@)? ?bG3=@1???m*?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????????!)焢4??)????????1)焢4??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!?/RAh??)2?3/?݇?1?_Mh??:Preprocessing2F
Iterator::Model!???0??!W?;q?e??);s	??k?1^?M?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMaps?V{?3=@!?????X@)???2#b?1xߑ6??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?[z??C@Q?N??_N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2??Ec@?2??Ec@!?2??Ec@      ??!       "	??.\:n@??.\:n@!??.\:n@*      ??!       2	?e??E???e??E??!?e??E??:	???s? @???s? @!???s? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?[z??C@y?N??_N@?"k
=gradient_tape/model_10/conv2d_275/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???*???!???*???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam??I?E??!P?@ԑ{??"k
=gradient_tape/model_10/conv2d_274/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??v???!?3?
̵??08"\
;gradient_tape/model_10/max_pooling2d_40/MaxPool/MaxPoolGradMaxPoolGradEjػ?×?!DAj?C???"8
model_10/conv2d_293/Conv2DConv2Dj?i?MB??!?}7S????0"8
model_10/conv2d_296/Conv2DConv2D7?r???!?
y??0"8
model_10/conv2d_287/Conv2DConv2Dq؊g??!9?`[?E??0"i
=gradient_tape/model_10/conv2d_281/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????????!?yq?-??0"i
=gradient_tape/model_10/conv2d_287/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterd	{????!S??q???0"k
=gradient_tape/model_10/conv2d_293/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2?w??[??!?΀?-#??08I?1ϰ?_A@Qg??!PP@Y?c5?25??a8?????X@q(]???@y??wɉP?"?
both?Your program is POTENTIALLY input-bound because 38.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 
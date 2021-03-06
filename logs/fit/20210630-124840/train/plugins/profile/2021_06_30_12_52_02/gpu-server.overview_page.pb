?	?-???x@?-???x@!?-???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?-???x@ۆQ?fc@12??Y@n@Ao?e?????I#K?X^ @rEagerKernelExecute 0*	?v???)?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?^?D?;@!h?g???X@)?^?D?;@1h?g???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch{???w???!?9?a?Ұ?){???w???1?9?a?Ұ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?=?N???!? ?y?b??)$??????1???/? ??:Preprocessing2F
Iterator::Model?Z(??ڡ?!]L? ??)\?J?p?1}?p?W???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?$????;@!?٤???X@)??ǘ??`?1??L??}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?dB?C@Q?Q???MN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ۆQ?fc@ۆQ?fc@!ۆQ?fc@      ??!       "	2??Y@n@2??Y@n@!2??Y@n@*      ??!       2	o?e?????o?e?????!o?e?????:	#K?X^ @#K?X^ @!#K?X^ @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?dB?C@y?Q???MN@?"k
=gradient_tape/model_34/conv2d_923/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]҅?6ܲ?!]҅?6ܲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?hvH_??!??	???"k
=gradient_tape/model_34/conv2d_922/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJ?yV??!?
ӕe???08"]
<gradient_tape/model_34/max_pooling2d_136/MaxPool/MaxPoolGradMaxPoolGrad?1 7s???!???????"8
model_34/conv2d_941/Conv2DConv2D????}??!X??,p???0"8
model_34/conv2d_944/Conv2DConv2D?>tx?e??!0[/???0"8
model_34/conv2d_935/Conv2DConv2D	TN۫q??!?%m?d}??0"i
=gradient_tape/model_34/conv2d_929/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?>>K?)??!??? ?B??0"i
=gradient_tape/model_34/conv2d_935/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter8??%????!?Ă, ??0"k
=gradient_tape/model_34/conv2d_944/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterjN?ھ???!k6lpI??08I??4A@Q>z-??eP@Y?c5?25??a8?????X@q??B@y!~?|%?N?"?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
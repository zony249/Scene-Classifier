?	{K9_lgy@{K9_lgy@!{K9_lgy@	 ?L??p? ?L??p?! ?L??p?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL{K9_lgy@H?S?d@1B?Ѫen@A)??????I?/K;5@Yi??Q???rEagerKernelExecute 0*	t????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Yf?	5@!n;?_??X@)?Yf?	5@1n;?_??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchc?#?w~??!?Xb?????)c?#?w~??1?Xb?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?[?tYL??!3-??????)Q??Û??1??+|???:Preprocessing2F
Iterator::ModelI?Vџ?!?ԗ?A???)?PS?'l?1?< ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????
5@!4ߏ?X@)????^?1???O??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ?L??p?I?G!D@QzF??X?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?S?d@H?S?d@!H?S?d@      ??!       "	B?Ѫen@B?Ѫen@!B?Ѫen@*      ??!       2	)??????)??????!)??????:	?/K;5@?/K;5@!?/K;5@B      ??!       J	i??Q???i??Q???!i??Q???R      ??!       Z	i??Q???i??Q???!i??Q???b      ??!       JGPUY ?L??p?b q?G!D@yzF??X?M@?"k
=gradient_tape/model_36/conv2d_977/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterhL?1????!hL?1????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam??D]??!??u}???"k
=gradient_tape/model_36/conv2d_976/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?@?[??!??S+
???08"]
<gradient_tape/model_36/max_pooling2d_144/MaxPool/MaxPoolGradMaxPoolGrad?2??꼘?!C??????"8
model_36/conv2d_998/Conv2DConv2Dq(?B?x??!Q??o????0"8
model_36/conv2d_995/Conv2DConv2D?o??p??!???????0"8
model_36/conv2d_989/Conv2DConv2D???Er[??!?b\KC???0"i
=gradient_tape/model_36/conv2d_983/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?tx???!;?jz?X??0"i
=gradient_tape/model_36/conv2d_989/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?(yY??!???T???0"k
=gradient_tape/model_36/conv2d_995/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?w?? ???!(??\?U??08I???l @A@Q&????_P@Y?c5?25??a8?????X@q?R?R@yT?Ρ4?N?"?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
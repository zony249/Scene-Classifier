?	???2?oy@???2?oy@!???2?oy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???2?oy@ک??`ad@1Z	?%?Dn@A4?f??IvöE???rEagerKernelExecute 0*	??C+?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorퟧ??3@!%O????X@)ퟧ??3@1%O????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch¡?xxϑ?!g_3`?ʶ?)¡?xxϑ?1g_3`?ʶ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?ʄ_????!??l"??)&R???0??1R?ы????:Preprocessing2F
Iterator::Modelr?	?OƠ?!kx?Q1w??)st??%m?1Ͳ?#???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap}?q ?3@!?-WgD?X@)????[_?1???{m??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?ޞ???D@QB!a[$?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ک??`ad@ک??`ad@!ک??`ad@      ??!       "	Z	?%?Dn@Z	?%?Dn@!Z	?%?Dn@*      ??!       2	4?f??4?f??!4?f??:	vöE???vöE???!vöE???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ޞ???D@yB!a[$?M@?"l
>gradient_tape/model_42/conv2d_1139/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter^??mW???!^??mW???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?~ ??l??!?W????"l
>gradient_tape/model_42/conv2d_1138/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??5????!?h?R????08"]
<gradient_tape/model_42/max_pooling2d_168/MaxPool/MaxPoolGradMaxPoolGradچ??f??!^?P????"9
model_42/conv2d_1157/Conv2DConv2D???A瀗?!??Q8????0"9
model_42/conv2d_1160/Conv2DConv2D@?56|d??!??-???0"9
model_42/conv2d_1151/Conv2DConv2D?p⇾??!*?Y?????0"j
>gradient_tape/model_42/conv2d_1145/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?O?????!h?LWL??0"j
>gradient_tape/model_42/conv2d_1151/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterEó????!B?G5???0"l
>gradient_tape/model_42/conv2d_1160/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!????6P??08IQ"?K>A@Q??2s?`P@Y?c5?25??a8?????X@qP?????@y|??n?O?"?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
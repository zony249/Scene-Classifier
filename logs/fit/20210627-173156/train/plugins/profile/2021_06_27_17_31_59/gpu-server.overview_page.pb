?	qU?w??o@qU?w??o@!qU?w??o@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCqU?w??o@??el??Q@1ę_́?f@Ap?DIH???I?W)???rEagerKernelExecute 0*	~j?t???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??_̖3@!%هd??X@)??_̖3@1%هd??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??C?b??!???@???)??C?b??1???@???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismc??Ց??!?ML?????)?[<?????1???3?y??:Preprocessing2F
Iterator::Model?KTol??!p?%?~???)?d??~?m?13??.fW??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???f?3@!	??@?X@)
?2?&W?1????7~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIx`~?Q<@Q??g`??Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??el??Q@??el??Q@!??el??Q@      ??!       "	ę_́?f@ę_́?f@!ę_́?f@*      ??!       2	p?DIH???p?DIH???!p?DIH???:	?W)????W)???!?W)???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qx`~?Q<@y??g`??Q@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdam|OaPl??!|OaPl??"g
;gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?2~?Q??!??o_??0"8
model_3/conv2d_83/Conv2DConv2D??!?Iڟ?!݆kԪ??08"e
:gradient_tape/model_3/conv2d_83/Conv2D/Conv2DBackpropInputConv2DBackpropInput??%b???!???????0"e
:gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropInputConv2DBackpropInput?~?????!?u?????0"6
model_3/conv2d_89/Conv2DConv2D??Z????!?Rջ???0"0
Adam/gradients/AddN_31AddN?m?xH???!j9b?????"g
;gradient_tape/model_3/conv2d_89/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????????!B?=?????0"G
.gradient_tape/dense_9/kernel/Regularizer/Mul_1Mul??_߶??!??3??"i
;gradient_tape/model_3/conv2d_82/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?E??m֗?!???s???08Iq?=??B@Q???PO@Y??A?????aL?:,??X@q ? l?
@y ???sR?"?
both?Your program is POTENTIALLY input-bound because 27.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
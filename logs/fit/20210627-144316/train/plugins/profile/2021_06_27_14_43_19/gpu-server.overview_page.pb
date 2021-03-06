?	?`?,?q@?`?,?q@!?`?,?q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?`?,?q@d????kX@1??O=?f@AH?]?ۥ?IQ?|ar??rEagerKernelExecute 0*	C?l?K??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorpΈ?ޘ=@!??ٻZ?X@)pΈ?ޘ=@1??ٻZ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchd?? w??!??5c????)d?? w??1??5c????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismdyW=`??!???C???)?J̳?V??1L\?A|???:Preprocessing2F
Iterator::ModelལƄ???!??????)?GĔH?g?1Zl,
????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap5&?\R?=@!??2??X@)?8?Վ?\?121
I?]x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???"?A@Q????1P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d????kX@d????kX@!d????kX@      ??!       "	??O=?f@??O=?f@!??O=?f@*      ??!       2	H?]?ۥ?H?]?ۥ?!H?]?ۥ?:	Q?|ar??Q?|ar??!Q?|ar??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???"?A@y????1P@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdamک?n??!ک?n??"g
;gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\?\!W??!?S??2??0"e
:gradient_tape/model_1/conv2d_35/Conv2D/Conv2DBackpropInputConv2DBackpropInput??F+?ɟ?!z^?B????0"8
model_1/conv2d_35/Conv2DConv2D???Xí??!????U???08"e
:gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropInputConv2DBackpropInput9v5???!?Y?????0"0
Adam/gradients/AddN_31AddN??,????!?\?????"g
;gradient_tape/model_1/conv2d_41/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter"윋	???!&??g???0"6
model_1/conv2d_41/Conv2DConv2D??@r??!aM??Wr??0"G
.gradient_tape/dense_3/kernel/Regularizer/Mul_1Mul?Rf6p+??!??????"i
;gradient_tape/model_1/conv2d_34/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterK???%???!??y?@s??08I[?????B@Q?dO@Y??A?????aL?:,??X@q8W??"@yi??pR?"?
both?Your program is POTENTIALLY input-bound because 34.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
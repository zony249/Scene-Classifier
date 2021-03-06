?	@???|??@@???|??@!@???|??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC@???|??@?˸??W@14-?2??}@Ah>?nף?I??3?cV??rEagerKernelExecute 0*	?S??[1?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorzUg???9@!?zl???X@)zUg???9@1?zl???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??*Q????![sC???)??*Q????1[sC???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?q?@H??!^Xm,?-??)??0a4+??1????5T??:Preprocessing2F
Iterator::Model???^Dۡ?!??U??M??)? 3??Ol?1S??A?o??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[^??6?9@!ղY?X@)T8?T?]?12?????|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIT??'|?0@Qk[? ?T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?˸??W@?˸??W@!?˸??W@      ??!       "	4-?2??}@4-?2??}@!4-?2??}@*      ??!       2	h>?nף?h>?nף?!h>?nף?:	??3?cV????3?cV??!??3?cV??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qT??'|?0@yk[? ?T@?"9
model_5/conv2d_143/Conv2DConv2D??ͱ?̥?!??ͱ?̥?08"j
<gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2?ڈ??!O??+Ī??08"f
;gradient_tape/model_5/conv2d_143/Conv2D/Conv2DBackpropInputConv2DBackpropInput????4???!??v????0"7
model_5/conv2d_137/Conv2DConv2Dvf????!?gi?????0"j
<gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5?????!??L?+???08"h
<gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'??i2+??!Ud?)????0"f
;gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropInputConv2DBackpropInput???]c??!???g???0"f
;gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropInputConv2DBackpropInput	???a??!k?YƋ???0"7
model_5/conv2d_131/Conv2DConv2Dp?_>??!n\??W4??0"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdamu ?Z?Ο?!u?B?E1??I??ڼ?m3@Q?D?Л$T@Y?^?????a???'?X@q?ag??	@y?S$?<?"?
both?Your program is POTENTIALLY input-bound because 16.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	?=@??Ԁ@?=@??Ԁ@!?=@??Ԁ@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?=@??Ԁ@L5??lr@1
?28n@A?U,~SX??I6\䞮? @rEagerKernelExecute 0*	P??n???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorZEh??2@!,??,0?X@)ZEh??2@1,??,0?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??}?<??!??O̞F??)??}?<??1??O̞F??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismÜ?M??!3?=wg??)"S>U???1ryWD??:Preprocessing2F
Iterator::Model?fh<??!??n????)?W?\Tk?1/???/???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???E?2@!???4??X@)??? !?W?1??Y?A?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI[???ӎK@Q?67,qF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L5??lr@L5??lr@!L5??lr@      ??!       "	
?28n@
?28n@!
?28n@*      ??!       2	?U,~SX???U,~SX??!?U,~SX??:	6\䞮? @6\䞮? @!6\䞮? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q[???ӎK@y?67,qF@?"j
<gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9?T?ò?!9?T?ò?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamq*Fɍ??!C?e???"j
<gradient_tape/model_5/conv2d_139/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??-N9???!)^?????08"[
:gradient_tape/model_5/max_pooling2d_20/MaxPool/MaxPoolGradMaxPoolGrad^fB9z??!???#????"7
model_5/conv2d_161/Conv2DConv2Dw??????!?? .???0"7
model_5/conv2d_158/Conv2DConv2DɊl	???!??????0"7
model_5/conv2d_152/Conv2DConv2DR?k?|<??!?j]?a??0"h
<gradient_tape/model_5/conv2d_146/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??]????!?	? ??0"h
<gradient_tape/model_5/conv2d_152/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterF??~????!?0zL????0"j
<gradient_tape/model_5/conv2d_158/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)??|^??!???1??08I?4s??cA@Q?eF?NP@Y?c5?25??a8?????X@qc?s??%@y:??ɦ[O?"?	
both?Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?10.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
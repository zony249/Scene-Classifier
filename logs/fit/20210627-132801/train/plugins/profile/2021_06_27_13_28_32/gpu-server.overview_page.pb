?	???x?y@???x?y@!???x?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???x?y@??w?-`R@1?XR??Pt@A??%?ɦ?I??!??`??rEagerKernelExecute 0*	n??f?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?u?B?F@!?m??X@)?u?B?F@1?m??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?X?|^??!??B?@??)?X?|^??1??B?@??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismw???閝?!??aHf??)??]?p??1?????:Preprocessing2F
Iterator::Model??ܠ?!???а??)?N???p?1?~?HDT??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?g???F@!>??S?X@)(??ȯ_?1 ???Տq?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIhژ? ?2@Qf??7OT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??w?-`R@??w?-`R@!??w?-`R@      ??!       "	?XR??Pt@?XR??Pt@!?XR??Pt@*      ??!       2	??%?ɦ???%?ɦ?!??%?ɦ?:	??!??`????!??`??!??!??`??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qhژ? ?2@yf??7OT@?"h
<gradient_tape/model_7/conv2d_190/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ζe֮?!?ζe֮?0"h
<gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?7??섥?!?:??-??0"f
;gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropInputConv2DBackpropInput??K?|{??!?-𡳵??0"7
model_7/conv2d_179/Conv2DConv2D?C:	z??!??>$<??0"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdam߱?]4o??!?*?;ߗ??"j
<gradient_tape/model_7/conv2d_178/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?9?Z?ϛ?!:????08"f
;gradient_tape/model_7/conv2d_178/Conv2D/Conv2DBackpropInputConv2DBackpropInput?!T?????!;??Qָ??0"h
<gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?30?????!x΂?d??0"7
model_7/conv2d_178/Conv2DConv2D?a?%2???!?$?/8???0"h
;gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropInputConv2DBackpropInput?+??'??!O?v᱑??08I?Q)?F@Q]????K@Y??A?????aL?:,??X@q:?ʜx@y3&?¼?D?"?
both?Your program is POTENTIALLY input-bound because 18.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	?0?x@?0?x@!?0?x@	^?H+??^?H+??!^?H+??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?0?x@Ӣ>??b@1?'*?Sn@A`?o`r???I?M?G????YbK??z2??rEagerKernelExecute 0*	?A`?X??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????OR5@!?.?@?X@)????OR5@1?.?@?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??·g	??!??w???)??·g	??1??w???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-????!??$F???)???Hhˉ?1 ????1??:Preprocessing2F
Iterator::Modelm 6 B\??!????R??)0???DKn?1J?t?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap<?b??R5@!q????X@)????`?1^,ֽ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9^?H+??I?#?DJC@Qӗ?I??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ӣ>??b@Ӣ>??b@!Ӣ>??b@      ??!       "	?'*?Sn@?'*?Sn@!?'*?Sn@*      ??!       2	`?o`r???`?o`r???!`?o`r???:	?M?G?????M?G????!?M?G????B      ??!       J	bK??z2??bK??z2??!bK??z2??R      ??!       Z	bK??z2??bK??z2??!bK??z2??b      ??!       JGPUY^?H+??b q?#?DJC@yӗ?I??N@?"l
>gradient_tape/model_37/conv2d_1004/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???3.ز?!???3.ز?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?f?^??!,a?????"l
>gradient_tape/model_37/conv2d_1003/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ???d??!։?qg???08"]
<gradient_tape/model_37/max_pooling2d_148/MaxPool/MaxPoolGradMaxPoolGrad???`Q???!??ڝQ???"9
model_37/conv2d_1022/Conv2DConv2D??????![=?@????0"9
model_37/conv2d_1025/Conv2DConv2D???+"~??!2?k?v???0"9
model_37/conv2d_1016/Conv2DConv2DSR??5???!|???????0"j
>gradient_tape/model_37/conv2d_1010/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?T?}t??!?y?+r??0"j
>gradient_tape/model_37/conv2d_1016/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4??????!?u??b??0"l
>gradient_tape/model_37/conv2d_1022/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterN?X?ۙ??!??? b??08I2?%?7A@Qg,??ydP@Y?c5?25??a8?????X@ql?????5@yee??`N?"?	
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?21.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
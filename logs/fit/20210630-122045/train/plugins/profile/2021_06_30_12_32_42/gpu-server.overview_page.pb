?	???,y@???,y@!???,y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???,y@4?f??c@1?????Mn@A)?7Ӆ??I???????rEagerKernelExecute 0*	?rh??t?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???v?2@!煱,r?X@)???v?2@1煱,r?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchS????g??!9ļ??X??)S????g??19ļ??X??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismjN^d~??!?Y^???).rOWw,??1(????T??:Preprocessing2F
Iterator::Modelq???h??!???????)|?ԗ??j?1?]?m????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap~U.T??2@!=??%?X@)˻??`?1AĊͪk??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?ۏ??C@Qn?$pN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4?f??c@4?f??c@!4?f??c@      ??!       "	?????Mn@?????Mn@!?????Mn@*      ??!       2	)?7Ӆ??)?7Ӆ??!)?7Ӆ??:	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ۏ??C@yn?$pN@?"k
=gradient_tape/model_21/conv2d_572/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1??ֲ?!1??ֲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam???){ߞ?!"~Ş???"k
=gradient_tape/model_21/conv2d_571/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?{???!Rt??n???08"\
;gradient_tape/model_21/max_pooling2d_84/MaxPool/MaxPoolGradMaxPoolGrad#>????!?n&???"8
model_21/conv2d_590/Conv2DConv2D? ?<???!j??-???0"8
model_21/conv2d_593/Conv2DConv2D??G)Lq??!\?;$W???0"8
model_21/conv2d_584/Conv2DConv2D?U0?m???!?a?$???0"i
=gradient_tape/model_21/conv2d_578/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ׂ]???!
7??x??0"i
=gradient_tape/model_21/conv2d_584/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?&R?	???!?=~???0"k
=gradient_tape/model_21/conv2d_593/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?̤!????!?????c??08IZ?%CA@Q?,?w~^P@Y?c5?25??a8?????X@q??D-c?9@y8/???3O?"?	
both?Your program is POTENTIALLY input-bound because 39.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?25.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
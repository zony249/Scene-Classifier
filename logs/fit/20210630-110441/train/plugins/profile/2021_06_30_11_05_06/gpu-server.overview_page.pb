?	Q1?߄?n@Q1?߄?n@!Q1?߄?n@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCQ1?߄?n@?}?e??@1???Rn@A??r-Z???I/???u@rEagerKernelExecute 0*	A`??j??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?\?B@!I?^}?X@)?\?B@1I?^}?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?U??{??!$]?N????)?U??{??1$]?N????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??sCS??!?~???Ѿ?)??H?+??1?Cj??R??:Preprocessing2F
Iterator::Model????	???!lj??????)x?W?f,j?1??J????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapl$	?B@!K???X@)?'*?TV?1?????n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@? x????Q{}.I?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}?e??@?}?e??@!?}?e??@      ??!       "	???Rn@???Rn@!???Rn@*      ??!       2	??r-Z?????r-Z???!??r-Z???:	/???u@/???u@!/???u@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@? x????y{}.I?X@?"f
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? C??ݲ?!? C??ݲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamѲ?]?-??!G??Cp???"f
8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??
???!!?-?K???08"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrady?ț???!?צ???"4
model/conv2d_23/Conv2DConv2Dԭ?%⬗?!?mY[????0"4
model/conv2d_26/Conv2DConv2D<??????!2_=?_???0"4
model/conv2d_17/Conv2DConv2D!??'???!vB??????0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???	?2??!?34??a??0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter<??l???!?}??c??0"g
9gradient_tape/model/conv2d_26/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?!dT????!տ?A#Z??08I?b?j0A@Q?N}??gP@Y?c5?25??a8?????X@q???s???y?&z@O?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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
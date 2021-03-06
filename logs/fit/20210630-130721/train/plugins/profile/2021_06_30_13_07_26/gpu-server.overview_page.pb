?	?^~???y@?^~???y@!?^~???y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?^~???y@??^a??d@1|?ycn@A??Pn????Iޓ??Z???rEagerKernelExecute 0*	??Q0??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??O@?=@!rhG???X@)??O@?=@1rhG???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchsG?˵h??!>???Y??)sG?˵h??1>???Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism]p????!(Uo????)?Qf`??1=l?????:Preprocessing2F
Iterator::Model??D?֠?!?=??d??)+?)?Tp?1
Cϡn???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapyܝ??=@!??^???X@)3?&c`]?1?Oۅ?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI/???D@Q??Ws?cM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??^a??d@??^a??d@!??^a??d@      ??!       "	|?ycn@|?ycn@!|?ycn@*      ??!       2	??Pn??????Pn????!??Pn????:	ޓ??Z???ޓ??Z???!ޓ??Z???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q/???D@y??Ws?cM@?"l
>gradient_tape/model_56/conv2d_1517/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???۲?!???۲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam???N???!m,?n????"l
>gradient_tape/model_56/conv2d_1516/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteryғ0T???!?g=????08"]
<gradient_tape/model_56/max_pooling2d_224/MaxPool/MaxPoolGradMaxPoolGrad]?=?????!??.<????"9
model_56/conv2d_1535/Conv2DConv2D(@??A??!??~???0"9
model_56/conv2d_1538/Conv2DConv2D,[?}?A??!???nK???0"9
model_56/conv2d_1529/Conv2DConv2Dc\T?V??!?)y'???0"j
>gradient_tape/model_56/conv2d_1523/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterI??K??!????K??0"j
>gradient_tape/model_56/conv2d_1529/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?V???ܕ?!?8????0"l
>gradient_tape/model_56/conv2d_1538/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?X{????!???;?L??08I?kBA@Q(????^P@Y?c5?25??a8?????X@qZ?????@y?1Ş`O?"?
both?Your program is POTENTIALLY input-bound because 40.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
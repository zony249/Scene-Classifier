?	???Ry@???Ry@!???Ry@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???Ry@TUh ,d@1M??E?n@A?e0F$
??I?&1????rEagerKernelExecute 0*	??~j?Q?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?)???<@!D???R?X@)?)???<@1D???R?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?"1Aߒ?!@&???D??)?"1Aߒ?1@&???D??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??˶?֠?!s?#???)? ?X4???1??T??:Preprocessing2F
Iterator::Modelo+?6??!Ţ??x??)???2#r?1??}?E??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapZ??/-?<@!?.3???X@)?!9??U`?1$g?a *|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIЖ?k?#D@Q0i>?$?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	TUh ,d@TUh ,d@!TUh ,d@      ??!       "	M??E?n@M??E?n@!M??E?n@*      ??!       2	?e0F$
???e0F$
??!?e0F$
??:	?&1?????&1????!?&1????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qЖ?k?#D@y0i>?$?M@?"k
=gradient_tape/model_33/conv2d_896/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?n+?ڲ?!?n+?ڲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamb?mZ??!Z6?,
???"k
=gradient_tape/model_33/conv2d_895/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterDQ?G?H??!V?9?????08"8
model_33/conv2d_914/Conv2DConv2D4?v????!<?H?????0"]
<gradient_tape/model_33/max_pooling2d_132/MaxPool/MaxPoolGradMaxPoolGrad?-E䁗?!?\????"8
model_33/conv2d_917/Conv2DConv2D??j?y??!??j????0"8
model_33/conv2d_908/Conv2DConv2D??x?!2@?(E???0"i
=gradient_tape/model_33/conv2d_902/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^+?0??!??\V??0"i
=gradient_tape/model_33/conv2d_908/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? *?????!?O<
??0"k
=gradient_tape/model_33/conv2d_914/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?cu? ???!LN?X\S??08I???	4A@Q?/?%?eP@Y?c5?25??a8?????X@q9?????@yC????P?"?
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
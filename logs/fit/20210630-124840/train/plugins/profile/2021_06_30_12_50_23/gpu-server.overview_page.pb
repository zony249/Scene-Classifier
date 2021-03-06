?	9?????x@9?????x@!9?????x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC9?????x@Z????Uc@1?J?8?<n@Ag???d??IAG?Z???rEagerKernelExecute 0*	(1??T?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????=@!?laTK?X@)????=@1?laTK?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??E}?;??!??Ç)???)O??Z}??1?Դn????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?D?[????!??A????)?D?[????1??A????:Preprocessing2F
Iterator::Model}?Жs)??!m?<r??)????n?1?ѻ4???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?	j??=@!Jx????X@)Y4???b?1I???N?~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIa%????C@Q??>?[N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z????Uc@Z????Uc@!Z????Uc@      ??!       "	?J?8?<n@?J?8?<n@!?J?8?<n@*      ??!       2	g???d??g???d??!g???d??:	AG?Z???AG?Z???!AG?Z???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qa%????C@y??>?[N@?"k
=gradient_tape/model_31/conv2d_842/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterxP 96ʲ?!xP 96ʲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamY	+?5q??!?+?????"k
=gradient_tape/model_31/conv2d_841/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter;???ޛ?!?i(n???08"]
<gradient_tape/model_31/max_pooling2d_124/MaxPool/MaxPoolGradMaxPoolGrad?~?q???!?YY<????"8
model_31/conv2d_860/Conv2DConv2D2??1~??!ώb????0"8
model_31/conv2d_863/Conv2DConv2D?q??:}??!B?ĸI???0"8
model_31/conv2d_854/Conv2DConv2D?$?)?g??!?!?]Fy??0"i
=gradient_tape/model_31/conv2d_848/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter⼀+-??!???>??0"i
=gradient_tape/model_31/conv2d_854/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????????!?K	W????0"k
=gradient_tape/model_31/conv2d_860/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw???????!?{?G??08Ih2??%8A@Q?f??cP@Y?c5?25??a8?????X@q0?J+?@y䮡??VO?"?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
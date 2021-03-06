?	?f?l?G{@?f?l?G{@!?f?l?G{@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?f?l?G{@jܛ߰?f@1? ????o@A?̒ 5???I`?eM,???rEagerKernelExecute 0*	C?l?C??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatort???F=@!??H?X@)t???F=@1??H?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?yq???!???!????)?yq???1???!????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??RB????!
l??'??)?"j??G??1?(
????:Preprocessing2F
Iterator::Model*?dq????!>X???)?܁:?q?1P?]P???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??K?[G=@!y?????X@)????&?a?1??}?K?~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIy???D@Q??]?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	jܛ߰?f@jܛ߰?f@!jܛ߰?f@      ??!       "	? ????o@? ????o@!? ????o@*      ??!       2	?̒ 5????̒ 5???!?̒ 5???:	`?eM,???`?eM,???!`?eM,???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qy???D@y??]?M@?"l
>gradient_tape/model_38/conv2d_1031/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???.߱?!???.߱?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?I%L@ޝ?!p*?A?V??"9
model_38/conv2d_1049/Conv2DConv2D?(??`8??!J??5^??0"l
>gradient_tape/model_38/conv2d_1030/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Sudh??!-S??j_??08"9
model_38/conv2d_1043/Conv2DConv2D???`??!Jͅ?l???0"j
>gradient_tape/model_38/conv2d_1043/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?U?1??!!ns???0"]
<gradient_tape/model_38/max_pooling2d_152/MaxPool/MaxPoolGradMaxPoolGradk??]Η?!?g@?>???"h
=gradient_tape/model_38/conv2d_1049/Conv2D/Conv2DBackpropInputConv2DBackpropInput?,??O??!???41???0"h
=gradient_tape/model_38/conv2d_1052/Conv2D/Conv2DBackpropInputConv2DBackpropInput:G??(???!;?N#?G??0"9
model_38/conv2d_1052/Conv2DConv2D?K?@N??!?;$Ϭ??0I??;??gA@Q3*?=1LP@Y?c5?25??a8?????X@q'?J4_6@yr?!ۦ?O?"?	
both?Your program is POTENTIALLY input-bound because 41.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?22.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
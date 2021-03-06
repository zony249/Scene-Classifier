?	}?r???x@}?r???x@!}?r???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC}?r???x@?/?ýb@1?0?q?"n@A?D???V??I??I`sN??rEagerKernelExecute 0*	???{s?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatornk?K%?@!t/\??X@)nk?K%?@1t/\??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?D?A???!?:+?X???)?D?A???1?:+?X???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism_(`;???!3d˗?h??)????W??1??k??P??:Preprocessing2F
Iterator::Modelb0?̕??!k?|??2??)&??|?k?1?I??O??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap/??%?@!ՠ?Y??X@)7???-_?1?X?X?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI!???SQC@Q?c*??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?/?ýb@?/?ýb@!?/?ýb@      ??!       "	?0?q?"n@?0?q?"n@!?0?q?"n@*      ??!       2	?D???V???D???V??!?D???V??:	??I`sN????I`sN??!??I`sN??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q!???SQC@y?c*??N@?"l
>gradient_tape/model_44/conv2d_1193/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?n?T????!?n?T????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?z7}????!jM??F˺?"l
>gradient_tape/model_44/conv2d_1192/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??a?W??!???k????08"]
<gradient_tape/model_44/max_pooling2d_176/MaxPool/MaxPoolGradMaxPoolGradƮ???y??!e:??????"9
model_44/conv2d_1211/Conv2DConv2D)??h[??!???7???0"9
model_44/conv2d_1214/Conv2DConv2Dɾ??gS??!a?񾤬??0"9
model_44/conv2d_1205/Conv2DConv2DW6ҽN??!,4y|v??0"j
>gradient_tape/model_44/conv2d_1199/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltera￁?,??!li<??0"j
>gradient_tape/model_44/conv2d_1205/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?!??.??!'?ࠩ???0"l
>gradient_tape/model_44/conv2d_1214/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?=f?{??!??D?bG??08I?Ǝ?a9A@Q???
OcP@Y?c5?25??a8?????X@qpU[?--@y?<??sO?"?	
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
Refer to the TF2 Profiler FAQb?14.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
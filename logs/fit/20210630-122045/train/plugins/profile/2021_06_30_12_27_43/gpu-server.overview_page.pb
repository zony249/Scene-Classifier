?	K;5?ۣy@K;5?ۣy@!K;5?ۣy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCK;5?ۣy@???a?d@1?ht?zn@AӾ??z??I?֍w???rEagerKernelExecute 0*	?"?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?A?L1D@!???x?X@)?A?L1D@1???x?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\?	????!??????)\?	????1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismٳ?25	??!SB??Γ??)??Rb׆?1-???CA??:Preprocessing2F
Iterator::Modelo?????!h?d?ꖴ?)"8.??j?1??????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?9>Z?1D@!??DE??X@)*T7?c?1??T?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???HD@Q&?o??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???a?d@???a?d@!???a?d@      ??!       "	?ht?zn@?ht?zn@!?ht?zn@*      ??!       2	Ӿ??z??Ӿ??z??!Ӿ??z??:	?֍w????֍w???!?֍w???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???HD@y&?o??M@?"k
=gradient_tape/model_12/conv2d_329/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Tz3Ӳ?!??Tz3Ӳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?Wt?M??!?߱?????"k
=gradient_tape/model_12/conv2d_328/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?q9???!?2Ε???08"\
;gradient_tape/model_12/max_pooling2d_48/MaxPool/MaxPoolGradMaxPoolGrad3?X??B??!lE?????"8
model_12/conv2d_347/Conv2DConv2D?>atlg??!Em?/????0"8
model_12/conv2d_350/Conv2DConv2D???Ig??!x?;Rȳ??0"8
model_12/conv2d_341/Conv2DConv2D4لCl"??!?`??x??0"i
=gradient_tape/model_12/conv2d_335/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter^?	????!
??b?:??0"i
=gradient_tape/model_12/conv2d_341/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?wTR?ו?!?l?????0"k
=gradient_tape/model_12/conv2d_350/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4???Az??!ܛ??B??08I???GA@Q?s?	\P@Y?c5?25??a8?????X@qظ???/@y?2dO?"?	
both?Your program is POTENTIALLY input-bound because 40.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?15.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
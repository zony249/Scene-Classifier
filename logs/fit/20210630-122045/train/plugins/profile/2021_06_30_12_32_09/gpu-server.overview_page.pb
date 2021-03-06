?	?b*=z@?b*=z@!?b*=z@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?b*=z@Vc	k?e@1?A_z{En@A{ܷZ'.??I?Y??/???rEagerKernelExecute 0*	???K???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator\?~l?>@!E?H??X@)\?~l?>@1E?H??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchl%t??Y??!?د/?i??)l%t??Y??1?د/?i??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisml?????!@{0Gѷ?)?????y??1???8??:Preprocessing2F
Iterator::Model?*8???!j%???g??)Z?Pۆq?1MQ?=????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap7??>@!7?&?X@)vq?-`?1????~z?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?x????D@QX?|g
M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Vc	k?e@Vc	k?e@!Vc	k?e@      ??!       "	?A_z{En@?A_z{En@!?A_z{En@*      ??!       2	{ܷZ'.??{ܷZ'.??!{ܷZ'.??:	?Y??/????Y??/???!?Y??/???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?x????D@yX?|g
M@?"k
=gradient_tape/model_20/conv2d_545/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?z??ز?!?z??ز?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamXT?hr\??!?LJ????"k
=gradient_tape/model_20/conv2d_544/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5??,???!Q??????08"\
;gradient_tape/model_20/max_pooling2d_80/MaxPool/MaxPoolGradMaxPoolGrad??j{_??!)(?????"8
model_20/conv2d_563/Conv2DConv2D??W??x??!;??????0"8
model_20/conv2d_566/Conv2DConv2D??4xU??!????c???0"8
model_20/conv2d_557/Conv2DConv2Dָ???k??!??)????0"i
=gradient_tape/model_20/conv2d_551/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??+??!(?eCQ??0"i
=gradient_tape/model_20/conv2d_557/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,|?˴???!?^D?<??0"k
=gradient_tape/model_20/conv2d_563/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter*?j?ۋ??!?˼?O??08I'{?5?9A@QmB?=cP@Y?c5?25??a8?????X@q??,??@y??U???P?"?
both?Your program is POTENTIALLY input-bound because 41.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
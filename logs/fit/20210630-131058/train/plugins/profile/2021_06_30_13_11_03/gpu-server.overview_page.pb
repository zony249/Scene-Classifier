?	FD1yy@FD1yy@!FD1yy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCFD1yy@|???ǟc@1??b?$n@A[??K????Ink?K??rEagerKernelExecute 0*	Y9?Ⱦ^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator9EGr?=@!??_0?X@)9EGr?=@1??_0?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?͌~4???!?q?????)?͌~4???1?q?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm?/?r??!?*?W?3??)'?E'K???1"rwX`??:Preprocessing2F
Iterator::Modeleު?PM??!p?Yǽ?)y=??p?1????????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap7??nf=@!???)??X@)?t?? ?[?1,?#}rw?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI|j?P_?C@Q??`??#N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|???ǟc@|???ǟc@!|???ǟc@      ??!       "	??b?$n@??b?$n@!??b?$n@*      ??!       2	[??K????[??K????![??K????:	nk?K??nk?K??!nk?K??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q|j?P_?C@y??`??#N@?"l
>gradient_tape/model_57/conv2d_1544/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQ??#伲?!Q??#伲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?qYGb???!?㖵|???"l
>gradient_tape/model_57/conv2d_1543/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteryD?l?ʛ?!o?f?????08"]
<gradient_tape/model_57/max_pooling2d_228/MaxPool/MaxPoolGradMaxPoolGradC9?Zpi??!?a??????"9
model_57/conv2d_1562/Conv2DConv2Dx,?m?T??!&gzab???0"9
model_57/conv2d_1565/Conv2DConv2D????O??!?$??C???0"9
model_57/conv2d_1556/Conv2DConv2D?G??|??!y7??z??0"j
>gradient_tape/model_57/conv2d_1550/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?d;?%??!y??????0"j
>gradient_tape/model_57/conv2d_1556/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?}?HT???!?_ο???0"l
>gradient_tape/model_57/conv2d_1565/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]Z(?j???!?"Rr?G??08IQ??3?DA@Qض;f?]P@Y?c5?25??a8?????X@qġHR?2@y?Gc??oO?"?	
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?19.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
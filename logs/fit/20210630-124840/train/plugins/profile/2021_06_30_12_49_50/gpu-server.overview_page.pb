?	?&c??x@?&c??x@!?&c??x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?&c??x@N`:?[nc@16?C6n@AV???4??I????? @rEagerKernelExecute 0*	bX9ds?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorQ?????@@!ў?bC?X@)Q?????@@1ў?bC?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???XǑ?!UO?u?b??)???XǑ?1UO?u?b??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?p???i??!????2ӵ?)N??oD??1ݟxU?C??:Preprocessing2F
Iterator::Model????=???!_/
???)ס????q?1.??#]??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?4?;?@@!?|?J??X@)X?%???c?1󝈷:}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??8??C@Qm??DN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N`:?[nc@N`:?[nc@!N`:?[nc@      ??!       "	6?C6n@6?C6n@!6?C6n@*      ??!       2	V???4??V???4??!V???4??:	????? @????? @!????? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??8??C@ym??DN@?"k
=gradient_tape/model_30/conv2d_815/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5??ײ?!5??ײ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamY?s/S??!K<??ҫ??"k
=gradient_tape/model_30/conv2d_814/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterc??G???!U??=???08"]
<gradient_tape/model_30/max_pooling2d_120/MaxPool/MaxPoolGradMaxPoolGrad???^?+??!?Ȱӷ???"8
model_30/conv2d_836/Conv2DConv2D?c?????!A?.-????0"8
model_30/conv2d_833/Conv2DConv2D???}??!_ѯ?Y???0"8
model_30/conv2d_827/Conv2DConv2D?v?*???!=@??^???0"i
=gradient_tape/model_30/conv2d_821/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery
?1+??!_?f"?V??0"i
=gradient_tape/model_30/conv2d_827/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterbO?7???!8??
??0"k
=gradient_tape/model_30/conv2d_836/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJNߙW???!?,F??S??08I?dS??8A@Q?Mֳ?cP@Y?c5?25??a8?????X@q0FS:{u@y??pԍP?"?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
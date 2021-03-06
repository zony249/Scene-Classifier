?	???ۗy@???ۗy@!???ۗy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???ۗy@?8?/?c@1?뤾?9n@A)??????I?y?~@rEagerKernelExecute 0*	?5^?i(?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatordx?g??3@!|????X@)dx?g??3@1|????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchǁW˝???!TBl??'??)ǁW˝???1TBl??'??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism{?%T??!?u-???)0?????1??}v??:Preprocessing2F
Iterator::Model?v???!?J?????)?t?? ?k?1??٤k\??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMape?3@!??.???X@)#??fF?Z?1]??be???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI3?u?C@Q???t?6N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8?/?c@?8?/?c@!?8?/?c@      ??!       "	?뤾?9n@?뤾?9n@!?뤾?9n@*      ??!       2	)??????)??????!)??????:	?y?~@?y?~@!?y?~@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q3?u?C@y???t?6N@?"i
;gradient_tape/model_2/conv2d_59/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW??????!W??????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam???j`G??!??x??r??"i
;gradient_tape/model_2/conv2d_58/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterjt??r???!vx1Z????08"Z
9gradient_tape/model_2/max_pooling2d_8/MaxPool/MaxPoolGradMaxPoolGrad?z?x???!?? y????"6
model_2/conv2d_77/Conv2DConv2D??z(?_??!?0?????0"6
model_2/conv2d_80/Conv2DConv2D?v
?H=??!g1?^???0"6
model_2/conv2d_71/Conv2DConv2D{@6?!?v??|T??0"g
;gradient_tape/model_2/conv2d_65/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=V?Ϸ???!???????0"g
;gradient_tape/model_2/conv2d_71/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??($=???!?N??????0"i
;gradient_tape/model_2/conv2d_77/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterg?[??!?Խ?x)??08I???$?KA@Q?)??ZP@Y?c5?25??a8?????X@q:?,V?a6@y|??e?zN?"?	
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	?ʅ???x@?ʅ???x@!?ʅ???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?ʅ???x@=?Е?b@1M3??$2n@A???????IʉvR???rEagerKernelExecute 0*	?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?٭e3@!\F????X@)?٭e3@1\F????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?a??????!???*}??)?a??????1???*}??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???4??!??jA?x??)??mU??1?????s??:Preprocessing2F
Iterator::Model???V???!V???>???){O崧?l?1k%?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??˙?3@!9??`??X@)?<?E~?`?1?}{nyL??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIˋ?E>mC@Q5tg???N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=?Е?b@=?Е?b@!=?Е?b@      ??!       "	M3??$2n@M3??$2n@!M3??$2n@*      ??!       2	??????????????!???????:	ʉvR???ʉvR???!ʉvR???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qˋ?E>mC@y5tg???N@?"k
=gradient_tape/model_29/conv2d_788/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????۲?!?????۲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?=?E????!&jk??ĺ?"k
=gradient_tape/model_29/conv2d_787/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterL3k4?ܛ?!|C?????08"]
<gradient_tape/model_29/max_pooling2d_116/MaxPool/MaxPoolGradMaxPoolGradE?Â??!廵????"8
model_29/conv2d_806/Conv2DConv2D?Ӝp?_??!cV?Q????0"8
model_29/conv2d_809/Conv2DConv2D???z8X??!??(aܶ??0"8
model_29/conv2d_800/Conv2DConv2D??["?o??!@	t?ӄ??0"i
=gradient_tape/model_29/conv2d_794/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter򶇁???! ??F??0"i
=gradient_tape/model_29/conv2d_800/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??&{????!)??r???0"k
=gradient_tape/model_29/conv2d_806/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ѷP????!Cl?@K??08I????,AA@Q??6?i_P@Y?c5?25??a8?????X@q9?	c?6@y??=?5oQ?"?	
both?Your program is POTENTIALLY input-bound because 38.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?22.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
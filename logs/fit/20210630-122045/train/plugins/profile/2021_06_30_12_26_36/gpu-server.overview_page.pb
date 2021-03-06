?	?M?Ūy@?M?Ūy@!?M?Ūy@	??D?=?~???D?=?~?!??D?=?~?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?M?Ūy@???B?d@1??r!Yn@A	????=??I??7(@Y_(`;???rEagerKernelExecute 0*	?l?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?4)?~?@!?y??T?X@)?4)?~?@1?y??T?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch*S?A?њ?!???qB??)*S?A?њ?1???qB??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismƋ?!r???!?P?Lu??)}(F??1??????:Preprocessing2F
Iterator::Model|*?=%???!??3ɑ??)d??1?n?1G???h??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapO;?5Y?@!?pf??X@)\???4_?1@ͺ}q?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??D?=?~?I??N??oD@Q0R?j7?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???B?d@???B?d@!???B?d@      ??!       "	??r!Yn@??r!Yn@!??r!Yn@*      ??!       2		????=??	????=??!	????=??:	??7(@??7(@!??7(@B      ??!       J	_(`;???_(`;???!_(`;???R      ??!       Z	_(`;???_(`;???!_(`;???b      ??!       JGPUY??D?=?~?b q??N??oD@y0R?j7?M@?"k
=gradient_tape/model_10/conv2d_275/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??A۲?!??A۲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?k?j??!??~?ﵺ?"k
=gradient_tape/model_10/conv2d_274/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?
??!?I?????08"\
;gradient_tape/model_10/max_pooling2d_40/MaxPool/MaxPoolGradMaxPoolGrad?9?0????!????????"8
model_10/conv2d_293/Conv2DConv2Dحׄ]v??!w?9V????0"8
model_10/conv2d_296/Conv2DConv2D???QU??!?z??E???0"8
model_10/conv2d_287/Conv2DConv2D,>????!=?
h???0"i
=gradient_tape/model_10/conv2d_281/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!?k}`??0"i
=gradient_tape/model_10/conv2d_287/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterUU¹?Ε?!^ '+??0"k
=gradient_tape/model_10/conv2d_293/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???Ar??!=i?-OT??08I???8EA@Q~,?c]P@Y?c5?25??a8?????X@q??e??2@y?i??N?"?	
both?Your program is POTENTIALLY input-bound because 40.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?18.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
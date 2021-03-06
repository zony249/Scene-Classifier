?	????o<y@????o<y@!????o<y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????o<y@s?}?p?c@1??<3n@A?????I̶?ֈ?@rEagerKernelExecute 0*	?|?5.?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???
?=@!T?K$??X@)???
?=@1T?K$??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch3?ۃ??!~)?8??)3?ۃ??1~)?8??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/?????!*iE??u??)??T????1֨?????:Preprocessing2F
Iterator::Model<?_?E??!j?t4???)??Z?p?1?1|19??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?lw??=@!?"??A?X@)m??)??b?1%?s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???p6D@Qd1???M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s?}?p?c@s?}?p?c@!s?}?p?c@      ??!       "	??<3n@??<3n@!??<3n@*      ??!       2	??????????!?????:	̶?ֈ?@̶?ֈ?@!̶?ֈ?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???p6D@yd1???M@?"k
=gradient_tape/model_28/conv2d_761/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterS5?<?ϲ?!S5?<?ϲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?gp?I???!7O??????"k
=gradient_tape/model_28/conv2d_760/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ܛ?!??????08"]
<gradient_tape/model_28/max_pooling2d_112/MaxPool/MaxPoolGradMaxPoolGrad??]????!???4??"8
model_28/conv2d_782/Conv2DConv2Dc??o?[??!?????0"8
model_28/conv2d_779/Conv2DConv2DH"
?U??!??I???0"8
model_28/conv2d_773/Conv2DConv2D???9So??!F?@+4???0"i
=gradient_tape/model_28/conv2d_767/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??;V~??!&H?#n??0"i
=gradient_tape/model_28/conv2d_773/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?x?55???!??bNe??0"k
=gradient_tape/model_28/conv2d_782/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=???ޗ??!2)?9?_??08IC?
{HA@Q^??x?[P@Y?c5?25??a8?????X@qܷ?P??2@y??)~N?"?	
both?Your program is POTENTIALLY input-bound because 39.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?18.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
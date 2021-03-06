?	u:????t@u:????t@!u:????t@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCu:????t@???$xAZ@1?Q*?	?k@AqN`:??I>"?D???rEagerKernelExecute 0*	??Q?E?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorZ??8?2@!?
?1%?X@)Z??8?2@1?
?1%?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchf?ʉv??!????[???)f?ʉv??1????[???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????g??!zD?o????)??_>Y??1???.??:Preprocessing2F
Iterator::Modelp
+TT??!??|??)??1 ?n?1?۪?[???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?_̖??2@!??????X@)8fٓ??\?1P?6{O??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 31.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI8?7g?0@@QddL??P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???$xAZ@???$xAZ@!???$xAZ@      ??!       "	?Q*?	?k@?Q*?	?k@!?Q*?	?k@*      ??!       2	qN`:??qN`:??!qN`:??:	>"?D???>"?D???!>"?D???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q8?7g?0@@yddL??P@?"j
<gradient_tape/model_6/conv2d_149/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??F?f???!??F?f???08"c
<cond_1/then/_10/cond_1/Adam/Adam/update_96/ResourceApplyAdamResourceApplyAdamb?_2??!?H$?$??"j
<gradient_tape/model_6/conv2d_148/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterf?\+)???!????d??08"[
:gradient_tape/model_6/max_pooling2d_24/MaxPool/MaxPoolGradMaxPoolGradqg????!	??(???"7
model_6/conv2d_167/Conv2DConv2D????g??!?)?j#???0"7
model_6/conv2d_161/Conv2DConv2Du??????!fX$c????0"h
<gradient_tape/model_6/conv2d_155/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?x?G????!?'l????0"h
<gradient_tape/model_6/conv2d_161/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterq{?Y???!z{	?K???0"j
<gradient_tape/model_6/conv2d_167/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterd?Q??r??!??,t_??08"f
;gradient_tape/model_6/conv2d_167/Conv2D/Conv2DBackpropInputConv2DBackpropInput?#??R??!!?𪝴??0IW??&zB@Q??uمO@YX?ܾ,??aÍM??X@q??????1@y?)pK?V?"?	
both?Your program is POTENTIALLY input-bound because 31.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?17.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
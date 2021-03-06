?	5???-v@5???-v@!5???-v@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC5???-v@?Q???`@1'g(?x?k@Azq??ř?Ii??T???rEagerKernelExecute 0*	G?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?? ?"7@!ŲA??X@)?? ?"7@1ŲA??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch{??v? ??!??=sF??){??v? ??1??=sF??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismͰQ?o&??!?1?׃???)<??kЗ??1Ҹ?r?~??:Preprocessing2F
Iterator::Model-Ӿ???!%?c?h???)V???4i?1????K.??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap/5B?S#7@!:εK5?X@)?"j??GY?1G<??B{?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??/u|?B@QOЊ?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q???`@?Q???`@!?Q???`@      ??!       "	'g(?x?k@'g(?x?k@!'g(?x?k@*      ??!       2	zq??ř?zq??ř?!zq??ř?:	i??T???i??T???!i??T???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??/u|?B@yOЊ?O@?"k
=gradient_tape/model_10/conv2d_245/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??}φD??!??}φD??08"c
<cond_1/then/_10/cond_1/Adam/Adam/update_96/ResourceApplyAdamResourceApplyAdam}?e?	??!2[apɼ?"k
=gradient_tape/model_10/conv2d_244/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?/????!???Ě#??08"\
;gradient_tape/model_10/max_pooling2d_40/MaxPool/MaxPoolGradMaxPoolGradX?[????!*??R???"8
model_10/conv2d_263/Conv2DConv2Du????R??!?
d????0"8
model_10/conv2d_257/Conv2DConv2D	H/??`??!??X????0"i
=gradient_tape/model_10/conv2d_251/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter; ????!??j????0"i
=gradient_tape/model_10/conv2d_257/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??quk???!?r?lk???0"k
=gradient_tape/model_10/conv2d_263/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????V??!6B???O??08"g
<gradient_tape/model_10/conv2d_263/Conv2D/Conv2DBackpropInputConv2DBackpropInput??7??9??!ֽ??v???0I?2`NЙB@QS͟?/fO@YX?ܾ,??aÍM??X@qQ????1@y???ǏN?"?	
both?Your program is POTENTIALLY input-bound because 37.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
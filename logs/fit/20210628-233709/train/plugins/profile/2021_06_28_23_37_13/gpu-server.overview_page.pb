?	?M?ҜFx@?M?ҜFx@!?M?ҜFx@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?M?ҜFx@(`;1?b@1f?"?OVm@A?T?????I.Ȗ??@rEagerKernelExecute 0*	+?	??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?????<@!?!$?1?X@)?????<@1?!$?1?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetcha?X5??!??Y????)a?X5??1??Y????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?qn?!F?^???)??5Φ#??1??)??:Preprocessing2F
Iterator::Model?h9?Cm??!'OD????)?p??[um?1H8X?.???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapk'JB?<@!?]????X@)D???XPX?1?V46u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??B??C@QX??Y6N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(`;1?b@(`;1?b@!(`;1?b@      ??!       "	f?"?OVm@f?"?OVm@!f?"?OVm@*      ??!       2	?T??????T?????!?T?????:	.Ȗ??@.Ȗ??@!.Ȗ??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??B??C@yX??Y6N@?"k
=gradient_tape/model_11/conv2d_269/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?'j('??!?'j('??08"c
<cond_1/then/_10/cond_1/Adam/Adam/update_96/ResourceApplyAdamResourceApplyAdam?yA?L??!?!?
?;?"k
=gradient_tape/model_11/conv2d_268/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery?V=R??!F??EU???08"\
;gradient_tape/model_11/max_pooling2d_44/MaxPool/MaxPoolGradMaxPoolGrad???~??!?"|?,+??"i
=gradient_tape/model_11/conv2d_275/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter/3	?75??!]I}?ӑ??0"i
=gradient_tape/model_11/conv2d_281/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterG??k?;??!e??M???0"8
model_11/conv2d_287/Conv2DConv2DM???S??!??d??a??0"8
model_11/conv2d_281/Conv2DConv2D?L?CSI??!???y???0"i
<gradient_tape/model_11/conv2d_275/Conv2D/Conv2DBackpropInputConv2DBackpropInput?5ʠ庖?!,?l'B??08"i
<gradient_tape/model_11/conv2d_269/Conv2D/Conv2DBackpropInputConv2DBackpropInputyX?4????!???Ϛ??08I KA???B@QാiCO@YX?ܾ,??aÍM??X@q1u?rs@y??
?ضL?"?
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
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 
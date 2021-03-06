?	?N>=?>y@?N>=?>y@!?N>=?>y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?N>=?>y@
p??c@1?)?:{n@A%????I1??PN???rEagerKernelExecute 0*	L7?A??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??o|A@!f?h??X@)??o|A@1f?h??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?	??$>??!??ID????)?	??$>??1??ID????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??3g}ʡ?!?&?.h??)=?U?????1?b?????:Preprocessing2F
Iterator::Model???vi??!??o?̸??)?=?N??i?1?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapk???|A@!????X@)Q?B?y?_?1?{?v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI+D?z}?C@Qջ}??/N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
p??c@
p??c@!
p??c@      ??!       "	?)?:{n@?)?:{n@!?)?:{n@*      ??!       2	%????%????!%????:	1??PN???1??PN???!1??PN???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q+D?z}?C@yջ}??/N@?"i
;gradient_tape/model_1/conv2d_32/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltere??????!e??????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamU??-9??!:u^?????"i
;gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~`?????!??lȢ???08"Z
9gradient_tape/model_1/max_pooling2d_4/MaxPool/MaxPoolGradMaxPoolGrad?J?????!D'??????"6
model_1/conv2d_53/Conv2DConv2D:Tbde??!?qb?????0"6
model_1/conv2d_50/Conv2DConv2D=?????!3JB	#???0"6
model_1/conv2d_44/Conv2DConv2DEoVz?9??!??W[??0"g
;gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?=#????!V???(??0"g
;gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterʋ B????!hf?????0"8
model_1/conv2d_38/Conv2DConv2D??P{???!|??w?4??08Ic??'A@Q?3? ?qP@Y?c5?25??a8?????X@q?}I??s*@y3L?,?O?"?	
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
Refer to the TF2 Profiler FAQb?13.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
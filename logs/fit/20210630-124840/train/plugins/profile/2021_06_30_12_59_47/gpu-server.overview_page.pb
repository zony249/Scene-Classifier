?	??C???x@??C???x@!??C???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??C???x@??};?0c@1|?y?Bn@A>w??׹??It{Ic???rEagerKernelExecute 0*	F???d??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?7???j6@!Ԡ?X@)?7???j6@1Ԡ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?2d????!O1H??ɵ?)?2d????1O1H??ɵ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism1?q?	۟?!ͳ?|????)Jy???1?l.$`\??:Preprocessing2F
Iterator::Model???????!??☻??)?q?j??l?1 ?bv???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?;?(Ak6@!???3"?X@)??A?]?10rR?+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI+?b?|?C@Q?<? ?wN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??};?0c@??};?0c@!??};?0c@      ??!       "	|?y?Bn@|?y?Bn@!|?y?Bn@*      ??!       2	>w??׹??>w??׹??!>w??׹??:	t{Ic???t{Ic???!t{Ic???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q+?b?|?C@y?<? ?wN@?"l
>gradient_tape/model_48/conv2d_1301/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter8??"????!8??"????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam ???qF??! ??&???"l
>gradient_tape/model_48/conv2d_1300/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~?s?/??!0?J????08"]
<gradient_tape/model_48/max_pooling2d_192/MaxPool/MaxPoolGradMaxPoolGrad/?.>??!M?A???"9
model_48/conv2d_1319/Conv2DConv2D?h???ϗ?!):ކ2???0"9
model_48/conv2d_1322/Conv2DConv2D??#F7y??!^??oY???0"9
model_48/conv2d_1313/Conv2DConv2D?q?Ŗ?!?Q???0"j
>gradient_tape/model_48/conv2d_1307/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.U?(???!%?%??h??0"j
>gradient_tape/model_48/conv2d_1313/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.???????!?B????0"l
>gradient_tape/model_48/conv2d_1319/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?n3????!?}ty?[??08Ij?)h2A@Q?0???fP@Y?c5?25??a8?????X@q?Ey??5@y?q;???P?"?	
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?21.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
?	????qy@????qy@!????qy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????qy@L7?A`fd@1T?Q??4n@Afh<?y??I????~?@rEagerKernelExecute 0*	???  ??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generators۾G?WA@!??c??X@)s۾G?WA@1??c??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ׂ?C??!?q9Lk??)?ׂ?C??1?q9Lk??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Đ?L??!z/??M`??)?d??1A???U??:Preprocessing2F
Iterator::ModelB?%U?M??!?I??z??)?fh<q?1=??҈?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?e6XA@!?-?[!?X@)?????\?1v*gQޏt?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI%???QD@Q?L?f ?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L7?A`fd@L7?A`fd@!L7?A`fd@      ??!       "	T?Q??4n@T?Q??4n@!T?Q??4n@*      ??!       2	fh<?y??fh<?y??!fh<?y??:	????~?@????~?@!????~?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q%???QD@y?L?f ?M@?"k
=gradient_tape/model_19/conv2d_518/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??L??˲?!??L??˲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam`?`??>??!G??4}???"k
=gradient_tape/model_19/conv2d_517/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?y???7??!]?Lڰ???08"\
;gradient_tape/model_19/max_pooling2d_76/MaxPool/MaxPoolGradMaxPoolGrad???X"8??!??e%????"8
model_19/conv2d_536/Conv2DConv2D3?+?ȗ?!X\??????0"8
model_19/conv2d_539/Conv2DConv2D?	?fn??!?|?ŕ???0"8
model_19/conv2d_530/Conv2DConv2D???(???!i-?՚???0"i
=gradient_tape/model_19/conv2d_524/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter-???????!O???6`??0"i
=gradient_tape/model_19/conv2d_530/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???D???!?????0"k
=gradient_tape/model_19/conv2d_536/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterT&*;???!I??k#W??08IP1?GS7A@QXg%\VdP@Y?c5?25??a8?????X@q????j@y?{6??^O?"?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
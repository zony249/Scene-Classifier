?	u?Bُy@u?Bُy@!u?Bُy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCu?Bُy@*s????d@1j?~?^;n@A??E}?;??I?B]?@rEagerKernelExecute 0*	?&1??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?[Ɏ??@!?N???X@)?[Ɏ??@1?N???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??L0?k??!?恻p:??)??L0?k??1?恻p:??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!?<??a??)? ??=@??1????N??:Preprocessing2F
Iterator::Model??;???!p~?|??)V??Dׅo?1;??҈?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapj??%??@!???? ?X@)??̔??b?1??T!?}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?t?[?nD@Q??=?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*s????d@*s????d@!*s????d@      ??!       "	j?~?^;n@j?~?^;n@!j?~?^;n@*      ??!       2	??E}?;????E}?;??!??E}?;??:	?B]?@?B]?@!?B]?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?t?[?nD@y??=?M@?"k
=gradient_tape/model_15/conv2d_410/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[?-+????![?-+????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam_ڟ?P@??!??O????"k
=gradient_tape/model_15/conv2d_409/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???B???!?N?OͶ??08"\
;gradient_tape/model_15/max_pooling2d_60/MaxPool/MaxPoolGradMaxPoolGrad?h5?/??!???HS???"8
model_15/conv2d_428/Conv2DConv2Dseb\???!C?<?>???0"8
model_15/conv2d_431/Conv2DConv2D?/͞Rj??!@P)????0"8
model_15/conv2d_422/Conv2DConv2D=?`?L???!H`"?Rw??0"i
=gradient_tape/model_15/conv2d_416/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???E???!?VأJ9??0"i
=gradient_tape/model_15/conv2d_422/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???
??!j}mF???0"k
=gradient_tape/model_15/conv2d_428/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?x1%t???!?Я?G??08I?ĺ}?;A@Q??"A8bP@Y?c5?25??a8?????X@q?G?*?93@yF"?:?N?"?	
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
Refer to the TF2 Profiler FAQb?19.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
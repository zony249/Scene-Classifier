?	?Hh?y?y@?Hh?y?y@!?Hh?y?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Hh?y?y@??x"?d@1$?w~шn@AS??Y??IW@??? @rEagerKernelExecute 0*	C`?Кl?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?gA(?@@!8?l?a?X@)?gA(?@@18?l?a?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??(ϼ??!????-??)??(ϼ??1????-??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?V
?\???!?h><??).??e?O??1????:Preprocessing2F
Iterator::Model?^?D???!,{g[??)?? @??m?1r8?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??ӝ'@@!!&?;??X@)u?)?:\?1?>???u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?2??fD@Q??N?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x"?d@??x"?d@!??x"?d@      ??!       "	$?w~шn@$?w~шn@!$?w~шn@*      ??!       2	S??Y??S??Y??!S??Y??:	W@??? @W@??? @!W@??? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?2??fD@y??N?M@?"k
=gradient_tape/model_14/conv2d_383/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterd??vԲ??!d??vԲ??08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?G????!P???Ly??"k
=gradient_tape/model_14/conv2d_382/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteri	-ut???!Uw?????08"\
;gradient_tape/model_14/max_pooling2d_56/MaxPool/MaxPoolGradMaxPoolGrad?c?????!5???????"8
model_14/conv2d_404/Conv2DConv2D?????!??Yuȅ??0"8
model_14/conv2d_401/Conv2DConv2D ?????!?0=?~e??0"8
model_14/conv2d_395/Conv2DConv2D*:??1??!6X]F?+??0"i
=gradient_tape/model_14/conv2d_389/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?<?8ڕ?!??t?????0"i
=gradient_tape/model_14/conv2d_395/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0Zg?????!???????0":
model_14/conv2d_389/Conv2DConv2D?_Y????!m?v????08Im?z|1A@QʀB?AgP@Y?c5?25??a8?????X@q??F	2@y?<?݀?O?"?	
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
Refer to the TF2 Profiler FAQb?18.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
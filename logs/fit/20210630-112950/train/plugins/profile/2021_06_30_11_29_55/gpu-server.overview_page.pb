?	8?k???x@8?k???x@!8?k???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC8?k???x@?q4GV?b@1?L1ATn@Aɓ?k&ߜ?Iv?|?H?@rEagerKernelExecute 0*	?rh?U>?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorR?Q???@!X?~?X@)R?Q???@1X?~?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?	Q???!"D??*??)?	Q???1"D??*??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?f?ba???!!0T?K??)C?Գ ???1!Y???l??:Preprocessing2F
Iterator::ModelA?+????!jP???)cAJh?1??"????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapl#??f??@!???9??X@)ҏ?S??[?1?{#v??u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??ȝ?kC@QN7b ?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q4GV?b@?q4GV?b@!?q4GV?b@      ??!       "	?L1ATn@?L1ATn@!?L1ATn@*      ??!       2	ɓ?k&ߜ?ɓ?k&ߜ?!ɓ?k&ߜ?:	v?|?H?@v?|?H?@!v?|?H?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??ȝ?kC@yN7b ?N@?"i
;gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??>?̲?!??>?̲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamh?<?c??!??%⥺?"i
;gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterk,?)???!???????08"[
:gradient_tape/model_3/max_pooling2d_12/MaxPool/MaxPoolGradMaxPoolGrad???j;5??!+?0Y????"7
model_3/conv2d_104/Conv2DConv2D??u?2??!??????0"7
model_3/conv2d_107/Conv2DConv2D#???+??!?c????0"6
model_3/conv2d_98/Conv2DConv2Dsz2????!?d?v?Z??0"g
;gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6D?????!???ȥ??0"g
;gradient_tape/model_3/conv2d_98/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterR3S0\???!?(~?????0"j
<gradient_tape/model_3/conv2d_107/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterJ$b??!?_?,??08I????/A@Q??y?%hP@Y?c5?25??a8?????X@q?㐨M?@y??u??O?"?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
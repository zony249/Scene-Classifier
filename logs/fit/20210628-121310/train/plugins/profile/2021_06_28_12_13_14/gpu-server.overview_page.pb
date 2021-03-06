?	^,?S?v@^,?S?v@!^,?S?v@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC^,?S?v@??:q?Ga@1=?)۴k@Amo?$???I??mē???rEagerKernelExecute 0*	??????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?+???A@!??????X@)?+???A@1??????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchA???FX??!?m??pŭ?)A???FX??1?m??pŭ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???B??!3???hB??)?}q?J[??1w?` a???:Preprocessing2F
Iterator::Modelu?BY???!b?O t»?)?n?;2Vk?1sI"X ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapQ0c
?A@!??b?X@)T8?T?]?1J???u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIC?? S`C@Q?g߬?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??:q?Ga@??:q?Ga@!??:q?Ga@      ??!       "	=?)۴k@=?)۴k@!=?)۴k@*      ??!       2	mo?$???mo?$???!mo?$???:	??mē?????mē???!??mē???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qC?? S`C@y?g߬?N@?"j
<gradient_tape/model_8/conv2d_197/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?d?Ɗq??!?d?Ɗq??08"c
<cond_1/then/_10/cond_1/Adam/Adam/update_96/ResourceApplyAdamResourceApplyAdamnDP???!?l?????"j
<gradient_tape/model_8/conv2d_196/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#?i:???!?Z??HN??08"[
:gradient_tape/model_8/max_pooling2d_32/MaxPool/MaxPoolGradMaxPoolGrad.?H???!??Tű???"7
model_8/conv2d_215/Conv2DConv2D?z?%a??!k?cv????0"7
model_8/conv2d_209/Conv2DConv2D??????!W?V????0"h
<gradient_tape/model_8/conv2d_209/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Wv????!".?????0"h
<gradient_tape/model_8/conv2d_203/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??-B?ŗ?!??9?(???0"j
<gradient_tape/model_8/conv2d_215/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr??c??!??bhW??08"f
;gradient_tape/model_8/conv2d_215/Conv2D/Conv2DBackpropInputConv2DBackpropInput?g ?F??!]?Mʫ??0I2?&?xB@Q???=?O@YX?ܾ,??aÍM??X@q?8?I'@y)?m??V?"?	
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
Refer to the TF2 Profiler FAQb?11.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
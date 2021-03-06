?	)?k{??v@)?k{??v@!)?k{??v@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC)?k{??v@??}?ua@1???cZ?k@A<??fԜ?I9
? @rEagerKernelExecute 0*	?I2?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorѱ?J\?4@!????X@)ѱ?J\?4@1????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?g?????!?`??%`??)?g?????1?`??%`??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???o????!??|????)?<?E~???1|l???:Preprocessing2F
Iterator::Model?,????!?	?c???)ҏ?S??k?1Cq"jtݐ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???ɣ4@!{N?w?X@)\;Qi[?1Ƹ?	f???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI? ?D?:C@Q`?f?@?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??}?ua@??}?ua@!??}?ua@      ??!       "	???cZ?k@???cZ?k@!???cZ?k@*      ??!       2	<??fԜ?<??fԜ?!<??fԜ?:	9
? @9
? @!9
? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ?D?:C@y`?f?@?N@?"j
<gradient_tape/model_9/conv2d_221/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???;????!???;????08"c
<cond_1/then/_10/cond_1/Adam/Adam/update_96/ResourceApplyAdamResourceApplyAdam??Pִ.??!????,??"j
<gradient_tape/model_9/conv2d_220/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?s?????!f?'*?k??08"[
:gradient_tape/model_9/max_pooling2d_36/MaxPool/MaxPoolGradMaxPoolGradE???tϙ?!T{ǻ???"7
model_9/conv2d_239/Conv2DConv2DаW*????!)J?,????0"7
model_9/conv2d_233/Conv2DConv2D?S2?ɘ?!"?S???0"h
<gradient_tape/model_9/conv2d_233/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!?"Q????0"h
<gradient_tape/model_9/conv2d_227/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterF???????!~S=????0"j
<gradient_tape/model_9/conv2d_239/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW?ƶl}??!c???f]??08"f
;gradient_tape/model_9/conv2d_239/Conv2D/Conv2DBackpropInputConv2DBackpropInput0?)؆f??!6[,@ϳ??0IE????oB@Q?]|?O@YX?ܾ,??aÍM??X@q?S?P??2@y??y?T?V?"?	
both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?18.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
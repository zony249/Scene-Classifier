?	Ǟ=???q@Ǟ=???q@!Ǟ=???q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCǞ=???q@0??9:X@1??C6?f@A????ģ?I?熦?4??rEagerKernelExecute 0*	?S??s??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???D@@!O9L??X@)???D@@1O9L??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?wD???!????Z??)?wD???1????Z??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?t"?T3??!?????x??)?????1???{w<??:Preprocessing2F
Iterator::Model(G?`Ƥ?!?6Po????)?(??0i?1???TU??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapP??@D@@!?+??X@)?E??U?1q????p?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI*":???A@Q??b`&:P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0??9:X@0??9:X@!0??9:X@      ??!       "	??C6?f@??C6?f@!??C6?f@*      ??!       2	????ģ?????ģ?!????ģ?:	?熦?4???熦?4??!?熦?4??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q*":???A@y??b`&:P@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdamX?\h8??!X?\h8??"g
;gradient_tape/model_2/conv2d_59/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?%q??O??!??D??0"e
:gradient_tape/model_2/conv2d_59/Conv2D/Conv2DBackpropInputConv2DBackpropInputR??pǴ??!X??s????0"8
model_2/conv2d_59/Conv2DConv2D?yʑ???!??-?x???08"e
:gradient_tape/model_2/conv2d_65/Conv2D/Conv2DBackpropInputConv2DBackpropInput??	??!<?䍬???0"6
model_2/conv2d_65/Conv2DConv2DS?"????!?-	p???0"0
Adam/gradients/AddN_31AddNɡhu???!?f?????"g
;gradient_tape/model_2/conv2d_65/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterG?㰑??!YU?h
v??0"G
.gradient_tape/dense_6/kernel/Regularizer/Mul_1Mul?2??$??!???cL???"i
;gradient_tape/model_2/conv2d_58/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter{??+ԗ?!6?m?u??08I?0??I?B@Q?{^?O@Y??A?????aL?:,??X@qӖ????@y?9?`
?R?"?
both?Your program is POTENTIALLY input-bound because 34.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
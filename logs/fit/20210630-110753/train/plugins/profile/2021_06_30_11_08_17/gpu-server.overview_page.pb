?	??Mo@??Mo@!??Mo@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??Mo@??????@1-=??Ikn@AW??????IN?E????rEagerKernelExecute 0*	?C?l???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorA???F:A@!O?&???X@)A???F:A@1O?&???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?IӠh??!jZ6?y]??)?IӠh??1jZ6?y]??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism㪲?????!???ט??)H¾?D???1?lqܷצ?:Preprocessing2F
Iterator::Model?Χ?U??!7?q????)?ҥI*s?1,t???ǋ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?<֌:A@!??#[?X@)G???R{a?1E?N@?Wy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI Ia?b=??Q?zBt
?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????@??????@!??????@      ??!       "	-=??Ikn@-=??Ikn@!-=??Ikn@*      ??!       2	W??????W??????!W??????:	N?E????N?E????!N?E????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q Ia?b=??y?zBt
?X@?"f
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterנn?Ҳ?!נn?Ҳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdami?V&.??!4?6?????"f
8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterT	G???!D'??????08"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad|?<HQ??!?Ë????"4
model/conv2d_23/Conv2DConv2D?X?rKЗ?!???v????0"4
model/conv2d_26/Conv2DConv2DU???x??!???	???0"4
model/conv2d_17/Conv2DConv2D???S???!p?+????0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??b<???!0R?R?[??0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?	 ????!0:Q	q??0"g
9gradient_tape/model/conv2d_23/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ٴ??!?8???X??08I? ??0A@Q??0?gP@Y?c5?25??a8?????X@q??? T??y??}ף&O?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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
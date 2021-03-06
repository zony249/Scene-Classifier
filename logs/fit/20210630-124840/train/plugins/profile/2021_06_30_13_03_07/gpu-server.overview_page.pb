?	?R	?&y@?R	?&y@!?R	?&y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?R	?&y@E???c@18ݲC?1n@A?(?	0??IS>U???rEagerKernelExecute 0*	?x?&?%?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???	?;@!Rv?ô?X@)???	?;@1Rv?ô?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?5&?\??!\?#񛃰?)?5&?\??1\?#񛃰?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^?SH??!xoт?;??)I??Z?և?18\[#qp??:Preprocessing2F
Iterator::Modelm??oB??!Q?]e??)˻??p?1?V}׆|??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap8h???;@!???&=?X@)??UJ??b?1n?,CY??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIMص?N?C@Q?'JS?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E???c@E???c@!E???c@      ??!       "	8ݲC?1n@8ݲC?1n@!8ݲC?1n@*      ??!       2	?(?	0???(?	0??!?(?	0??:	S>U???S>U???!S>U???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qMص?N?C@y?'JS?N@?"l
>gradient_tape/model_54/conv2d_1463/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]??ٲ?!]??ٲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?|^D?g??!T? i????"l
>gradient_tape/model_54/conv2d_1462/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter֬z0X???!?ӏ?????08"9
model_54/conv2d_1481/Conv2DConv2D??>?s??!hf???0"9
model_54/conv2d_1484/Conv2DConv2D&???Zk??!?B]Ѵ??0"]
<gradient_tape/model_54/max_pooling2d_216/MaxPool/MaxPoolGradMaxPoolGrad????#??!Y???H???"9
model_54/conv2d_1475/Conv2DConv2D;=?pp??! ?LLg??0"j
>gradient_tape/model_54/conv2d_1469/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter8??x??!??e;*??0"j
>gradient_tape/model_54/conv2d_1475/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?](p??!??t?4???0"l
>gradient_tape/model_54/conv2d_1481/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?s"E???! ?f????08I,???@0A@Q봀??gP@Y?c5?25??a8?????X@q?4c>?63@y?5u?R?P?"?	
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	?a?îy@?a?îy@!?a?îy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?a?îy@g)Y??d@1?*?3Tn@A??y??w??I??P?l??rEagerKernelExecute 0*	F???4??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator){K9_?=@!BK???X@)){K9_?=@1BK???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Ր?ǒ?!?r??P???)??Ր?ǒ?1?r??P???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?V	?3??!?Gݯ?]??)bg
?׈?1 ????:Preprocessing2F
Iterator::ModelM??u??!L???耽?)?J?E?m?1X??X???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapUܸ???=@!B?ş?X@)X?%???c?1F?۞@???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIgXP?MzD@Q???9??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	g)Y??d@g)Y??d@!g)Y??d@      ??!       "	?*?3Tn@?*?3Tn@!?*?3Tn@*      ??!       2	??y??w????y??w??!??y??w??:	??P?l????P?l??!??P?l??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qgXP?MzD@y???9??M@?"l
>gradient_tape/model_49/conv2d_1328/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?9'?Ѳ?!?9'?Ѳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam??om[??!:?????"l
>gradient_tape/model_49/conv2d_1327/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter:~T??Л?!?	5????08"]
<gradient_tape/model_49/max_pooling2d_196/MaxPool/MaxPoolGradMaxPoolGrad????v???!u?s????"9
model_49/conv2d_1346/Conv2DConv2D??e??}??!t???????0"9
model_49/conv2d_1349/Conv2DConv2D???qx??!?O??????0"9
model_49/conv2d_1340/Conv2DConv2DX??fl??!??ЀS???0"j
>gradient_tape/model_49/conv2d_1334/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???[?&??!??L?&\??0"j
>gradient_tape/model_49/conv2d_1340/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??6=l???!???9
??0"l
>gradient_tape/model_49/conv2d_1346/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>E??)???!QW??V??08I?????A@Q{:`P@Y?c5?25??a8?????X@qV?Zm?@y(h\???N?"?
both?Your program is POTENTIALLY input-bound because 40.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
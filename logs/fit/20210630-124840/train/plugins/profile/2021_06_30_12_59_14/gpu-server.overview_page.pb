?	?2?}1y@?2?}1y@!?2?}1y@	???^??????^???!???^???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?2?}1y@?m3??c@1??8<<n@A??mT???I?q7?????Y,?S???rEagerKernelExecute 0*	'1???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??L0??=@!-7??X@)??L0??=@1-7??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch	?????!?C?ȏ???)	?????1?C?ȏ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?}?Az???!#?w^????)*?Z^?ކ?1??K??7??:Preprocessing2F
Iterator::Model??9??q??!?S?????) ?8?@dq?1w??^z:??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap+5{??=@!:k?B?X@)6??\^?1?7ނy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???^???I??B?C@QL?x?dN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?m3??c@?m3??c@!?m3??c@      ??!       "	??8<<n@??8<<n@!??8<<n@*      ??!       2	??mT?????mT???!??mT???:	?q7??????q7?????!?q7?????B      ??!       J	,?S???,?S???!,?S???R      ??!       Z	,?S???,?S???!,?S???b      ??!       JGPUY???^???b q??B?C@yL?x?dN@?"l
>gradient_tape/model_47/conv2d_1274/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?TZ????!?TZ????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?5OT?j??!4"nay???"l
>gradient_tape/model_47/conv2d_1273/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?n.?ћ?!?~yn???08"]
<gradient_tape/model_47/max_pooling2d_188/MaxPool/MaxPoolGradMaxPoolGrad??̌w??!????_???"9
model_47/conv2d_1292/Conv2DConv2D?x{??!r?*ҿ???0"9
model_47/conv2d_1295/Conv2DConv2D??Y??x??!(?u?????0"9
model_47/conv2d_1286/Conv2DConv2Df?WJe??!??j?????0"j
>gradient_tape/model_47/conv2d_1280/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\z??:$??!4?`V	W??0"j
>gradient_tape/model_47/conv2d_1286/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???'????!Je?s
??0"l
>gradient_tape/model_47/conv2d_1295/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterF??%???!?6?{T??08I1??q>A@Qh?G?`P@Y?c5?25??a8?????X@q_??2@y?????SO?"?	
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
Refer to the TF2 Profiler FAQb?18.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
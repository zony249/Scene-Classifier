?	?ܘ??py@?ܘ??py@!?ܘ??py@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?ܘ??py@???z?Od@1*?J=On@A??J"? ??I?
Y?J @rEagerKernelExecute 0*	B?l????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator'K??}=@!C?]?X@)'K??}=@1C?]?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ΤM?=??!M?X?????)?ΤM?=??1M?X?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?X?????!?? ?]??)??an??1???3@ף?:Preprocessing2F
Iterator::Model?}s???!:lT????)(??ȯo?1?y\??Ԋ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?s???}=@!??????X@)??E?>a?1?nN%4}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI4S5;67D@Q̬????M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???z?Od@???z?Od@!???z?Od@      ??!       "	*?J=On@*?J=On@!*?J=On@*      ??!       2	??J"? ????J"? ??!??J"? ??:	?
Y?J @?
Y?J @!?
Y?J @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q4S5;67D@y̬????M@?"l
>gradient_tape/model_46/conv2d_1247/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?΍?(???!?΍?(???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdaml???G??!???ͺ?"l
>gradient_tape/model_46/conv2d_1246/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!&d??!?}ƅC???08"]
<gradient_tape/model_46/max_pooling2d_184/MaxPool/MaxPoolGradMaxPoolGrad???????!? ?_???"9
model_46/conv2d_1265/Conv2DConv2D??f
???!B??5!???0"9
model_46/conv2d_1268/Conv2DConv2D^C??k??!???????0"9
model_46/conv2d_1259/Conv2DConv2D?'??????!?R?'W???0"j
>gradient_tape/model_46/conv2d_1253/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???*???!??$M?Z??0"j
>gradient_tape/model_46/conv2d_1259/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?[$A???!???N??0"l
>gradient_tape/model_46/conv2d_1268/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(??,???!????Y??08I?n?W/A@Qv??"ThP@Y?c5?25??a8?????X@qH#?m?@yy8G??N?"?
both?Your program is POTENTIALLY input-bound because 39.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	 
f?x@ 
f?x@! 
f?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC 
f?x@??q?c@1?B?l3n@A???g%???I???????rEagerKernelExecute 0*	???M??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??X??3@!?@????X@)??X??3@1?@????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchF$
-????!?F? ????)F$
-????1?F? ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!/}??,??)M?^?iN??1??k7:[??:Preprocessing2F
Iterator::Model?PN?????!n?V????)?`??o?1???ԵW??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap{??v??3@!ɹT?$?X@)CY??Z?Z?1<?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI9OH8?~C@Qǰ??u?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??q?c@??q?c@!??q?c@      ??!       "	?B?l3n@?B?l3n@!?B?l3n@*      ??!       2	???g%??????g%???!???g%???:	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q9OH8?~C@yǰ??u?N@?"l
>gradient_tape/model_41/conv2d_1112/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter޴ײ?!޴ײ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam+?5???!?h??????"l
>gradient_tape/model_41/conv2d_1111/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?̿?0???!?(?z???08"]
<gradient_tape/model_41/max_pooling2d_164/MaxPool/MaxPoolGradMaxPoolGrad?x?????!???r???"9
model_41/conv2d_1133/Conv2DConv2D#?U?I??!`?mu????0"9
model_41/conv2d_1130/Conv2DConv2D9???LA??!g?Կ??0"9
model_41/conv2d_1124/Conv2DConv2D!f?p??!????????0"j
>gradient_tape/model_41/conv2d_1118/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?A4>??!I?P??P??0"j
>gradient_tape/model_41/conv2d_1124/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????ߕ?!?i??T??0"l
>gradient_tape/model_41/conv2d_1133/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ø?g???!???Z?N??08I7?? ?BA@Q???^P@Y?c5?25??a8?????X@q??D?*?@y???9 P?"?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	Ϡ???@Ϡ???@!Ϡ???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCϠ???@?Q?=/t@1?T[?;n@A??ޫV&??I?J????rEagerKernelExecute 0*	?E???2?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?O???>@!S?ƴ??X@)?O???>@1S?ƴ??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch~p>u???!?D??_1??)~p>u???1?D??_1??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?M?t"??!??L?R???)T:X??0??1????E7??:Preprocessing2F
Iterator::Model9??!??!,?F????)??~?Ϛo?12?0&䌉?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?1???>@!V{.?F?X@)W??x??Y?1? ?ٳt?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?a??6?L@Q?G?UE@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q?=/t@?Q?=/t@!?Q?=/t@      ??!       "	?T[?;n@?T[?;n@!?T[?;n@*      ??!       2	??ޫV&????ޫV&??!??ޫV&??:	?J?????J????!?J????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?a??6?L@y?G?UE@?"l
>gradient_tape/model_60/conv2d_1625/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????ɲ?!?????ɲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?$?]????!0t?????"l
>gradient_tape/model_60/conv2d_1624/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?s???!9Z??m???08"]
<gradient_tape/model_60/max_pooling2d_240/MaxPool/MaxPoolGradMaxPoolGrad=hb????!A? ?????"9
model_60/conv2d_1643/Conv2DConv2D?@????!S?????0"9
model_60/conv2d_1646/Conv2DConv2D??kvw??!3;?????0"9
model_60/conv2d_1637/Conv2DConv2D<?l???!N??c????0"j
>gradient_tape/model_60/conv2d_1631/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??DI???!J?L"R??0"j
>gradient_tape/model_60/conv2d_1637/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?l]ծ???!???L??0"l
>gradient_tape/model_60/conv2d_1643/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?.
V???!??et?P??08I u1;A@Q?pvEgbP@Y?c5?25??a8?????X@q???l4@y?O~?ЊP?"?
both?Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
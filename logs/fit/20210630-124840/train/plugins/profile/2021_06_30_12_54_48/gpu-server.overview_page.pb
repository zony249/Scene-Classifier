?	???~/y@???~/y@!???~/y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???~/y@?]??c@1F?x?/n@AS??.???IZ????#@rEagerKernelExecute 0*	D?l????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorB#ظ?I=@!WL&ki?X@)B#ظ?I=@1WL&ki?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?<???!??w? ??)?<???1??w? ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismqZ??? ??!MX??1ո?)?;l"3??1???׉??:Preprocessing2F
Iterator::Model??8~??!??Q?N??)????n?1(T???P??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap:??*?J=@!??Z,??X@)<?.9?d?1??&؁?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noII????D@Q?[mj?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?]??c@?]??c@!?]??c@      ??!       "	F?x?/n@F?x?/n@!F?x?/n@*      ??!       2	S??.???S??.???!S??.???:	Z????#@Z????#@!Z????#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qI????D@y?[mj?M@?"l
>gradient_tape/model_39/conv2d_1058/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?O?????!?O?????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamtث]?R??!*ƺ׎ĺ?"l
>gradient_tape/model_39/conv2d_1057/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter*?j?$???!Z??
????08"]
<gradient_tape/model_39/max_pooling2d_156/MaxPool/MaxPoolGradMaxPoolGrad????S???!R2?????"9
model_39/conv2d_1076/Conv2DConv2Dz??uo??!a05:????0"9
model_39/conv2d_1079/Conv2DConv2D???mZ??!?O?????0"9
model_39/conv2d_1070/Conv2DConv2D?y?Fy??!5??}??0"j
>gradient_tape/model_39/conv2d_1064/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ېI???!zP??@??0"j
>gradient_tape/model_39/conv2d_1070/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?6y)???!?;??????0"l
>gradient_tape/model_39/conv2d_1079/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??cG???! J??{G??08I]1?Y1A@Q|Q?SgP@Y?c5?25??a8?????X@qbJ?V?@yN?KyYP?"?
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
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 
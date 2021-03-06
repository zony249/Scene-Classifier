?	?n?o?^g@?n?o?^g@!?n?o?^g@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?n?o?^g@?'c|?]@1??Q???f@A??j?=&??I#e???(??rEagerKernelExecute 0*	??"????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???Ft?2@!???X@)???Ft?2@1???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch#M?<i??!@Ъ????)#M?<i??1@Ъ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?8
??!???jW'??)?H?F?q??1?@?**b??:Preprocessing2F
Iterator::Model?p????!LS,m????)k?ѯ?o?1rM??ڔ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap3NCT??2@!?iɨ?X@)?'eRC[?17d_?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@?Ñ???Q?K???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'c|?]@?'c|?]@!?'c|?]@      ??!       "	??Q???f@??Q???f@!??Q???f@*      ??!       2	??j?=&????j?=&??!??j?=&??:	#e???(??#e???(??!#e???(??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@?Ñ???y?K???X@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdam??8??I??!??8??I??"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????aH??!?UI??0"6
model/conv2d_11/Conv2DConv2D??????!fۇ????08"c
8gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput ???ǝ??!fׯ????0"c
8gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropInputConv2DBackpropInput`?6?d,??!????&???0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?mFE???!Tr?M????0"4
model/conv2d_17/Conv2DConv2D???Go???!$??6???0"0
Adam/gradients/AddN_31AddN9he?2z??!??Sʱw??"E
,gradient_tape/dense/kernel/Regularizer/Mul_1Mul?ӵ0???!?,?????"g
9gradient_tape/model/conv2d_10/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6?v??×?!c?65.s??08Il?????B@Q?"O@Y??A?????aL?:,??X@q-0????yCM?X}hR?"?
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
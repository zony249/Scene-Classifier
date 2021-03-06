?	????(Fy@????(Fy@!????(Fy@	?-^??????-^?????!?-^?????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL????(Fy@?v?>X?c@1:毐In@A}\*????I<1??P?@YJ?????rEagerKernelExecute 0*	@5^?Q?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????ɶ=@!m?6j?X@)????ɶ=@1m?6j?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?@?"??!??P??j??)?@?"??1??P??j??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismuYLl>???!?"??;???)?1??8*??1W?&?w??:Preprocessing2F
Iterator::Modelu???l??!9!??۔??)???mRq?1??GJ ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap*???O?=@!7B???X@)???$??`?1z??N?$|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?-^?????IjE?d?D@Q?D??}?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v?>X?c@?v?>X?c@!?v?>X?c@      ??!       "	:毐In@:毐In@!:毐In@*      ??!       2	}\*????}\*????!}\*????:	<1??P?@<1??P?@!<1??P?@B      ??!       J	J?????J?????!J?????R      ??!       Z	J?????J?????!J?????b      ??!       JGPUY?-^?????b qjE?d?D@y?D??}?M@?"l
>gradient_tape/model_43/conv2d_1166/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltertu2A ???!tu2A ???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?k?]S??!6_?׎??"l
>gradient_tape/model_43/conv2d_1165/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???{3???!?_?O????08"]
<gradient_tape/model_43/max_pooling2d_172/MaxPool/MaxPoolGradMaxPoolGrad??6?_??!ƚ?????"9
model_43/conv2d_1184/Conv2DConv2Dti2ۗ???!??"????0"9
model_43/conv2d_1187/Conv2DConv2D?%D?[??!??z[???0"9
model_43/conv2d_1178/Conv2DConv2D?W!????!?7?JT???0"j
>gradient_tape/model_43/conv2d_1172/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh? ???!攍
?Q??0"j
>gradient_tape/model_43/conv2d_1178/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'?HW???!&?????0"l
>gradient_tape/model_43/conv2d_1187/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?p0%???!?5?L?R??08I??~;A@Qv??tbP@Y?c5?25??a8?????X@q?˺+2@yv?????O?"?	
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
Refer to the TF2 Profiler FAQb?18.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
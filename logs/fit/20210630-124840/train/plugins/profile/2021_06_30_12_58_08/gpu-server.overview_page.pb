?	k?ѯz@k?ѯz@!k?ѯz@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCk?ѯz@!??^??e@1l_@/?3n@A?????N??I?q?P???rEagerKernelExecute 0*	??x?N??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?֤?!3@!I????X@)?֤?!3@1I????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??,`??!v n?V???)??,`??1v n?V???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismȲ`⏢??!a;????)?߽?Ƅ??1?$????:Preprocessing2F
Iterator::Model2??8*7??!&?Y??t??)?L?x$^n?1#???xΓ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?۞ ?!3@!????X@)??@??c?1??9??Ή?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI^~S??D@Q?????M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!??^??e@!??^??e@!!??^??e@      ??!       "	l_@/?3n@l_@/?3n@!l_@/?3n@*      ??!       2	?????N???????N??!?????N??:	?q?P????q?P???!?q?P???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q^~S??D@y?????M@?"l
>gradient_tape/model_45/conv2d_1220/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterLt}lβ?!Lt}lβ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamN??uS??!nK??I???"l
>gradient_tape/model_45/conv2d_1219/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterc?_???!??\?h???08"]
<gradient_tape/model_45/max_pooling2d_180/MaxPool/MaxPoolGradMaxPoolGrad?~͟Hx??!??V?q???"9
model_45/conv2d_1238/Conv2DConv2D???m????!??f$???0"9
model_45/conv2d_1241/Conv2DConv2DG9??)F??!?J??????0"9
model_45/conv2d_1232/Conv2DConv2D?<ˣ???!??i????0"j
>gradient_tape/model_45/conv2d_1226/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`\p????!???X??0"j
>gradient_tape/model_45/conv2d_1232/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter-.?????!??>??
??0"l
>gradient_tape/model_45/conv2d_1241/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter{? +ڈ??!@??"S??08I_?Z&?A@Q???l`P@Y?c5?25??a8?????X@q??M?Oe@y???`O?"?
both?Your program is POTENTIALLY input-bound because 41.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
?	
?g?x@
?g?x@!
?g?x@	???%ϒ?????%ϒ??!???%ϒ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL
?g?x@?b?T4?b@1|?q??n@Acb?qm???I???[ @YQf?L2r??rEagerKernelExecute 0*	?? ??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???BlA@!?𗸖?X@)???BlA@1?𗸖?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?s??q5??!b^????)?s??q5??1b^????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^?SH??!PC??*???)?[[%??1<(??ZN??:Preprocessing2F
Iterator::Model1??B?ʠ?!2y+?I??)q???imj?1?	?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?u?X?lA@!"5?m??X@)?t><K?a?1^??>-y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???%ϒ??I??(5AC@Q??e}x?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b?T4?b@?b?T4?b@!?b?T4?b@      ??!       "	|?q??n@|?q??n@!|?q??n@*      ??!       2	cb?qm???cb?qm???!cb?qm???:	???[ @???[ @!???[ @B      ??!       J	Qf?L2r??Qf?L2r??!Qf?L2r??R      ??!       Z	Qf?L2r??Qf?L2r??!Qf?L2r??b      ??!       JGPUY???%ϒ??b q??(5AC@y??e}x?N@?"i
;gradient_tape/model_3/conv2d_86/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?J??T߲?!?J??T߲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?ru*.??!Z??jߪ??"i
;gradient_tape/model_3/conv2d_85/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ke???!???b????08"6
model_3/conv2d_98/Conv2DConv2D\j"l2??!:c?????0"[
:gradient_tape/model_3/max_pooling2d_12/MaxPool/MaxPoolGradMaxPoolGrad??M ????!??l?V???"7
model_3/conv2d_104/Conv2DConv2D??i??`??!`3rr???0"7
model_3/conv2d_107/Conv2DConv2Dq_$R#Y??!N?^ܖ???0"g
;gradient_tape/model_3/conv2d_92/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??`^??!"v??]??0"g
;gradient_tape/model_3/conv2d_98/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?xpW?ʕ?!??9???0"8
model_3/conv2d_92/Conv2DConv2D [?*sZ??!D>\lDa??08Ik??5A@Qˑ?~?yP@Y?c5?25??a8?????X@q??3m,@y$?+bP?"?	
both?Your program is POTENTIALLY input-bound because 38.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?14.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
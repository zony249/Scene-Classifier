?	???x@???x@!???x@	???v*z????v*z?!???v*z?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???x@???<?c@1????spn@A5^?I??I?ݓ??Z??Ym?i?*???rEagerKernelExecute 0*	??nk??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorEdX?iC@!Ce???X@)EdX?iC@1Ce???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch1??B?ʐ?!Q???[???)1??B?ʐ?1Q???[???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`?o`r???!???T?h??)??~?{??1?????5??:Preprocessing2F
Iterator::Model膦?????!?k6??ҷ?)z?m?(n?1??A%R??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??^iC@!e??H?X@)??ډ?`?1?{???Ru?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???v*z?I?μXC@Q?܀ﮦN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???<?c@???<?c@!???<?c@      ??!       "	????spn@????spn@!????spn@*      ??!       2	5^?I??5^?I??!5^?I??:	?ݓ??Z???ݓ??Z??!?ݓ??Z??B      ??!       J	m?i?*???m?i?*???!m?i?*???R      ??!       Z	m?i?*???m?i?*???!m?i?*???b      ??!       JGPUY???v*z?b q?μXC@y?܀ﮦN@?"i
;gradient_tape/model_2/conv2d_59/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,Au?Ų?!,Au?Ų?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamG??#?7??!>ow~????"i
;gradient_tape/model_2/conv2d_58/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?&?????!b??????08"Z
9gradient_tape/model_2/max_pooling2d_8/MaxPool/MaxPoolGradMaxPoolGrad?/?~Z>??!X6] ????"6
model_2/conv2d_77/Conv2DConv2D??o?#??!X?YNU???0"6
model_2/conv2d_80/Conv2DConv2DH"z?x"??!?Il????0"6
model_2/conv2d_71/Conv2DConv2D.5A??!d????Y??0"g
;gradient_tape/model_2/conv2d_65/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??? ??!??G6???0"g
;gradient_tape/model_2/conv2d_71/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterm^???ƕ?!V?=?`???0"8
model_2/conv2d_65/Conv2DConv2D??u?zo??!B??rX0??08IG?B.A@Q?v\??hP@Y?c5?25??a8?????X@q	??J?(@y?w???O?"?	
both?Your program is POTENTIALLY input-bound because 38.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?12.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
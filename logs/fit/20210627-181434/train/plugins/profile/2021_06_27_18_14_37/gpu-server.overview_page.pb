?	Z?rL?'q@Z?rL?'q@!Z?rL?'q@	h?ܕW??h?ܕW??!h?ܕW??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLZ?rL?'q@??(?zV@1x~Q?~?f@A??k?ȝ?I????G??Yb??BW"??rEagerKernelExecute 0*	???x?g?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator? n/?2@!?5v?4?X@)? n/?2@1?5v?4?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?????^??!?aeAmY??)?????^??1?aeAmY??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismq???9??!?'>fT-??)??z0)??1???;??:Preprocessing2F
Iterator::Modeli7????!ҢZ?????){O崧?l?1n????)??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?,????2@!??#???X@)?@?vX?1J???u9??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 32.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9h?ܕW??I??n???@@Q??j?̯P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??(?zV@??(?zV@!??(?zV@      ??!       "	x~Q?~?f@x~Q?~?f@!x~Q?~?f@*      ??!       2	??k?ȝ???k?ȝ?!??k?ȝ?:	????G??????G??!????G??B      ??!       J	b??BW"??b??BW"??!b??BW"??R      ??!       Z	b??BW"??b??BW"??!b??BW"??b      ??!       JGPUYh?ܕW??b q??n???@@y??j?̯P@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdam??????!??????"h
<gradient_tape/model_4/conv2d_107/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?_}Ђ??!???G]N??0"f
;gradient_tape/model_4/conv2d_107/Conv2D/Conv2DBackpropInputConv2DBackpropInput?F????!?"&???0"9
model_4/conv2d_107/Conv2DConv2D?+eo???!Z?gyt???08"f
;gradient_tape/model_4/conv2d_113/Conv2D/Conv2DBackpropInputConv2DBackpropInput?c????!?Ø????0"7
model_4/conv2d_113/Conv2DConv2D?j?Mcؘ?!$qN#???0"0
Adam/gradients/AddN_31AddN?X*????!:???????"h
<gradient_tape/model_4/conv2d_113/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'?%????!?;???~??0"H
/gradient_tape/dense_12/kernel/Regularizer/Mul_1MulS?????!D}?????"j
<gradient_tape/model_4/conv2d_106/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter'c?????!?J@1}??08Is?y5??B@Q?f??O@Y??A?????aL?:,??X@q(?X??+@yp
?oR?"?	
both?Your program is POTENTIALLY input-bound because 32.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?13.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
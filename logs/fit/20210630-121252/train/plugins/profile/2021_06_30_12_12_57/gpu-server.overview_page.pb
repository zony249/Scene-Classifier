?	?? ?x@?? ?x@!?? ?x@	???F?ȍ????F?ȍ?!???F?ȍ?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?? ?x@??s???b@1B??	\on@A?D??]??I<?\? @YW??m??rEagerKernelExecute 0*	??S???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorE??|@@!]????X@)E??|@@1]????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchg?R@????!?AX4????)g?R@????1?AX4????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??/J?_??!?8nθ?)?????1($??ۧ?:Preprocessing2F
Iterator::ModelzT????!~?}??d??)?W?\Tk?1??)j$???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap˃?9|@@!? ??&?X@)?ù?Z?1??	??s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???F?ȍ?I?Ʀ?0C@Q??đ?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??s???b@??s???b@!??s???b@      ??!       "	B??	\on@B??	\on@!B??	\on@*      ??!       2	?D??]???D??]??!?D??]??:	<?\? @<?\? @!<?\? @B      ??!       J	W??m??W??m??!W??m??R      ??!       Z	W??m??W??m??!W??m??b      ??!       JGPUY???F?ȍ?b q?Ʀ?0C@y??đ?N@?"j
<gradient_tape/model_9/conv2d_248/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter;G?????!;G?????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?+]p???!:??u?E??"j
<gradient_tape/model_9/conv2d_247/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Ӡߏ??!?F/????08"[
:gradient_tape/model_9/max_pooling2d_36/MaxPool/MaxPoolGradMaxPoolGrad??V6???!?$͵f???"7
model_9/conv2d_266/Conv2DConv2D?H???0??!njЄr??0"7
model_9/conv2d_269/Conv2DConv2D???2?!??!???V?V??0"7
model_9/conv2d_260/Conv2DConv2D??,??m??!#9h
}$??0"h
<gradient_tape/model_9/conv2d_254/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?'{?Ε?!d+?9Q???0"h
<gradient_tape/model_9/conv2d_260/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}?m????!????????0"j
<gradient_tape/model_9/conv2d_269/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh??c`??!j?5???08I'}A@Qm???rsP@Y?c5?25??a8?????X@q=f?@y????d"O?"?
both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
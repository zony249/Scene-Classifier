?	?x@ٔ?^@?x@ٔ?^@!?x@ٔ?^@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?x@ٔ?^@N??;?@1?ѯ??]@A?1Xq????I<??????rEagerKernelExecute 0*	*??N??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????$@!?????X@)????$@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?C??{??!?C1??9??)?C??{??1?C1??9??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????K???!f?????)<i??
???16?P???:Preprocessing2F
Iterator::Model?+??f*??!??:??)?T?-??i?1z??R!??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??"?$@!????X@)]?@?"Y?1?3k?{@??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI`?G&C@Q%?͎?EX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N??;?@N??;?@!N??;?@      ??!       "	?ѯ??]@?ѯ??]@!?ѯ??]@*      ??!       2	?1Xq?????1Xq????!?1Xq????:	<??????<??????!<??????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`?G&C@y%?͎?EX@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdam??p?H??!??p?H??"0
Adam/gradients/AddN_31AddN$2Ʒ???!p??i????"H
/gradient_tape/dense_12/kernel/Regularizer/Mul_1Mul???M$P??!0?(}?T??"7
model_4/conv2d_107/Conv2DConv2D??Z?????!HB?T1??0"j
<gradient_tape/model_4/conv2d_107/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??x?VS??!?_?3????08"f
;gradient_tape/model_4/conv2d_107/Conv2D/Conv2DBackpropInputConv2DBackpropInput??Lq?6??!??́l???0"
mul_58Mul=?-????!???Qu??">
"dense_12/kernel/Regularizer/SquareSquare~?????!?S?????"F
-gradient_tape/dense_12/kernel/Regularizer/MulMul?G?/???! ?=????"I
/gradient_tape/model_4/dense_12/MatMul/Cast/CastCast????U`??!???7%??I??I?.C@@Q?
ۘh?P@Y&??~h???a??^??X@qCwghZ??y?w?"e?"?
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
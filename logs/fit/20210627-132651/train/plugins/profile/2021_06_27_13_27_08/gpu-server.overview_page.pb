?	??$x??q@??$x??q@!??$x??q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??$x??q@??|	?X@1??gp?f@A[_$??\??I????q_??rEagerKernelExecute 0*	??S???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator)?*??\3@!???"?X@))?*??\3@1???"?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchf??
???!?+j???)f??
???1?+j???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismz?,C???!???V?_??)ip[[x??1????ؠ??:Preprocessing2F
Iterator::Model??]ؚ???!T?%????)?PS?'l?1+%4#??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap? ?X4]3@!m???X@)??PN??`?16?YZz??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI4?u???A@Q?=?6)*P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??|	?X@??|	?X@!??|	?X@      ??!       "	??gp?f@??gp?f@!??gp?f@*      ??!       2	[_$??\??[_$??\??![_$??\??:	????q_??????q_??!????q_??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q4?u???A@y?=?6)*P@?"c
<cond_1/then/_10/cond_1/Adam/Adam/update_56/ResourceApplyAdamResourceApplyAdam??j?m??!??j?m??"h
<gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterns?oAx??!*|?>??0"f
;gradient_tape/model_5/conv2d_131/Conv2D/Conv2DBackpropInputConv2DBackpropInputj?ܮџ?!/?𳎙??0"9
model_5/conv2d_131/Conv2DConv2D???k????!?)a.???08"f
;gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropInputConv2DBackpropInputАE???!??
ɯ??0"7
model_5/conv2d_137/Conv2DConv2D???7???!2??????0"0
Adam/gradients/AddN_31AddN?c??ʘ?!??31!???"h
<gradient_tape/model_5/conv2d_137/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!?
Vb?|??0"H
/gradient_tape/dense_15/kernel/Regularizer/Mul_1Mul?.?*s???!??????"j
<gradient_tape/model_5/conv2d_130/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??~Ăٗ?!L??y??08Ij??o?B@Q??sO?O@Y??A?????aL?:,??X@qv?|??*@y˔???aR?"?	
both?Your program is POTENTIALLY input-bound because 34.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?13.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
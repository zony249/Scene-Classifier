?	?
Y?Hy@?
Y?Hy@!?
Y?Hy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?
Y?Hy@?3?[d@1rM???Ln@A?,`????IΪ??Vl??rEagerKernelExecute 0*	??K7as?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????3@!K.ޥ??X@)????3@1K.ޥ??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??B:<???!?I?????)??B:<???1?I?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism-??2:??!?I+??W??)????k??16?Ļe??:Preprocessing2F
Iterator::Model?j??P???!B?N1???)???2p?1??usT??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap35	ސ?3@!??X???X@)?,D??a?1?t?P/H??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???
D@Q*??i}?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?3?[d@?3?[d@!?3?[d@      ??!       "	rM???Ln@rM???Ln@!rM???Ln@*      ??!       2	?,`?????,`????!?,`????:	Ϊ??Vl??Ϊ??Vl??!Ϊ??Vl??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???
D@y*??i}?M@?"k
=gradient_tape/model_25/conv2d_680/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????)???!????)???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamZ????M??!?Y]~????"k
=gradient_tape/model_25/conv2d_679/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?{U?$??!???d???08"]
<gradient_tape/model_25/max_pooling2d_100/MaxPool/MaxPoolGradMaxPoolGradVԤ????!??x???"8
model_25/conv2d_698/Conv2DConv2D?0,?]m??!??A??0"8
model_25/conv2d_701/Conv2DConv2D?6?uld??!k??c????0"8
model_25/conv2d_692/Conv2DConv2D?Trzz???!?+?>???0"i
=gradient_tape/model_25/conv2d_686/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??N????!???9???0"i
=gradient_tape/model_25/conv2d_692/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(/[??ە?!ҕ`|X??0"k
=gradient_tape/model_25/conv2d_701/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterU2?????!??J??h??08I??rHA@Q?F~?[P@Y?c5?25??a8?????X@q????N8@y?CI?eQ?"?	
both?Your program is POTENTIALLY input-bound because 39.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?24.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
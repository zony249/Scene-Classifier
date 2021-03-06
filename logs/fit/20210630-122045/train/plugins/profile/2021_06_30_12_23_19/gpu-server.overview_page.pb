?	j????y@j????y@!j????y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCj????y@?7j?iDc@1??q?d?n@A.IIC??I??V????rEagerKernelExecute 0*	}?5^?^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator+?????B@!0?|?[?X@)+?????B@10?|?[?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchݗ3????!??0	???)ݗ3????1??0	???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm???{???!ˈ?)??Ɋ????1??T?{??:Preprocessing2F
Iterator::Model<?H??ڢ?!??????)?????n?1??ryH??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?S??B@!??@??X@)???2#b?1???D?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?j?5?yC@Q]??*?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?7j?iDc@?7j?iDc@!?7j?iDc@      ??!       "	??q?d?n@??q?d?n@!??q?d?n@*      ??!       2	.IIC??.IIC??!.IIC??:	??V??????V????!??V????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?j?5?yC@y]??*?N@?"j
<gradient_tape/model_4/conv2d_113/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ŀQoĲ?!?ŀQoĲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?d???)??!??'B玺?"j
<gradient_tape/model_4/conv2d_112/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0??dۛ?!?͕3????08"[
:gradient_tape/model_4/max_pooling2d_16/MaxPool/MaxPoolGradMaxPoolGrad?JT?????!5W`?????"7
model_4/conv2d_125/Conv2DConv2D?y:ɗ?!e??K????0"7
model_4/conv2d_131/Conv2DConv2De??0!??!???b!???0"7
model_4/conv2d_134/Conv2DConv2D?e	11 ??!r???'???0"h
<gradient_tape/model_4/conv2d_119/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??[?w??!?ei?D??0"h
<gradient_tape/model_4/conv2d_125/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?b???!????9???0"j
<gradient_tape/model_4/conv2d_134/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0J?We??!?]W@?D??08I?J?&A@Q}~?Z?lP@Y?c5?25??a8?????X@q??0??(@y?L ?O?"?	
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?12.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
?	}w+K?#y@}w+K?#y@!}w+K?#y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC}w+K?#y@H1@???c@1?m?8nn@A?2?FY???I?KTo,??rEagerKernelExecute 0*	!?rh"?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?????3@!?:^`?X@)?????3@1?:^`?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchI?0e???!???????)I?0e???1???????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismu?V??!3???ĉ??)W$&??[??1??Y+??:Preprocessing2F
Iterator::Model??%?<??!???6????)?n?;2Vk?1r{g??p??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap,??yp?3@!
?d??X@)f??(ϼ\?1"?.??U??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?wy6L?C@Q@??ɳBN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H1@???c@H1@???c@!H1@???c@      ??!       "	?m?8nn@?m?8nn@!?m?8nn@*      ??!       2	?2?FY????2?FY???!?2?FY???:	?KTo,???KTo,??!?KTo,??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?wy6L?C@y@??ɳBN@?"l
>gradient_tape/model_40/conv2d_1085/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter#???????!#???????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam??!J?B??!.Y?C???"l
>gradient_tape/model_40/conv2d_1084/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw???????!?ͨ1????08"]
<gradient_tape/model_40/max_pooling2d_160/MaxPool/MaxPoolGradMaxPoolGrad????@??!.?K0??"9
model_40/conv2d_1103/Conv2DConv2D?i?R???!`Ѣ????0"9
model_40/conv2d_1106/Conv2DConv2D<?r?i??!???HK???0"9
model_40/conv2d_1097/Conv2DConv2DM?		???!??	j̹??0"j
>gradient_tape/model_40/conv2d_1091/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?`*p?"??!??(~??0"j
>gradient_tape/model_40/conv2d_1097/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?nH??!?h?@???0"l
>gradient_tape/model_40/conv2d_1103/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_[	????!? ?ej??08I\y??:A@Q?wC??bP@Y?c5?25??a8?????X@q5x?cu8@yIh?*5 O?"?	
both?Your program is POTENTIALLY input-bound because 39.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?24.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
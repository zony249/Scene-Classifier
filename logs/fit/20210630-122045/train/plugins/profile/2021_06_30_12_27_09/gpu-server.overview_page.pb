?	]ݱ??y@]ݱ??y@!]ݱ??y@	???<?֋????<?֋?!???<?֋?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL]ݱ??y@T??$[?c@1?^?;n@A?i????I???NK @Y??հ߫?rEagerKernelExecute 0*	??ʡ???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator+N?f=@@!??~??X@)+N?f=@@1??~??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchӅX????!?T?K???)ӅX????1?T?K???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismF?T?=ϧ?!?ŵ?%L??)ʥ??$??1?l?g?R??:Preprocessing2F
Iterator::ModelU??-?|??!?2?U???)?0{?v?j?1?ն?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??S??=@@!g?1?4?X@)f??(ϼ\?1dQt|?v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???<?֋?I??????C@Q"L???1N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T??$[?c@T??$[?c@!T??$[?c@      ??!       "	?^?;n@?^?;n@!?^?;n@*      ??!       2	?i?????i????!?i????:	???NK @???NK @!???NK @B      ??!       J	??հ߫???հ߫?!??հ߫?R      ??!       Z	??հ߫???հ߫?!??հ߫?b      ??!       JGPUY???<?֋?b q??????C@y"L???1N@?"k
=gradient_tape/model_11/conv2d_302/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???L?Ĳ?!???L?Ĳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam???4????!???gb???"k
=gradient_tape/model_11/conv2d_301/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?<?$1ޛ?!	u3X????08"\
;gradient_tape/model_11/max_pooling2d_44/MaxPool/MaxPoolGradMaxPoolGradM~C?a??!?????"8
model_11/conv2d_320/Conv2DConv2D???a?U??!$|S?????0"8
model_11/conv2d_323/Conv2DConv2D?????R??!"?g? ???0"8
model_11/conv2d_314/Conv2DConv2D? qi??!@?j?N???0"i
=gradient_tape/model_11/conv2d_308/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????a*??!?????a??0"i
=gradient_tape/model_11/conv2d_314/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter8?m<????!0è?(??0"k
=gradient_tape/model_11/conv2d_323/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterӮ??}???!^?`Y??08IsSL?:A@QG?Y??bP@Y?c5?25??a8?????X@q(???l?/@y?]o???P?"?	
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
Refer to the TF2 Profiler FAQb?15.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
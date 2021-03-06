?	??z?]uy@??z?]uy@!??z?]uy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??z?]uy@[|
??=d@1?3?\n@A??????I*S?Aб@rEagerKernelExecute 0*	?&1Ш?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?ݓ??
A@!??d?0?X@)?ݓ??
A@1??d?0?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?T1?ϓ?!W}?Kz??)?T1?ϓ?1W}?Kz??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?`?ٟ?!?n?V??)v????1۞N?????:Preprocessing2F
Iterator::Model??+?z???!?????ڹ?)J??	?yk?1V~W??!??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?:???
A@!??L??X@)Lo.2^?1-?aE( v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?27N7/D@QF?ȱ??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[|
??=d@[|
??=d@![|
??=d@      ??!       "	?3?\n@?3?\n@!?3?\n@*      ??!       2	????????????!??????:	*S?Aб@*S?Aб@!*S?Aб@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?27N7/D@yF?ȱ??M@?"k
=gradient_tape/model_23/conv2d_626/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw()[@???!w()[@???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam3?em?P??!????x˺?"k
=gradient_tape/model_23/conv2d_625/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???NB??!tń???08"\
;gradient_tape/model_23/max_pooling2d_92/MaxPool/MaxPoolGradMaxPoolGrad?m???!1?G'????"8
model_23/conv2d_644/Conv2DConv2D?Ė9Fw??!?z?????0"8
model_23/conv2d_647/Conv2DConv2DRDJGIs??!Y?c,???0"8
model_23/conv2d_638/Conv2DConv2D?.(\??!/I'????0"i
=gradient_tape/model_23/conv2d_632/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterѷ???'??!)??T?W??0"i
=gradient_tape/model_23/conv2d_638/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?-Yh???!??????0"k
=gradient_tape/model_23/conv2d_644/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?&?=????!^e??V??08I(h?dM2A@Q?˴M?fP@Y?c5?25??a8?????X@q???6q?.@y?=?P?"?	
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?15.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
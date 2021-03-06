?	M,??$z@M,??$z@!M,??$z@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCM,??$z@????d@1???sn?o@An?|?b???I?
DO? @rEagerKernelExecute 0*	~?5^2#?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator2?F??;@!Nߧ???X@)2?F??;@1Nߧ???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???W?ё?!>1?????)???W?ё?1>1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismⱟ?R$??!n?a????)???ۂ???1`ڪ?????:Preprocessing2F
Iterator::Modell?p?握?!??Ej???)?w???o?1?!
mU???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap4J??%?;@!G?n??X@)2: 	?vb?1?(?ۘ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIhT???C@Q????!N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????d@????d@!????d@      ??!       "	???sn?o@???sn?o@!???sn?o@*      ??!       2	n?|?b???n?|?b???!n?|?b???:	?
DO? @?
DO? @!?
DO? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qhT???C@y????!N@?"k
=gradient_tape/model_26/conv2d_707/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!?l|?&??!!?l|?&??08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?R?????!?B?"???"k
=gradient_tape/model_26/conv2d_706/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?(J?ך?!>";+.??08"8
model_26/conv2d_728/Conv2DConv2D?2Z|H??!?h?:?o??0"8
model_26/conv2d_725/Conv2DConv2D??QLv"??!8?Pt??0"8
model_26/conv2d_719/Conv2DConv2Di?nLn??!?s>??a??0"k
=gradient_tape/model_26/conv2d_728/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteri??m??!?3f?O??08"]
<gradient_tape/model_26/max_pooling2d_104/MaxPool/MaxPoolGradMaxPoolGradj?
޵Z??!߇?!?:??"i
=gradient_tape/model_26/conv2d_713/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(?????!?^??i???0"i
=gradient_tape/model_26/conv2d_719/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????ؔ?!???Y?;??0I	??6_#A@Q|?dPnP@Y?c5?25??a8?????X@q9nS?u?3@yd_{'?N?"?	
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?19.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
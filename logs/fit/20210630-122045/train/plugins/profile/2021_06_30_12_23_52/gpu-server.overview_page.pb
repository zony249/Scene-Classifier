?	???*?x@???*?x@!???*?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???*?x@Ϊ??Vc@1<?ʃ?~n@A?4E?ӻ??I???s????rEagerKernelExecute 0*	?z????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?n?=A@!P????X@)?n?=A@1P????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??????!? ?????)??????1? ?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?R?1?#??!???XԵ?)s?,&6??1[?0???:Preprocessing2F
Iterator::ModelF??\???!??[-(B??)???<,?j?10`L?n??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??H??=A@!	??u??X@)?wak??b?1 ?:V?r{?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??<T?XC@QGë6?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ϊ??Vc@Ϊ??Vc@!Ϊ??Vc@      ??!       "	<?ʃ?~n@<?ʃ?~n@!<?ʃ?~n@*      ??!       2	?4E?ӻ???4E?ӻ??!?4E?ӻ??:	???s???????s????!???s????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??<T?XC@yGë6?N@?"j
<gradient_tape/model_5/conv2d_140/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterF?X?R²?!F?X?R²?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamt;c?0??!#?1CW???"j
<gradient_tape/model_5/conv2d_139/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=??f????!Y?l?????08"7
model_5/conv2d_152/Conv2DConv2DY?\????!rp????0"[
:gradient_tape/model_5/max_pooling2d_20/MaxPool/MaxPoolGradMaxPoolGradU??J]???!??o~???"7
model_5/conv2d_158/Conv2DConv2Dj2T}?]??!??4???0"7
model_5/conv2d_161/Conv2DConv2D??l?4??!???ʉ??0"h
<gradient_tape/model_5/conv2d_146/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterȒ?????!x??TIJ??0"h
<gradient_tape/model_5/conv2d_152/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Mf2ϕ?!U)????0"j
<gradient_tape/model_5/conv2d_161/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter3??ۉq??!hX?n0I??08In?٬BA@Qɠs??^P@Y?c5?25??a8?????X@qe????@y/j?8O?"?
both?Your program is POTENTIALLY input-bound because 38.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
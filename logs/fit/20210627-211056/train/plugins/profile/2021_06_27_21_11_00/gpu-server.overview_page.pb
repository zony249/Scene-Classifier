?	?3???|w@?3???|w@!?3???|w@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?3???|w@?ӝ'??b@1ByG??k@A??zM??I:??*???rEagerKernelExecute 0*	??n?G?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator{?%9`?3@!k??/?X@){?%9`?3@1k??/?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?=yX???!??l??)?=yX???1??l??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??h?x???!?d??????)?4'/2??1:??????:Preprocessing2F
Iterator::Model???cꮤ?!kΉ1?0??)?unڌ?p?1?N5rN??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapaS?Q??3@!;g???X@)???2#b?1??uwT???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIg{?MD@Q?????M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ӝ'??b@?ӝ'??b@!?ӝ'??b@      ??!       "	ByG??k@ByG??k@!ByG??k@*      ??!       2	??zM????zM??!??zM??:	:??*???:??*???!:??*???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qg{?MD@y?????M@?"j
<gradient_tape/model_7/conv2d_173/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(V??ߴ?!(V??ߴ?08"c
<cond_1/then/_10/cond_1/Adam/Adam/update_96/ResourceApplyAdamResourceApplyAdam?s2????! ???XU??"j
<gradient_tape/model_7/conv2d_172/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterx*??-??!>,O^p??08"[
:gradient_tape/model_7/max_pooling2d_28/MaxPool/MaxPoolGradMaxPoolGradG*?ް??!\?/ z???"7
model_7/conv2d_191/Conv2DConv2D? ?Ny??!????????0"7
model_7/conv2d_185/Conv2DConv2Dɒ"??|??!K??????0"h
<gradient_tape/model_7/conv2d_179/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter3?dR??!???ީ???0"h
<gradient_tape/model_7/conv2d_185/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?
??ʗ?!#6?{????0"j
<gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?j??y??!?|??W??08"f
;gradient_tape/model_7/conv2d_191/Conv2D/Conv2DBackpropInputConv2DBackpropInput??tPu^??!}ƭko???0I?i^yB@Q?"????O@YX?ܾ,??aÍM??X@q??\?Қ2@y0??a2N?"?	
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?18.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
?	Ҩ?Io@Ҩ?Io@!Ҩ?Io@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCҨ?Io@>??WX?@1?w?Rgn@A??8G??I?z0)>>??rEagerKernelExecute 0*	Yd;???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorQJVգ?@!??ʫ?X@)QJVգ?@1??ʫ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??	.VԐ?!??C????)??	.VԐ?1??C????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism'?y?3M??!?k,?E???)???!???11Ɍ$????:Preprocessing2F
Iterator::Model?,D????! ?f??);?/K;5g?1????Q??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?!S>??@!@ [???X@)#??fF?Z?1?p?t?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI ?~?<	??Q??ۃX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>??WX?@>??WX?@!>??WX?@      ??!       "	?w?Rgn@?w?Rgn@!?w?Rgn@*      ??!       2	??8G????8G??!??8G??:	?z0)>>???z0)>>??!?z0)>>??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?~?<	??y??ۃX@?"f
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????Բ?!?????Բ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?D????!???nk???"f
8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterN`ӳ|???!u??M%???08"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad????????!Ѱ?uU???"4
model/conv2d_26/Conv2DConv2D^?;?$u??!?-?????0"4
model/conv2d_23/Conv2DConv2D?????s??!r?.?l???0"4
model/conv2d_17/Conv2DConv2DT~,a?r??!<yTpƍ??0"e
9gradient_tape/model/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??뗘??!?Q??P??0"e
9gradient_tape/model/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?mT????!h<??X	??0"g
9gradient_tape/model/conv2d_26/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?;󕮜??!(p?#S??08I'??8?8A@Q쫔??cP@Y?c5?25??a8?????X@qz}?>?'??y??S:'O?"?
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
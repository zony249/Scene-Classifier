?	I?Ǵ6y@I?Ǵ6y@!I?Ǵ6y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCI?Ǵ6y@?}?mAc@1???d?n@A˅ʿ?W??I C??@rEagerKernelExecute 0*	ˡE??3?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?T????A@!??bo9?X@)?T????A@1??bo9?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?k?,	P??!(?5??h??)?k?,	P??1(?5??h??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?(	?????!??c?>d??)tys?V{??1??P?_??:Preprocessing2F
Iterator::Model=??tZ???!?????$??)?~?o?1????^??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapY?E???A@!?B_Ŷ?X@)?f??f?1??U?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI ?*???C@Q?i?9	{N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}?mAc@?}?mAc@!?}?mAc@      ??!       "	???d?n@???d?n@!???d?n@*      ??!       2	˅ʿ?W??˅ʿ?W??!˅ʿ?W??:	 C??@ C??@! C??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?*???C@y?i?9	{N@?"j
<gradient_tape/model_7/conv2d_194/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#?Ѳ?!?#?Ѳ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?)ux?M??!P?????"j
<gradient_tape/model_7/conv2d_193/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??z?d???!$??????08"7
model_7/conv2d_215/Conv2DConv2D??ٗ桘?!????????0"[
:gradient_tape/model_7/max_pooling2d_28/MaxPool/MaxPoolGradMaxPoolGrad?????!ה+?L???"7
model_7/conv2d_212/Conv2DConv2D?????H??!?|?a???0"7
model_7/conv2d_206/Conv2DConv2D??/?:J??!,p)?????0"h
<gradient_tape/model_7/conv2d_200/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterCj	????!t?*$U??0"h
<gradient_tape/model_7/conv2d_206/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??gڐȕ?!ʻ???0"j
<gradient_tape/model_7/conv2d_212/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????L??!??h??K??08I?f3??A@Q?L??
`P@Y?c5?25??a8?????X@q??-7??*@y)??5?O?"?	
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
Refer to the TF2 Profiler FAQb?13.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
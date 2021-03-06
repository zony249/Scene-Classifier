?	#?	???y@#?	???y@!#?	???y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC#?	???y@~?ƃ-d@1@?]o@A?????"??I@??T?@rEagerKernelExecute 0*	;?O????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?n?1?<@!D???-?X@)?n?1?<@1D???-?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??m??!0???;???)??m??10???;???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm???"??!?6k^c??)[|
????1|????Ϧ?:Preprocessing2F
Iterator::Model?	??bՠ?!?k?z??)?J???>l?1^y?ᢻ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap*?D/??<@!?eK??X@)\?J?`?16ƾ???|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?$????C@Qo?]9WN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~?ƃ-d@~?ƃ-d@!~?ƃ-d@      ??!       "	@?]o@@?]o@!@?]o@*      ??!       2	?????"???????"??!?????"??:	@??T?@@??T?@!@??T?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?$????C@yo?]9WN@?"j
<gradient_tape/model_6/conv2d_167/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???R???!???R???08"h
<gradient_tape/model_6/conv2d_173/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterج?a?*??!??z?????0"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?t?F???!x?1Uf???"j
<gradient_tape/model_6/conv2d_166/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?.e????!Xp~ӥ???08"[
:gradient_tape/model_6/max_pooling2d_24/MaxPool/MaxPoolGradMaxPoolGrad???C9F??!?$??l???"7
model_6/conv2d_185/Conv2DConv2D<4j?L???!Vk???b??0"7
model_6/conv2d_188/Conv2DConv2D?&?x飖?!(???7??0"7
model_6/conv2d_179/Conv2DConv2DՔ?~???!a!P?[v??0"h
<gradient_tape/model_6/conv2d_179/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??)?n8??!Y?b?????0"j
<gradient_tape/model_6/conv2d_188/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???+X???!?W"*??08I|??7?A@QB?8zd
P@Y?c5?25??a8?????X@qd??? 5@yu?U]??P?"?	
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?21.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
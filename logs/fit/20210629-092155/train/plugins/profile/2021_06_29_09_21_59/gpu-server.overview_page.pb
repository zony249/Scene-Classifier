?	??dV?y@??dV?y@!??dV?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??dV?y@b.??c@1?|	??n@A b???4??I.?ED1?@rEagerKernelExecute 0*	ʡE?3)?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generators??+?<@![??{H?X@)s??+?<@1[??{H?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9+?&?|??!'Uʽñ?)9+?&?|??1'Uʽñ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????碡?!?/N?_???)????????1r?Ɂ???:Preprocessing2F
Iterator::Model{Cr2q??!?eW????){O崧?l?1x????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapq?Ws??<@!M?????X@)??-YU?1?|???r?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI|????C@Q?	j?8N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b.??c@b.??c@!b.??c@      ??!       "	?|	??n@?|	??n@!?|	??n@*      ??!       2	 b???4?? b???4??! b???4??:	.?ED1?@.?ED1?@!.?ED1?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q|????C@y?	j?8N@?"k
=gradient_tape/model_12/conv2d_293/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?K??n+??!?K??n+??08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?14????!	X-rع?"k
=gradient_tape/model_12/conv2d_292/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=??J???!??f"L??08"\
;gradient_tape/model_12/max_pooling2d_48/MaxPool/MaxPoolGradMaxPoolGrad???N;j??!????i9??"8
model_12/conv2d_314/Conv2DConv2D???????!7¹k???0"8
model_12/conv2d_311/Conv2DConv2D$?<힖?!??Nk???0"8
model_12/conv2d_305/Conv2DConv2D???t?˕?!???!????0"i
=gradient_tape/model_12/conv2d_299/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? 6}3Y??!?]}?F??0"i
=gradient_tape/model_12/conv2d_305/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Dc??H??!5?T(?w??0"k
=gradient_tape/model_12/conv2d_311/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`??+Փ?!K???????08I?]GEA@Q
@wQ\]P@Yw??????aF?l?X@qb?5??2@y?? V?~T?"?	
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?18.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
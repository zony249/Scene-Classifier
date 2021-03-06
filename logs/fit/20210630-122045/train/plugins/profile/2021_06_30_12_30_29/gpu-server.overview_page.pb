?	???~??x@???~??x@!???~??x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???~??x@??O?Tc@1z??-n@A?c?~???Ii????@rEagerKernelExecute 0*	??~jtk?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?;????@@!?(?R}?X@)?;????@@1?(?R}?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????V%??!yT9?b~??)????V%??1yT9?b~??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism-&?(???!Fc?u??)?<Y????1rI???:Preprocessing2F
Iterator::Model`!sePm??!??˸?l??)??f??o?1ڬQ?m??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapA?;?@@!????X@)ޭ,?Yfa?1?%??y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?Kf{?C@Q?NN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??O?Tc@??O?Tc@!??O?Tc@      ??!       "	z??-n@z??-n@!z??-n@*      ??!       2	?c?~????c?~???!?c?~???:	i????@i????@!i????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Kf{?C@y?NN@?"k
=gradient_tape/model_17/conv2d_464/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter7VvW????!7VvW????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdamH?aq???!	?N??ź?"k
=gradient_tape/model_17/conv2d_463/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ڹ???!{p?????08"\
;gradient_tape/model_17/max_pooling2d_68/MaxPool/MaxPoolGradMaxPoolGrad?WV^Ә?!l??T0???"8
model_17/conv2d_482/Conv2DConv2D?c?>?G??!懥?$???0"8
model_17/conv2d_485/Conv2DConv2D!p??E??!굽=????0"8
model_17/conv2d_476/Conv2DConv2D?f???u??!?????0"i
=gradient_tape/model_17/conv2d_470/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb??׽&??!+???kb??0"i
=gradient_tape/model_17/conv2d_476/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter73C????!??????0"k
=gradient_tape/model_17/conv2d_485/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterHE׺[???!???	Z??08I{?>A@Q|?t?`P@Y?c5?25??a8?????X@q)??5?-@y^qmzkO?"?	
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
Refer to the TF2 Profiler FAQb?14.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
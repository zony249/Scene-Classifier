?	??S?y@??S?y@!??S?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??S?y@??????d@1??! ?En@A??gyܝ?I#?????rEagerKernelExecute 0*	V-???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??????>@!??L???X@)??????>@1??L???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?V?f???!J?Hb'??)?V?f???1J?Hb'??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??,g??!???????)??HLPÇ?1ǰv??E??:Preprocessing2F
Iterator::Model???sE)??!Y??orֻ?)????[o?1?F?@#o??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapn?HJ?>@!?dc
?X@)?l???_?1????ůy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIc6?Q҉D@Q??H?-vM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????d@??????d@!??????d@      ??!       "	??! ?En@??! ?En@!??! ?En@*      ??!       2	??gyܝ???gyܝ?!??gyܝ?:	#?????#?????!#?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qc6?Q҉D@y??H?-vM@?"k
=gradient_tape/model_16/conv2d_437/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterPU?J???!PU?J???08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam۹?}?@??!ǃ??Sź?"k
=gradient_tape/model_16/conv2d_436/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??!??!?d?????08"\
;gradient_tape/model_16/max_pooling2d_64/MaxPool/MaxPoolGradMaxPoolGradD??&????!??n:???"8
model_16/conv2d_455/Conv2DConv2Do?\?r??!l?F=???0"8
model_16/conv2d_458/Conv2DConv2D. -?e??!r	??????0"8
model_16/conv2d_449/Conv2DConv2D?xc07???!?x??޶??0"i
=gradient_tape/model_16/conv2d_443/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!???N?y??0"i
=gradient_tape/model_16/conv2d_449/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterC? ??Օ?!???pZ??0"k
=gradient_tape/model_16/conv2d_458/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????????!?u?rd??08I???0?>A@Q?&???`P@Y?c5?25??a8?????X@q?%9???@y?nh?MP?"?
both?Your program is POTENTIALLY input-bound because 40.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
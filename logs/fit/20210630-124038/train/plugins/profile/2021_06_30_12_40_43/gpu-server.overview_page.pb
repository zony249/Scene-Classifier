?	?HZ?x@?HZ?x@!?HZ?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?HZ?x@p?4(+c@1i??QU1n@A??>Ȳ`??I?$xC???rEagerKernelExecute 0*	?K7????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??[;Q??@!??q|??X@)??[;Q??@1??q|??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??辜??!??[4??)??辜??1??[4??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??]?????!???????)x?g?ɇ?1?9p?oâ?:Preprocessing2F
Iterator::Model7???N???!f???;??)???i?:m?1sWZY???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapf?-???@!?????X@)9CqǛ?V?1?A?""r?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?H????C@Q4?}InN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p?4(+c@p?4(+c@!p?4(+c@      ??!       "	i??QU1n@i??QU1n@!i??QU1n@*      ??!       2	??>Ȳ`????>Ȳ`??!??>Ȳ`??:	?$xC????$xC???!?$xC???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?H????C@y4?}InN@?"k
=gradient_tape/model_27/conv2d_734/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ňʉٲ?!?ňʉٲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?I?????! us???"k
=gradient_tape/model_27/conv2d_733/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterf??5'ޛ?!?`?i~???08"]
<gradient_tape/model_27/max_pooling2d_108/MaxPool/MaxPoolGradMaxPoolGrad????{\??!??????"8
model_27/conv2d_755/Conv2DConv2Dk#?vc??! ?g?o???0"8
model_27/conv2d_752/Conv2DConv2D;??v.\??!L9f????0"8
model_27/conv2d_746/Conv2DConv2D??GY?n??!Fb?ϩ??0"i
=gradient_tape/model_27/conv2d_740/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!o??V1??!????o??0"i
=gradient_tape/model_27/conv2d_746/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??٢???!;?&????0"k
=gradient_tape/model_27/conv2d_755/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????횔?!??UmFa??08I?9~?HA@Q7?@~?[P@Y?c5?25??a8?????X@q?Ǡ?.@yl???AfO?"?	
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?15.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
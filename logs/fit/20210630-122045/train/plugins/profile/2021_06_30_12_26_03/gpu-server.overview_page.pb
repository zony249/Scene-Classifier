?	?????*y@?????*y@!?????*y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?????*y@?aۢL?c@1?1??lNn@A?m??כ?I????{??rEagerKernelExecute 0*	???K??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??ME*??@!??ˁ??X@)??ME*??@1??ˁ??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??/?????!?{?p?T??)??/?????1?{?p?T??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???V`Ȣ?!t|?s?~??)$&??[X??1m@?T??:Preprocessing2F
Iterator::Model?>:u峤?!????bA??)???Q?n?1?;????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapޏ?/???@!'??N??X@)???i?:]?1?b<<?v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noILI2?C@Q?????N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?aۢL?c@?aۢL?c@!?aۢL?c@      ??!       "	?1??lNn@?1??lNn@!?1??lNn@*      ??!       2	?m??כ??m??כ?!?m??כ?:	????{??????{??!????{??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qLI2?C@y?????N@?"j
<gradient_tape/model_9/conv2d_248/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter|N??˲?!|N??˲?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam(?n??8??!F?w継??"j
<gradient_tape/model_9/conv2d_247/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??vh????!??? Z???08"[
:gradient_tape/model_9/max_pooling2d_36/MaxPool/MaxPoolGradMaxPoolGradY]?7n??!??????"7
model_9/conv2d_266/Conv2DConv2D?c+?&c??!?%ؤ???0"7
model_9/conv2d_269/Conv2DConv2D??Z?Y??!?{/?????0"7
model_9/conv2d_260/Conv2DConv2DDܱ/_??!"?%?????0"h
<gradient_tape/model_9/conv2d_254/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQdg?c??!??R,{??0"h
<gradient_tape/model_9/conv2d_260/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Vm?ӕ?!???\???0"j
<gradient_tape/model_9/conv2d_269/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter#qOm~??!???1?b??08I?I?[bQA@Q.[?NWP@Y?c5?25??a8?????X@qښ?́O1@y?a~??O?"?	
both?Your program is POTENTIALLY input-bound because 39.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?17.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
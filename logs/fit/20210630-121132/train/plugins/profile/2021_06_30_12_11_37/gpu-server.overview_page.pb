?	???՝?y@???՝?y@!???՝?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???՝?y@???`??c@1?DJ?yXn@A?3??O??I????Xg @rEagerKernelExecute 0*	??"?Y??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??ؙB;3@!???X@)??ؙB;3@1???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?>#K??!JѬ????)?>#K??1JѬ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?`???p??!?0????)p??-??1?J?^T??:Preprocessing2F
Iterator::Model???3ڪ??!F???????)????y?1u?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???`?;3@!;1????X@)??{?qY?1c?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIڤ????C@Q&[N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???`??c@???`??c@!???`??c@      ??!       "	?DJ?yXn@?DJ?yXn@!?DJ?yXn@*      ??!       2	?3??O???3??O??!?3??O??:	????Xg @????Xg @!????Xg @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qڤ????C@y&[N@?"j
<gradient_tape/model_7/conv2d_194/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterA#?????!A#?????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam??(:???!=Q*?t??"j
<gradient_tape/model_7/conv2d_193/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX`?q\ț?!???m???08"[
:gradient_tape/model_7/max_pooling2d_28/MaxPool/MaxPoolGradMaxPoolGrad2?'????!p??????"7
model_7/conv2d_212/Conv2DConv2D1"љY???!???J???0"7
model_7/conv2d_215/Conv2DConv2D???????!?1?L?m??0"7
model_7/conv2d_206/Conv2DConv2D?Q?V??!'?!E?8??0"h
<gradient_tape/model_7/conv2d_200/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter)s	
??!?]?s????0"h
<gradient_tape/model_7/conv2d_206/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?_ҝ??!(@??????0"9
model_7/conv2d_200/Conv2DConv2Dx]k??̔?! ?a??#??08I@0??5A@Q???{!eP@Y?c5?25??a8?????X@qdŐ??o6@ybP?{?P?"?	
both?Your program is POTENTIALLY input-bound because 39.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?22.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
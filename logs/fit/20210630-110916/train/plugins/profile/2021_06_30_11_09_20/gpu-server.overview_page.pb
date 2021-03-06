?	?|]??Uy@?|]??Uy@!?|]??Uy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?|]??Uy@??-d@1~?T??;n@A ??q???I?g?o}8 @rEagerKernelExecute 0*	?Q??G?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator ??L?<@!v6B??X@) ??L?<@1v6B??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?L???Ɣ?!?>?w????)?L???Ɣ?1?>?w????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-???!??Wۺ?)xe?????1?>O?s֡?:Preprocessing2F
Iterator::Model<??~K??!E9|l?۽?)&??|?k?1??R?(??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap:?%??<@!??$??X@)??????^?1x??8?z?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI,N?q?)D@QԱ??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??-d@??-d@!??-d@      ??!       "	~?T??;n@~?T??;n@!~?T??;n@*      ??!       2	 ??q??? ??q???! ??q???:	?g?o}8 @?g?o}8 @!?g?o}8 @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q,N?q?)D@yԱ??M@?"i
;gradient_tape/model_1/conv2d_32/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5=6,????!5=6,????08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam????K??!Zn?Yz??"i
;gradient_tape/model_1/conv2d_31/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??7H????!l(ڗ ???08"Z
9gradient_tape/model_1/max_pooling2d_4/MaxPool/MaxPoolGradMaxPoolGrad@???R??!?d5?T???"6
model_1/conv2d_50/Conv2DConv2D
??<??!?X>????0"6
model_1/conv2d_53/Conv2DConv2D???$??!P?I"z???0"6
model_1/conv2d_44/Conv2DConv2D}??2????!`??rR??0"g
;gradient_tape/model_1/conv2d_38/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0?*?????!?V?????0"g
;gradient_tape/model_1/conv2d_44/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}??n%???!?wE6????0"i
;gradient_tape/model_1/conv2d_50/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterߗ???[??!??Բ(??08I}????[A@QB? RP@Y?c5?25??a8?????X@q?\(݂-@y K@??xN?"?	
both?Your program is POTENTIALLY input-bound because 39.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?14.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 
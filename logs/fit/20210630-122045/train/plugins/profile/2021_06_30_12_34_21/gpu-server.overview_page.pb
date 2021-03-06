?	?Xz$y@?Xz$y@!?Xz$y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?Xz$y@?a????c@1-??\nOn@A??9?٘?I??U-?h??rEagerKernelExecute 0*	H?z^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?k????5@!p????X@)?k????5@1p????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????#ӑ?!?D?`㨴?)????#ӑ?1?D?`㨴?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismM?*?????!??>??3??)}%?????1ʈ?A|??:Preprocessing2F
Iterator::Model??ek}???!???4??)?uʣk?1ۧ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?Z|
??5@!@???e?X@)-?}?a?1;u?b桄?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIN?\c??C@Q???t#N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?a????c@?a????c@!?a????c@      ??!       "	-??\nOn@-??\nOn@!-??\nOn@*      ??!       2	??9?٘???9?٘?!??9?٘?:	??U-?h????U-?h??!??U-?h??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qN?\c??C@y???t#N@?"k
=gradient_tape/model_24/conv2d_653/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?zC?ڲ?!?zC?ڲ?08"d
=cond_1/then/_10/cond_1/Adam/Adam/update_108/ResourceApplyAdamResourceApplyAdam?)?µW??!v?)?鰺?"k
=gradient_tape/model_24/conv2d_652/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW??x??!?u?#???08"\
;gradient_tape/model_24/max_pooling2d_96/MaxPool/MaxPoolGradMaxPoolGradw?mZ<??!???7????"8
model_24/conv2d_671/Conv2DConv2D?D?t????!?^????0"8
model_24/conv2d_674/Conv2DConv2D??wH?V??!s?m????0"8
model_24/conv2d_665/Conv2DConv2D)???????!?C*MQ???0"i
=gradient_tape/model_24/conv2d_659/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ś??!O??ń_??0"i
=gradient_tape/model_24/conv2d_665/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterf8?????!.?????0"k
=gradient_tape/model_24/conv2d_674/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5&?N????!???X?X??08I??M??5A@Q??-eP@Y?c5?25??a8?????X@qh?d???@y?/??N?"?
both?Your program is POTENTIALLY input-bound because 39.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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
	?-???x@?-???x@!?-???x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?-???x@ۆQ?fc@12??Y@n@Ao?e?????I#K?X^ @rEagerKernelExecute 0*	?v???)?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?^?D?;@!h?g???X@)?^?D?;@1h?g???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch{???w???!?9?a?Ұ?){???w???1?9?a?Ұ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?=?N???!? ?y?b??)$??????1???/? ??:Preprocessing2F
Iterator::Model?Z(??ڡ?!]L? ??)\?J?p?1}?p?W???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?$????;@!?٤???X@)??ǘ??`?1??L??}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?dB?C@Q?Q???MN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ۆQ?fc@ۆQ?fc@!ۆQ?fc@      ??!       "	2??Y@n@2??Y@n@!2??Y@n@*      ??!       2	o?e?????o?e?????!o?e?????:	#K?X^ @#K?X^ @!#K?X^ @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?dB?C@y?Q???MN@
	????qy@????qy@!????qy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????qy@L7?A`fd@1T?Q??4n@Afh<?y??I????~?@rEagerKernelExecute 0*	???  ??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generators۾G?WA@!??c??X@)s۾G?WA@1??c??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ׂ?C??!?q9Lk??)?ׂ?C??1?q9Lk??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Đ?L??!z/??M`??)?d??1A???U??:Preprocessing2F
Iterator::ModelB?%U?M??!?I??z??)?fh<q?1=??҈?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?e6XA@!?-?[!?X@)?????\?1v*gQޏt?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI%???QD@Q?L?f ?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L7?A`fd@L7?A`fd@!L7?A`fd@      ??!       "	T?Q??4n@T?Q??4n@!T?Q??4n@*      ??!       2	fh<?y??fh<?y??!fh<?y??:	????~?@????~?@!????~?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q%???QD@y?L?f ?M@
	u?Bُy@u?Bُy@!u?Bُy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCu?Bُy@*s????d@1j?~?^;n@A??E}?;??I?B]?@rEagerKernelExecute 0*	?&1??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?[Ɏ??@!?N???X@)?[Ɏ??@1?N???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??L0?k??!?恻p:??)??L0?k??1?恻p:??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!?<??a??)? ??=@??1????N??:Preprocessing2F
Iterator::Model??;???!p~?|??)V??Dׅo?1;??҈?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapj??%??@!???? ?X@)??̔??b?1??T!?}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?t?[?nD@Q??=?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*s????d@*s????d@!*s????d@      ??!       "	j?~?^;n@j?~?^;n@!j?~?^;n@*      ??!       2	??E}?;????E}?;??!??E}?;??:	?B]?@?B]?@!?B]?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?t?[?nD@y??=?M@
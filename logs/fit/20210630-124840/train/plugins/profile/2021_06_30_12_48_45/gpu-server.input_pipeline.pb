	????o<y@????o<y@!????o<y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????o<y@s?}?p?c@1??<3n@A?????I̶?ֈ?@rEagerKernelExecute 0*	?|?5.?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???
?=@!T?K$??X@)???
?=@1T?K$??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch3?ۃ??!~)?8??)3?ۃ??1~)?8??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/?????!*iE??u??)??T????1֨?????:Preprocessing2F
Iterator::Model<?_?E??!j?t4???)??Z?p?1?1|19??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?lw??=@!?"??A?X@)m??)??b?1%?s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???p6D@Qd1???M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s?}?p?c@s?}?p?c@!s?}?p?c@      ??!       "	??<3n@??<3n@!??<3n@*      ??!       2	??????????!?????:	̶?ֈ?@̶?ֈ?@!̶?ֈ?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???p6D@yd1???M@
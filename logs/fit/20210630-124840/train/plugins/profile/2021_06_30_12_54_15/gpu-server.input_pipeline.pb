	?f?l?G{@?f?l?G{@!?f?l?G{@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?f?l?G{@jܛ߰?f@1? ????o@A?̒ 5???I`?eM,???rEagerKernelExecute 0*	C?l?C??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatort???F=@!??H?X@)t???F=@1??H?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?yq???!???!????)?yq???1???!????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??RB????!
l??'??)?"j??G??1?(
????:Preprocessing2F
Iterator::Model*?dq????!>X???)?܁:?q?1P?]P???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??K?[G=@!y?????X@)????&?a?1??}?K?~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIy???D@Q??]?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	jܛ߰?f@jܛ߰?f@!jܛ߰?f@      ??!       "	? ????o@? ????o@!? ????o@*      ??!       2	?̒ 5????̒ 5???!?̒ 5???:	`?eM,???`?eM,???!`?eM,???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qy???D@y??]?M@
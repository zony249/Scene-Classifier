	Q1?߄?n@Q1?߄?n@!Q1?߄?n@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCQ1?߄?n@?}?e??@1???Rn@A??r-Z???I/???u@rEagerKernelExecute 0*	A`??j??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?\?B@!I?^}?X@)?\?B@1I?^}?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?U??{??!$]?N????)?U??{??1$]?N????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??sCS??!?~???Ѿ?)??H?+??1?Cj??R??:Preprocessing2F
Iterator::Model????	???!lj??????)x?W?f,j?1??J????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapl$	?B@!K???X@)?'*?TV?1?????n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@? x????Q{}.I?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}?e??@?}?e??@!?}?e??@      ??!       "	???Rn@???Rn@!???Rn@*      ??!       2	??r-Z?????r-Z???!??r-Z???:	/???u@/???u@!/???u@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@? x????y{}.I?X@
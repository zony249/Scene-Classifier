	?x@ٔ?^@?x@ٔ?^@!?x@ٔ?^@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?x@ٔ?^@N??;?@1?ѯ??]@A?1Xq????I<??????rEagerKernelExecute 0*	*??N??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????$@!?????X@)????$@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?C??{??!?C1??9??)?C??{??1?C1??9??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????K???!f?????)<i??
???16?P???:Preprocessing2F
Iterator::Model?+??f*??!??:??)?T?-??i?1z??R!??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??"?$@!????X@)]?@?"Y?1?3k?{@??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI`?G&C@Q%?͎?EX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N??;?@N??;?@!N??;?@      ??!       "	?ѯ??]@?ѯ??]@!?ѯ??]@*      ??!       2	?1Xq?????1Xq????!?1Xq????:	<??????<??????!<??????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`?G&C@y%?͎?EX@
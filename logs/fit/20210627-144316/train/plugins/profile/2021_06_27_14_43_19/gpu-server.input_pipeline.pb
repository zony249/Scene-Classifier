	?`?,?q@?`?,?q@!?`?,?q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?`?,?q@d????kX@1??O=?f@AH?]?ۥ?IQ?|ar??rEagerKernelExecute 0*	C?l?K??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorpΈ?ޘ=@!??ٻZ?X@)pΈ?ޘ=@1??ٻZ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchd?? w??!??5c????)d?? w??1??5c????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismdyW=`??!???C???)?J̳?V??1L\?A|???:Preprocessing2F
Iterator::ModelལƄ???!??????)?GĔH?g?1Zl,
????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap5&?\R?=@!??2??X@)?8?Վ?\?121
I?]x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???"?A@Q????1P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d????kX@d????kX@!d????kX@      ??!       "	??O=?f@??O=?f@!??O=?f@*      ??!       2	H?]?ۥ?H?]?ۥ?!H?]?ۥ?:	Q?|ar??Q?|ar??!Q?|ar??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???"?A@y????1P@
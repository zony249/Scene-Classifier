	??S?y@??S?y@!??S?y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??S?y@??????d@1??! ?En@A??gyܝ?I#?????rEagerKernelExecute 0*	V-???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??????>@!??L???X@)??????>@1??L???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?V?f???!J?Hb'??)?V?f???1J?Hb'??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??,g??!???????)??HLPÇ?1ǰv??E??:Preprocessing2F
Iterator::Model???sE)??!Y??orֻ?)????[o?1?F?@#o??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapn?HJ?>@!?dc
?X@)?l???_?1????ůy?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIc6?Q҉D@Q??H?-vM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????d@??????d@!??????d@      ??!       "	??! ?En@??! ?En@!??! ?En@*      ??!       2	??gyܝ???gyܝ?!??gyܝ?:	#?????#?????!#?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qc6?Q҉D@y??H?-vM@
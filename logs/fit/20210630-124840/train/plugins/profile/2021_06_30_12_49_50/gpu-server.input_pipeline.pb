	?&c??x@?&c??x@!?&c??x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?&c??x@N`:?[nc@16?C6n@AV???4??I????? @rEagerKernelExecute 0*	bX9ds?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorQ?????@@!ў?bC?X@)Q?????@@1ў?bC?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???XǑ?!UO?u?b??)???XǑ?1UO?u?b??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?p???i??!????2ӵ?)N??oD??1ݟxU?C??:Preprocessing2F
Iterator::Model????=???!_/
???)ס????q?1.??#]??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?4?;?@@!?|?J??X@)X?%???c?1󝈷:}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??8??C@Qm??DN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N`:?[nc@N`:?[nc@!N`:?[nc@      ??!       "	6?C6n@6?C6n@!6?C6n@*      ??!       2	V???4??V???4??!V???4??:	????? @????? @!????? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??8??C@ym??DN@
	U???jRy@U???jRy@!U???jRy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCU???jRy@@1?d??c@1?Z'.G`n@A?Q??Z???I?9??@rEagerKernelExecute 0*	?? ?ʚ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??E|'3@!?-T??X@)??E|'3@1?-T??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch5???#b??!???:?O??)5???#b??1???:?O??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism1%??e??!ɽ?	??)Zd;?O???1,o?????:Preprocessing2F
Iterator::Model??ZӼ???!e4?3?i??)h??n?l?1??S?( ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap'?ei?3@!?f3K?X@)??Z
H?_?1?C?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI0????D@Q?5	i?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@1?d??c@@1?d??c@!@1?d??c@      ??!       "	?Z'.G`n@?Z'.G`n@!?Z'.G`n@*      ??!       2	?Q??Z????Q??Z???!?Q??Z???:	?9??@?9??@!?9??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q0????D@y?5	i?M@
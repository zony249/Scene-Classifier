	???2?oy@???2?oy@!???2?oy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???2?oy@ک??`ad@1Z	?%?Dn@A4?f??IvöE???rEagerKernelExecute 0*	??C+?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorퟧ??3@!%O????X@)ퟧ??3@1%O????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch¡?xxϑ?!g_3`?ʶ?)¡?xxϑ?1g_3`?ʶ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?ʄ_????!??l"??)&R???0??1R?ы????:Preprocessing2F
Iterator::Modelr?	?OƠ?!kx?Q1w??)st??%m?1Ͳ?#???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap}?q ?3@!?-WgD?X@)????[_?1???{m??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?ޞ???D@QB!a[$?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ک??`ad@ک??`ad@!ک??`ad@      ??!       "	Z	?%?Dn@Z	?%?Dn@!Z	?%?Dn@*      ??!       2	4?f??4?f??!4?f??:	vöE???vöE???!vöE???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ޞ???D@yB!a[$?M@
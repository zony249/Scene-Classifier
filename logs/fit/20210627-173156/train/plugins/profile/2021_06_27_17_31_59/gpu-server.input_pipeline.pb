	qU?w??o@qU?w??o@!qU?w??o@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCqU?w??o@??el??Q@1ę_́?f@Ap?DIH???I?W)???rEagerKernelExecute 0*	~j?t???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??_̖3@!%هd??X@)??_̖3@1%هd??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??C?b??!???@???)??C?b??1???@???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismc??Ց??!?ML?????)?[<?????1???3?y??:Preprocessing2F
Iterator::Model?KTol??!p?%?~???)?d??~?m?13??.fW??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???f?3@!	??@?X@)
?2?&W?1????7~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIx`~?Q<@Q??g`??Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??el??Q@??el??Q@!??el??Q@      ??!       "	ę_́?f@ę_́?f@!ę_́?f@*      ??!       2	p?DIH???p?DIH???!p?DIH???:	?W)????W)???!?W)???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qx`~?Q<@y??g`??Q@
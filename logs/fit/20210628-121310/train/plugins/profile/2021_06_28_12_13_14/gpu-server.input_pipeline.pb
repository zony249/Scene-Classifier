	^,?S?v@^,?S?v@!^,?S?v@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC^,?S?v@??:q?Ga@1=?)۴k@Amo?$???I??mē???rEagerKernelExecute 0*	??????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?+???A@!??????X@)?+???A@1??????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchA???FX??!?m??pŭ?)A???FX??1?m??pŭ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???B??!3???hB??)?}q?J[??1w?` a???:Preprocessing2F
Iterator::Modelu?BY???!b?O t»?)?n?;2Vk?1sI"X ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapQ0c
?A@!??b?X@)T8?T?]?1J???u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIC?? S`C@Q?g߬?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??:q?Ga@??:q?Ga@!??:q?Ga@      ??!       "	=?)۴k@=?)۴k@!=?)۴k@*      ??!       2	mo?$???mo?$???!mo?$???:	??mē?????mē???!??mē???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qC?? S`C@y?g߬?N@
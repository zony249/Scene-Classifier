	{K9_lgy@{K9_lgy@!{K9_lgy@	 ?L??p? ?L??p?! ?L??p?"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL{K9_lgy@H?S?d@1B?Ѫen@A)??????I?/K;5@Yi??Q???rEagerKernelExecute 0*	t????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Yf?	5@!n;?_??X@)?Yf?	5@1n;?_??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchc?#?w~??!?Xb?????)c?#?w~??1?Xb?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?[?tYL??!3-??????)Q??Û??1??+|???:Preprocessing2F
Iterator::ModelI?Vџ?!?ԗ?A???)?PS?'l?1?< ???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????
5@!4ߏ?X@)????^?1???O??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ?L??p?I?G!D@QzF??X?M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	H?S?d@H?S?d@!H?S?d@      ??!       "	B?Ѫen@B?Ѫen@!B?Ѫen@*      ??!       2	)??????)??????!)??????:	?/K;5@?/K;5@!?/K;5@B      ??!       J	i??Q???i??Q???!i??Q???R      ??!       Z	i??Q???i??Q???!i??Q???b      ??!       JGPUY ?L??p?b q?G!D@yzF??X?M@
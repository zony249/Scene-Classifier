	???,y@???,y@!???,y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???,y@4?f??c@1?????Mn@A)?7Ӆ??I???????rEagerKernelExecute 0*	?rh??t?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???v?2@!煱,r?X@)???v?2@1煱,r?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchS????g??!9ļ??X??)S????g??19ļ??X??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismjN^d~??!?Y^???).rOWw,??1(????T??:Preprocessing2F
Iterator::Modelq???h??!???????)|?ԗ??j?1?]?m????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap~U.T??2@!=??%?X@)˻??`?1AĊͪk??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?ۏ??C@Qn?$pN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4?f??c@4?f??c@!4?f??c@      ??!       "	?????Mn@?????Mn@!?????Mn@*      ??!       2	)?7Ӆ??)?7Ӆ??!)?7Ӆ??:	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ۏ??C@yn?$pN@
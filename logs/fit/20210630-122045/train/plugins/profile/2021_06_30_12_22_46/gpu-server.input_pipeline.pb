	
?g?x@
?g?x@!
?g?x@	???%ϒ?????%ϒ??!???%ϒ??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL
?g?x@?b?T4?b@1|?q??n@Acb?qm???I???[ @YQf?L2r??rEagerKernelExecute 0*	?? ??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???BlA@!?𗸖?X@)???BlA@1?𗸖?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?s??q5??!b^????)?s??q5??1b^????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^?SH??!PC??*???)?[[%??1<(??ZN??:Preprocessing2F
Iterator::Model1??B?ʠ?!2y+?I??)q???imj?1?	?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?u?X?lA@!"5?m??X@)?t><K?a?1^??>-y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???%ϒ??I??(5AC@Q??e}x?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b?T4?b@?b?T4?b@!?b?T4?b@      ??!       "	|?q??n@|?q??n@!|?q??n@*      ??!       2	cb?qm???cb?qm???!cb?qm???:	???[ @???[ @!???[ @B      ??!       J	Qf?L2r??Qf?L2r??!Qf?L2r??R      ??!       Z	Qf?L2r??Qf?L2r??!Qf?L2r??b      ??!       JGPUY???%ϒ??b q??(5AC@y??e}x?N@
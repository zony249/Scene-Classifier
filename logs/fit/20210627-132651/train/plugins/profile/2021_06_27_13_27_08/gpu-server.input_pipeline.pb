	??$x??q@??$x??q@!??$x??q@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??$x??q@??|	?X@1??gp?f@A[_$??\??I????q_??rEagerKernelExecute 0*	??S???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator)?*??\3@!???"?X@))?*??\3@1???"?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchf??
???!?+j???)f??
???1?+j???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismz?,C???!???V?_??)ip[[x??1????ؠ??:Preprocessing2F
Iterator::Model??]ؚ???!T?%????)?PS?'l?1+%4#??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap? ?X4]3@!m???X@)??PN??`?16?YZz??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI4?u???A@Q?=?6)*P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??|	?X@??|	?X@!??|	?X@      ??!       "	??gp?f@??gp?f@!??gp?f@*      ??!       2	[_$??\??[_$??\??![_$??\??:	????q_??????q_??!????q_??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q4?u???A@y?=?6)*P@
	?^~???y@?^~???y@!?^~???y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?^~???y@??^a??d@1|?ycn@A??Pn????Iޓ??Z???rEagerKernelExecute 0*	??Q0??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??O@?=@!rhG???X@)??O@?=@1rhG???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchsG?˵h??!>???Y??)sG?˵h??1>???Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism]p????!(Uo????)?Qf`??1=l?????:Preprocessing2F
Iterator::Model??D?֠?!?=??d??)+?)?Tp?1
Cϡn???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapyܝ??=@!??^???X@)3?&c`]?1?Oۅ?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI/???D@Q??Ws?cM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??^a??d@??^a??d@!??^a??d@      ??!       "	|?ycn@|?ycn@!|?ycn@*      ??!       2	??Pn??????Pn????!??Pn????:	ޓ??Z???ޓ??Z???!ޓ??Z???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q/???D@y??Ws?cM@
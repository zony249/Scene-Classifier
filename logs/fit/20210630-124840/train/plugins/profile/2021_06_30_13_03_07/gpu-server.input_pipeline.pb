	?R	?&y@?R	?&y@!?R	?&y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?R	?&y@E???c@18ݲC?1n@A?(?	0??IS>U???rEagerKernelExecute 0*	?x?&?%?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???	?;@!Rv?ô?X@)???	?;@1Rv?ô?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?5&?\??!\?#񛃰?)?5&?\??1\?#񛃰?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^?SH??!xoт?;??)I??Z?և?18\[#qp??:Preprocessing2F
Iterator::Modelm??oB??!Q?]e??)˻??p?1?V}׆|??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap8h???;@!???&=?X@)??UJ??b?1n?,CY??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIMص?N?C@Q?'JS?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E???c@E???c@!E???c@      ??!       "	8ݲC?1n@8ݲC?1n@!8ݲC?1n@*      ??!       2	?(?	0???(?	0??!?(?	0??:	S>U???S>U???!S>U???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qMص?N?C@y?'JS?N@
	?0?x@?0?x@!?0?x@	^?H+??^?H+??!^?H+??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?0?x@Ӣ>??b@1?'*?Sn@A`?o`r???I?M?G????YbK??z2??rEagerKernelExecute 0*	?A`?X??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????OR5@!?.?@?X@)????OR5@1?.?@?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??·g	??!??w???)??·g	??1??w???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-????!??$F???)???Hhˉ?1 ????1??:Preprocessing2F
Iterator::Modelm 6 B\??!????R??)0???DKn?1J?t?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap<?b??R5@!q????X@)????`?1^,ֽ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9^?H+??I?#?DJC@Qӗ?I??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ӣ>??b@Ӣ>??b@!Ӣ>??b@      ??!       "	?'*?Sn@?'*?Sn@!?'*?Sn@*      ??!       2	`?o`r???`?o`r???!`?o`r???:	?M?G?????M?G????!?M?G????B      ??!       J	bK??z2??bK??z2??!bK??z2??R      ??!       Z	bK??z2??bK??z2??!bK??z2??b      ??!       JGPUY^?H+??b q?#?DJC@yӗ?I??N@
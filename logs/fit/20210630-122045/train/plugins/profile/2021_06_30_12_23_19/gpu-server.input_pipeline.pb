	j????y@j????y@!j????y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCj????y@?7j?iDc@1??q?d?n@A.IIC??I??V????rEagerKernelExecute 0*	}?5^?^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator+?????B@!0?|?[?X@)+?????B@10?|?[?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchݗ3????!??0	???)ݗ3????1??0	???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm???{???!ˈ?)??Ɋ????1??T?{??:Preprocessing2F
Iterator::Model<?H??ڢ?!??????)?????n?1??ryH??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?S??B@!??@??X@)???2#b?1???D?x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?j?5?yC@Q]??*?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?7j?iDc@?7j?iDc@!?7j?iDc@      ??!       "	??q?d?n@??q?d?n@!??q?d?n@*      ??!       2	.IIC??.IIC??!.IIC??:	??V??????V????!??V????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?j?5?yC@y]??*?N@
	 
f?x@ 
f?x@! 
f?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC 
f?x@??q?c@1?B?l3n@A???g%???I???????rEagerKernelExecute 0*	???M??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??X??3@!?@????X@)??X??3@1?@????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchF$
-????!?F? ????)F$
-????1?F? ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!/}??,??)M?^?iN??1??k7:[??:Preprocessing2F
Iterator::Model?PN?????!n?V????)?`??o?1???ԵW??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap{??v??3@!ɹT?$?X@)CY??Z?Z?1<?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI9OH8?~C@Qǰ??u?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??q?c@??q?c@!??q?c@      ??!       "	?B?l3n@?B?l3n@!?B?l3n@*      ??!       2	???g%??????g%???!???g%???:	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q9OH8?~C@yǰ??u?N@
	?A	3?9y@?A	3?9y@!?A	3?9y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?A	3?9y@?ۻ}?c@1/???Bn@A?r/0+??IN?????rEagerKernelExecute 0*	w??B?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??? ??@@!?u?1,?X@)??? ??@@1?u?1,?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?iO?9???!???ʱ?)?iO?9???1???ʱ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ù_͡?!g???????)}Xo?
Ӈ?1?:8?e???:Preprocessing2F
Iterator::Model??sb???!*??%???)?<?E~?p?1FxC????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??=??@@!??????X@)?? @??]?1?qa=1v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?H?`?D@Q:?B??M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ۻ}?c@?ۻ}?c@!?ۻ}?c@      ??!       "	/???Bn@/???Bn@!/???Bn@*      ??!       2	?r/0+???r/0+??!?r/0+??:	N?????N?????!N?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?H?`?D@y:?B??M@
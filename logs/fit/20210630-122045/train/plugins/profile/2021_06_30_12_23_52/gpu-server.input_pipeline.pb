	???*?x@???*?x@!???*?x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???*?x@Ϊ??Vc@1<?ʃ?~n@A?4E?ӻ??I???s????rEagerKernelExecute 0*	?z????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?n?=A@!P????X@)?n?=A@1P????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??????!? ?????)??????1? ?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?R?1?#??!???XԵ?)s?,&6??1[?0???:Preprocessing2F
Iterator::ModelF??\???!??[-(B??)???<,?j?10`L?n??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??H??=A@!	??u??X@)?wak??b?1 ?:V?r{?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??<T?XC@QGë6?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ϊ??Vc@Ϊ??Vc@!Ϊ??Vc@      ??!       "	<?ʃ?~n@<?ʃ?~n@!<?ʃ?~n@*      ??!       2	?4E?ӻ???4E?ӻ??!?4E?ӻ??:	???s???????s????!???s????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??<T?XC@yGë6?N@
	9?????x@9?????x@!9?????x@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC9?????x@Z????Uc@1?J?8?<n@Ag???d??IAG?Z???rEagerKernelExecute 0*	(1??T?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????=@!?laTK?X@)????=@1?laTK?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??E}?;??!??Ç)???)O??Z}??1?Դn????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?D?[????!??A????)?D?[????1??A????:Preprocessing2F
Iterator::Model}?Жs)??!m?<r??)????n?1?ѻ4???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?	j??=@!Jx????X@)Y4???b?1I???N?~?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIa%????C@Q??>?[N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z????Uc@Z????Uc@!Z????Uc@      ??!       "	?J?8?<n@?J?8?<n@!?J?8?<n@*      ??!       2	g???d??g???d??!g???d??:	AG?Z???AG?Z???!AG?Z???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qa%????C@y??>?[N@
	???ۗy@???ۗy@!???ۗy@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???ۗy@?8?/?c@1?뤾?9n@A)??????I?y?~@rEagerKernelExecute 0*	?5^?i(?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatordx?g??3@!|????X@)dx?g??3@1|????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchǁW˝???!TBl??'??)ǁW˝???1TBl??'??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism{?%T??!?u-???)0?????1??}v??:Preprocessing2F
Iterator::Model?v???!?J?????)?t?? ?k?1??٤k\??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMape?3@!??.???X@)#??fF?Z?1]??be???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI3?u?C@Q???t?6N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8?/?c@?8?/?c@!?8?/?c@      ??!       "	?뤾?9n@?뤾?9n@!?뤾?9n@*      ??!       2	)??????)??????!)??????:	?y?~@?y?~@!?y?~@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q3?u?C@y???t?6N@
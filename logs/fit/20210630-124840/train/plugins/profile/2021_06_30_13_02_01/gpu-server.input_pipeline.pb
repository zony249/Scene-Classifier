	?i???{z@?i???{z@!?i???{z@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?i???{z@??O?m?f@1???K?n@AW@?ի?Iyxρ???rEagerKernelExecute 0*	?$????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatoro????I;@!:5??X@)o????I;@1:5??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?r??h???!?ֶ???)?r??h???1?ֶ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???Mb??!d5?d??)7?^?????1?? (???:Preprocessing2F
Iterator::Model??Ӻ??!??!P???)31]??o?11?~W???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??ׁsJ;@!?v?׽?X@)o??\??f?1u??Gץ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 42.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?j??E@Q;???yoL@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??O?m?f@??O?m?f@!??O?m?f@      ??!       "	???K?n@???K?n@!???K?n@*      ??!       2	W@?ի?W@?ի?!W@?ի?:	yxρ???yxρ???!yxρ???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?j??E@y;???yoL@
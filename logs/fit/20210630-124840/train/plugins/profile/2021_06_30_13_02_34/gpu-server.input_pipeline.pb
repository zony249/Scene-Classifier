	?'֩2,y@?'֩2,y@!?'֩2,y@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?'֩2,y@?ӀA??c@1?????Nn@A?(??=$??Iݴ?!???rEagerKernelExecute 0*	???x![?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorh?u???3@!?r?{D?X@)h?u???3@1?r?{D?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchf??CÒ?!?r<?r???)f??CÒ?1?r<?r???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/?????!?Y?~???)???|~??1??ț???:Preprocessing2F
Iterator::Model??[u???!???B?G??)?X???Fq?1??i߭ʕ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??߼8?3@!:????X@)??U?Z^?1j??aA???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 39.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?UEY?C@QV?溦N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ӀA??c@?ӀA??c@!?ӀA??c@      ??!       "	?????Nn@?????Nn@!?????Nn@*      ??!       2	?(??=$???(??=$??!?(??=$??:	ݴ?!???ݴ?!???!ݴ?!???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?UEY?C@yV?溦N@
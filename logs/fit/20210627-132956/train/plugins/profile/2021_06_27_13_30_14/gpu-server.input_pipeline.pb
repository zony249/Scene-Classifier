	?n?o?^g@?n?o?^g@!?n?o?^g@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?n?o?^g@?'c|?]@1??Q???f@A??j?=&??I#e???(??rEagerKernelExecute 0*	??"????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???Ft?2@!???X@)???Ft?2@1???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch#M?<i??!@Ъ????)#M?<i??1@Ъ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?8
??!???jW'??)?H?F?q??1?@?**b??:Preprocessing2F
Iterator::Model?p????!LS,m????)k?ѯ?o?1rM??ڔ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap3NCT??2@!?iɨ?X@)?'eRC[?17d_?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@?Ñ???Q?K???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'c|?]@?'c|?]@!?'c|?]@      ??!       "	??Q???f@??Q???f@!??Q???f@*      ??!       2	??j?=&????j?=&??!??j?=&??:	#e???(??#e???(??!#e???(??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@?Ñ???y?K???X@